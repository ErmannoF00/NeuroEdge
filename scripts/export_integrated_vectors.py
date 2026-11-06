#!/usr/bin/env python3
"""Generate independent Python INT8 oracle files for unchanged real existing vectors.
Missing input files cause a hard error; fixture provenance is recorded.
The selected committed coefficient must change an actual logit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from neuroedge.speech.fixed_decoder import (
    FixedParameters, FixedStreamingDecoder, TOTAL_META, TOTAL_WEIGHTS,
    write_hex, INPUTS, CLASSES,
)
from neuroedge.speech.offset_reference import apply_offsets


def load_hex(path: Path, expected_count: int, bits: int, dtype):
    if not path.is_file():
        raise FileNotFoundError(f"existing requires original WSL vector: {path}")
    raw = [int(v.strip(), 16) for v in path.read_text().splitlines() if v.strip()]
    if len(raw) != expected_count or any(v < 0 or v >= 2**bits for v in raw):
        raise ValueError(f"Invalid {path}: expected {expected_count} {bits}-bit values; got {len(raw)}")
    signed = np.asarray([v - (1 << bits) if v & (1 << (bits-1)) else v
                          for v in raw], dtype=dtype)
    return signed


def make_vectors(vectors: Path, out: Path):
    w=load_hex(vectors/'all_weights.mem',TOTAL_WEIGHTS,8,np.int8)
    b=load_hex(vectors/'all_biases.mem',TOTAL_META,32,np.int32)
    s=load_hex(vectors/'all_shifts.mem',TOTAL_META,8,np.int32)
    if np.any((s<0)|(s>31)):
        raise ValueError('bad shift')
    nwords=sum(1 for v in (vectors/'input_frames.mem').open() if v.strip())
    if nwords % INPUTS or nwords<INPUTS*8 or nwords//INPUTS%4:
        raise ValueError('Need >=8 complete bins and a multiple of 4')
    frames=load_hex(vectors/'input_frames.mem',nwords,8,np.int8).reshape(-1, INPUTS)
    expected=load_hex(vectors/'expected_logits.mem',frames.shape[0]//4*CLASSES,16,np.int16).reshape(-1,CLASSES)
    p=FixedParameters(w,b,s.astype(np.uint8))
    baseline=FixedStreamingDecoder(p).run(frames)
    if not np.array_equal(baseline,expected):
        raise AssertionError(f'existing source vector/oracle disagreement: {(baseline!=expected).sum()} logits')

    # Select based on a deterministic real-vector effect, not assumed nonzero
    # feature sensitivity. Prefer channels with high projection coupling.
    weight_strength=np.abs(w[:64*512].reshape(64,512).astype(np.int32)).sum(axis=0)
    ranked=sorted(range(INPUTS),key=lambda i:(-weight_strength[i],i))
    chosen=None
    out_values=None
    for channel in ranked:
        for offset in (-32768,32767,-128,127):
            offsets=np.zeros(INPUTS,dtype=np.int16)
            offsets[channel]=offset
            first=FixedStreamingDecoder(p).push(apply_offsets(frames[0],offsets))
            if not np.array_equal(first,baseline[0]):
                chosen=(channel,offset)
                out_values=offsets
                break
        if chosen:
            break
    if chosen is None:
        raise RuntimeError('No first-bin logit sensitivity found; DO NOT invent a changed output. Investigate this model.')

    calibrated=apply_offsets(frames,out_values)
    committed=FixedStreamingDecoder(p).run(calibrated)
    changed=int(np.count_nonzero(committed!=expected))
    if changed==0: raise AssertionError('Selected coefficient did not alter any logit')
    out.mkdir(parents=True,exist_ok=True)
    write_hex(out/'expected_commit.mem',committed,4)
    write_hex(out/'calibrated_commit.mem',calibrated,2)
    write_hex(out/'calibrated_zero.mem',frames,2)
    evidence={
        'source':('public synthetic fixture, NOT neural recording'
                  if (vectors/'fixture_manifest.json').is_file()
                  else 'caller-supplied external input vectors; provenance unverified'),
        'n_bins':len(frames),'baseline_logits':expected.size,
        'zero_offset_exact':True,'commit_channel':int(chosen[0]),
        'commit_offset_q4':int(chosen[1]),'changed_logits':changed,
        'commit_before_bin':0,'input_feature_changes':int(np.count_nonzero(calibrated!=frames)),
        'assumed_clk_hz':25_000_000,'bin_deadline_cycles':500_000,
    }
    (out/'oracle_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('ZERO_OFFSET_PY_ORACLE_PASS logits='+str(expected.size))
    print('COMMIT_PY_ORACLE_PASS changed_logits='+str(changed)+' channel='+str(chosen[0])+' offset='+str(chosen[1]))
    return evidence

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('vectors',type=Path)
    a.add_argument('output',type=Path)
    x=a.parse_args()
    make_vectors(x.vectors,x.output)
