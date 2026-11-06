#!/usr/bin/env python3
"""Private 32-bin host-RTL fixture from a local recorded attempted-speech trial.

No raw HDF5/participant data or trained weights are included in a release ZIP.
Writes to ignored build/ only. This is an OFFLINE numeric equivalence fixture,
not a trial PER or clinical assessment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder, write_rtl_vectors
from scripts.run_connected_stream import load_real_trials


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdf5',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--quant-profile',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('build/private_host_32'))
    args=ap.parse_args()
    p,trials=load_real_trials(hdf5=args.hdf5,checkpoint=args.checkpoint,
                              profile_path=args.quant_profile,
                              weights_dir=None,max_trials=1)
    x=trials[0][0]
    if x.shape[0]<32: raise ValueError('first trial has fewer than 32 bins')
    x=x[:32].copy()
    expected=FixedStreamingDecoder(p).run(x)
    if expected.shape!=(8,41): raise AssertionError('CTC output shape error')
    write_rtl_vectors(args.out,p,x,expected)
    sha={q.name:hashlib.sha256(q.read_bytes()).hexdigest()
         for q in sorted(args.out.glob('*.mem'))}
    provenance={'source':'private local attempted-speech recording; no participant data redistributed',
                'quantization':'frozen train-only normalization from checkpoint, Q4 INT8',
                'bins':32,'expected_logits':328,'trained_weights':44224,
                'sha256':sha,'contains_private_derived_data':True}
    (args.out/'private_fixture_evidence.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('NEUROEDGE_PRIVATE_HOST_VECTORS_READY bins=32 logits=328')
    print('WARNING: generated .mem files contain private derived features and trained weights; keep build/ private.')

if __name__=='__main__': main()
