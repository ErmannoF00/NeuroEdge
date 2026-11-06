#!/usr/bin/env python3
"""Deterministic *synthetic* 512-feature decoder stimulus; no human neural data.

Creates weight/metadata, a complete input stream, and the expected CTC logits
using the independently testable integer reference. The parameters are a sparse
functional fixture, NOT a learned speech decoder or an accuracy benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from neuroedge.speech.fixed_decoder import (
    FixedStreamingDecoder, INPUTS, CLASSES, TOTAL_WEIGHTS, TOTAL_META, write_rtl_vectors,
)

from neuroedge.speech.synthetic_fixture import synthetic_parameters, synthetic_features


def generate(output: Path, bins: int = 32) -> dict:
    if bins != 32:
        raise ValueError('full public integration regression requires 32 bins')
    p = synthetic_parameters()
    x = synthetic_features(bins)
    logits = FixedStreamingDecoder(p).run(x)
    if logits.shape != (8, CLASSES):
        raise AssertionError(f'unexpected output shape: {logits.shape}')
    if not np.count_nonzero(logits):
        raise AssertionError('synthetic stimulus must exercise output arithmetic')
    output.mkdir(parents=True, exist_ok=True)
    write_rtl_vectors(output, p, x, logits)
    paths = sorted(output.glob('*.mem'))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    evidence = {
        'provenance': 'deterministic synthetic data; no human recordings or learned weights',
        'bins': bins, 'features_per_bin': INPUTS, 'logits': int(logits.size),
        'weights': TOTAL_WEIGHTS, 'metadata_rows': TOTAL_META,
        'nonzero_logits': int(np.count_nonzero(logits)),
        'sha256': hashes,
    }
    (output / 'fixture_manifest.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'PUBLIC_SYNTHETIC_VECTORS_READY bins={bins} logits={logits.size}')
    return evidence


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('build/public_vectors'))
    args = ap.parse_args()
    generate(args.out)
