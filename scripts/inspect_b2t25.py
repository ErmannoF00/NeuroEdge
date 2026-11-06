#!/usr/bin/env python3
"""Check a local, official Brain-to-Text '25 HDF5 recording (no model inference)."""
import argparse
import json
from neuroedge.datasets.b2t25_contract import inspect_file, BIN_MS, N_FEATURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', help='Official data_train.hdf5/data_val.hdf5/data_test.hdf5')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    trials = list(inspect_file(args.file, limit=args.limit))
    report = {
      'contract': 'brain_to_text25', 'n_features': N_FEATURES,
      'bin_ms': BIN_MS, 'trials_inspected': len(trials),
      'duration_s_inspected': round(sum(t.n_time_steps for t in trials) * BIN_MS / 1000, 3),
      'trials': [t.__dict__ for t in trials],
      'note': 'INPUT_SCHEMA_PASS is not neural decoding, trained accuracy, or ASIC inference.'
    }
    print(json.dumps(report, indent=2))
    if trials: print('SPEECH_HDF5_CONTRACT_PASS')
    else: raise SystemExit('No trial groups found')


if __name__ == '__main__':
    main()
