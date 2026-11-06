#!/usr/bin/env python3
"""Train an 8-feature EEG classifier and export an RTL-verifiable integer ABI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from neuroedge.vertical.dataset import load_features
from neuroedge.vertical.quantized_linear import IntegerLinear, X_SCALE, W_SCALE
from neuroedge.vertical.export import export_abi


def parse_ids(arg: str):
    ids = [int(s) for s in arg.split(',') if s.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Subject lists must be nonempty and unique')
    return ids


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['synthetic', 'physionet'], default='synthetic')
    p.add_argument('--train-subjects', default='1,2,3,4,5,6,7,8')
    p.add_argument('--val-subjects', default='9,10')
    p.add_argument('--test-subjects', default='11,12')
    p.add_argument('--cache', default='data/physionet')
    p.add_argument('--out', default='results/demo/vertical')
    p.add_argument('--vectors', type=int, default=32)
    args = p.parse_args(argv)
    groups = {n: parse_ids(getattr(args, n+'_subjects')) for n in ('train','val','test')}
    if not (set(groups['train']).isdisjoint(groups['val']) and
            set(groups['train']).isdisjoint(groups['test']) and
            set(groups['val']).isdisjoint(groups['test'])):
        raise ValueError('Subject leakage: train/validation/test IDs must be disjoint')
    sets = {part: load_features(args.dataset, subjects, args.cache)
            for part, subjects in groups.items()}
    train = sets['train']
    mean = train.X.mean(0)
    std = train.X.std(0)
    std[std < 1e-8] = 1.0
    ztrain = (train.X-mean)/std
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
    clf.fit(ztrain, train.y)
    # The exporter expects exactly the same train-only normalization.
    integer_model = IntegerLinear(
        np.clip(np.rint(clf.coef_[0]*W_SCALE),-128,127).astype(np.int8),
        int(np.rint(clf.intercept_[0]*X_SCALE*W_SCALE)), mean, std)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    result = {
        'dataset': sets['train'].provenance,
        'scientific_results_valid': args.dataset == 'physionet',
        'disclaimer': 'CI DEMO ONLY: synthetic labels, not neural decoding'
                      if args.dataset == 'synthetic' else
                      'Real EEG; limited subject subset, not publication-ready benchmark',
        'subjects': groups, 'seed': 42,
        'software_versions':{'numpy':np.__version__,'sklearn':sklearn.__version__},
        'source_file_hashes':{p: datasets.source_files for p,datasets in sets.items()},
        'preprocessing': '8 channels; 8-30Hz offline IIR; average-reference; log bandpower',
        'scope_of_rtl': 'quantized linear classification of precomputed features ONLY',
        'metrics': {},
    }
    for part, dataset in sets.items():
        z = (dataset.X-mean)/std
        yf = clf.predict(z)
        xq = integer_model.quantize_inputs(dataset.X)
        yi = integer_model.predict(xq)
        result['metrics'][part] = {
            'n': len(dataset.y), 'n_subjects': len(groups[part]),
            'float_accuracy': float(accuracy_score(dataset.y,yf)),
            'int_accuracy': float(accuracy_score(dataset.y,yi)),
            'int_balanced_accuracy': float(balanced_accuracy_score(dataset.y, yi)),
            'float_to_int_disagreement': float(np.mean(yf != yi)),
            'confusion_matrix_int': confusion_matrix(dataset.y,yi,labels=[0,1]).tolist(),
        }
    export = export_abi(integer_model, sets['test'].X, str(root/'rtl_vectors'),
                        max_vectors=args.vectors)
    result['abi'] = {k: export[k] for k in ['abi','input_features','bias','test_vector_count']}
    (root/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"DATA={result['dataset']} RESEARCH_VALID={result['scientific_results_valid']}")
    for name, m in result['metrics'].items():
        print(f"{name.upper()}: {m['n']} epochs float_acc={m['float_accuracy']:.3f} int_acc={m['int_accuracy']:.3f}")
    print(f"EXPORT_PASS: {root/'rtl_vectors'}  VECTORS={export['test_vector_count']}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
