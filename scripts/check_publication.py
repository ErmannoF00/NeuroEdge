#!/usr/bin/env python3
"""Pre-commit checks for private artifacts, obsolete labels, and stale public claims.

A passing result cannot establish rights clearance or absence of all secrets.
Inspect the Git index, the entire history, and contractual IP ownership separately.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOW_DEMO = Path('results/demo/vertical')
BAD_SUFFIX = {'.hdf5', '.h5', '.edf', '.fif', '.pkl', '.pickle', '.pt', '.pth',
              '.ckpt', '.npz', '.npy', '.onnx', '.vcd', '.fst', '.vvp', '.pem', '.key',
              '.gds', '.gdsii', '.p12'}
SOURCE_SUFFIXES = {'.py', '.sv', '.v', '.c', '.h', '.sh', '.md', '.json', '.html',
                   '.js', '.css', '.yml', '.yaml', '.cff', '.sdc'}
RELEASE_CODES = re.compile(r'(?i)(?:\br\d+\.\d+\b|_r\d+(?=\b|\W)|\bR\d+_)')


def is_local(rel: Path) -> bool:
    parts = rel.parts
    return any(x in parts for x in ('.git', 'build', 'dist', '__pycache__',
              '.pytest_cache', '.mypy_cache', '.venv', '.venv_bci')) or any(
              x.startswith('.venv') for x in parts)


def check_paths(paths: list[Path]) -> list[str]:
    errors = []
    for rel in paths:
        if is_local(rel):
            continue
        if rel.parts and rel.parts[0] == 'results' and not (
                rel == Path('results/.gitkeep') or
                rel == Path('results/demo/vertical/metrics.json') or
                (str(rel).startswith(str(ALLOW_DEMO) + '/') and
                 rel.name in ('model.json', 'model_params.vh', 'manifest.sha256',
                              'vectors.mem', 'weights.mem'))):
            errors.append(f'NONPUBLIC RESULTS: {rel}')
        if rel.parts and rel.parts[0] == 'data' and rel != Path('data/README.md'):
            errors.append(f'PARTICIPANT DATA: {rel}')
        if rel.suffix.lower() in BAD_SUFFIX or rel.name.startswith('.env') and rel.name != '.env.example':
            errors.append(f'RESTRICTED ARTIFACT: {rel}')
        if RELEASE_CODES.search(str(rel)):
            errors.append(f'RELEASE-NUMBER PATH: {rel}')
        p = ROOT / rel
        if p.exists() and p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES:
            try:
                data = p.read_text(encoding='utf-8')
            except UnicodeError:
                errors.append(f'NONUTF8 SOURCE: {rel}')
                continue
            match = RELEASE_CODES.search(data)
            if match:
                errors.append(f'RELEASE-NUMBER SOURCE: {rel} token={match.group(0)}')
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ci', action='store_true', help='source-only checks for cloned CI runner')
    args = ap.parse_args()
    paths = [p.relative_to(ROOT) for p in ROOT.rglob('*') if p.is_file() and not is_local(p.relative_to(ROOT))]
    errors = check_paths(paths)
    if not (ROOT/'website/evidence.json').exists(): errors.append('NO WEBSITE EVIDENCE')
    if not (ROOT/'scripts/verify_public.sh').exists(): errors.append('NO PUBLIC RTL RUNNER')
    git = ROOT/'.git'
    if not args.ci and git.exists():
        result = subprocess.run(['git','ls-files','-z'], cwd=ROOT, capture_output=True, check=True)
        tracked = [Path(v.decode()) for v in result.stdout.split(b'\0') if v]
        errors.extend(check_paths(tracked))
        print(f'GIT_TRACKED_PATHS={len(tracked)}')
        print('Review all historical blobs and any existing remote separately.')
    elif not args.ci:
        print('NO_GIT_HISTORY: create a fresh repository only after acceptance checks.')
    print(f'CHECKED_SOURCE_FILES={len(paths)}')
    print('RIGHTS_REVIEW: human approval still required for employer IP, datasets, images and third-party code.')
    if errors:
        for e in errors: print('BLOCKER:',e)
        raise SystemExit(f'PUBLICATION_SOURCE_AUDIT_BLOCKED count={len(errors)}')
    print('PUBLICATION_SOURCE_AUDIT_PASS')


if __name__ == '__main__': main()
