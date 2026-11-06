#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${NEUROEDGE_PYTHON:-python}
PYTHONPATH=. "$PYTHON" -m pytest -q verification/python/test_connected_stream.py verification/python/test_wire_receiver.py
NEUROEDGE_PYTHON="$PYTHON" bash scripts/verify_feature_packet.sh
PYTHONPATH=. "$PYTHON" scripts/run_connected_stream.py \
    --synthetic --compare --banks 64 --out build/connected_stream_report.json
PYTHONPATH=. "$PYTHON" - <<'PY'
import json
from pathlib import Path
p=Path('build/connected_stream_report.json')
r=json.loads(p.read_text())
assert r['provenance'].startswith('nonlearned')
assert set(r['mode_results'])=={'off','sham','supervised'}
for m in r['mode_results'].values():
    assert m['trials']==8 and m['bins']==32 and m['outputs']==8
    assert m['release_control'].startswith('explicit')
    assert all(t['released_phonemes'] is None for t in m['records'])
print('NEUROEDGE_CONNECTED_STREAM_REFERENCE_PASS')
PY
if [[ ${1:-} == --with-existing-rtl ]]; then
    # This is the verified, unchanged digital decoder/offset pipeline test,
    # NOT a claim that the new host policy runs in hardware.
    bash scripts/verify_public.sh
    echo 'NEUROEDGE_EXISTING_RTL_REGRESSION_PASS'
fi
