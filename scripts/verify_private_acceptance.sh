#!/usr/bin/env bash
# One private entrypoint for source, software, firmware, memory tile and RTL.
# Does NOT run a participant-data experiment or create a Git repository.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${NEUROEDGE_PYTHON:-python3}"
"$PYTHON" scripts/check_publication.py --ci
PYTHONPATH=. "$PYTHON" -m pytest -q verification/python
bash scripts/verify_firmware.sh
PYTHONPATH=. "$PYTHON" scripts/check_website.py
NEUROEDGE_PYTHON="$PYTHON" bash scripts/verify_feature_packet.sh
PYTHONPATH=. "$PYTHON" scripts/run_connected_stream.py --synthetic --compare --banks 16 \
  --out build/private_connected_stream_synthetic.json
PYTHONPATH=. "$PYTHON" scripts/run_closed_loop_replay.py \
  --out build/private_closed_loop_synthetic.json
NEUROEDGE_PYTHON="$PYTHON" bash scripts/verify_memory_compute.sh
NEUROEDGE_PYTHON="$PYTHON" bash scripts/verify_public.sh
echo 'NEUROEDGE_PRIVATE_ACCEPTANCE_PASS'
