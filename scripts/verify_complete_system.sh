#!/usr/bin/env bash
# Single extended private acceptance, without modifying previously verified gates.
set -euo pipefail
cd "$(dirname "$0")/.."
NEUROEDGE_PYTHON="${NEUROEDGE_PYTHON:-python3}" bash scripts/verify_private_acceptance.sh
NEUROEDGE_PYTHON="${NEUROEDGE_PYTHON:-python3}" bash scripts/verify_host_adapter.sh
PYTHONPATH=. "${NEUROEDGE_PYTHON:-python3}" scripts/diagnose_feedback_candidates.py \
    --synthetic --out build/host_adapter/feedback_diagnostic_synthetic.json
echo 'NEUROEDGE_COMPLETE_PRIVATE_SYSTEM_PASS'
