#!/usr/bin/env bash
# Run the entire digital reference + RTL integration without human data.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${NEUROEDGE_PYTHON:-python3}"
command -v iverilog >/dev/null || { echo 'ERROR: install Icarus Verilog (iverilog)'; exit 1; }
command -v vvp >/dev/null || { echo 'ERROR: install Icarus Verilog (vvp)'; exit 1; }
export PYTHONPATH=".:${PYTHONPATH:-}"
"$PYTHON" scripts/generate_public_vectors.py --out build/public_vectors
NEUROEDGE_PYTHON="$PYTHON" NEUROEDGE_BUILD_DIR=build/integration \
  bash scripts/verify_integrated.sh build/public_vectors
printf '\nNEUROEDGE_PUBLIC_REGRESSION_PASS\n'
