#!/usr/bin/env bash
# Standalone near-memory tile gate. No claim of complete banked projection RTL.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${NEUROEDGE_PYTHON:-python3}"
command -v iverilog >/dev/null || { echo 'ERROR: missing iverilog' >&2; exit 2; }
command -v vvp >/dev/null || { echo 'ERROR: missing vvp' >&2; exit 2; }
PYTHONPATH=. "$PYTHON" -m pytest -q \
  verification/python/test_near_memory_dot.py \
  verification/python/test_cim_bitplanes.py \
  verification/python/test_near_memory_projection.py
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
iverilog -g2012 -s tb_near_memory_dot -o "$TMP/memory_dot.vvp" \
  rtl/speech/sync/neuroedge_sram_1rw.sv \
  rtl/memory/neuroedge_near_memory_dot.sv \
  verification/rtl/memory/tb_near_memory_dot.sv
vvp "$TMP/memory_dot.vvp"
echo 'NEUROEDGE_STANDALONE_MEMORY_COMPUTE_GATE_PASS'
