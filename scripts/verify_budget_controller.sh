#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/control}"
mkdir -p "$OUT"
PY="${PWD}/.venv_bci/bin/python"
PYTHONPATH=. "$PY" scripts/build_budget_vectors.py --out "$OUT/controller_vectors.txt" --bins 256
iverilog -g2012 -s tb_budget_controller -o "$OUT/tb_budget.vvp" rtl/control/neuroedge_budget_controller.sv verification/rtl/control/tb_budget_controller.sv
vvp "$OUT/tb_budget.vvp" +VECTORS="$OUT/controller_vectors.txt"
