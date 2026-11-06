#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/results/demo/vertical/rtl_vectors}"
if ! command -v iverilog >/dev/null 2>&1 || ! command -v vvp >/dev/null 2>&1; then
  echo 'RTL_NOT_RUN: Icarus Verilog missing. Ubuntu/WSL: sudo apt install iverilog' >&2
  exit 2
fi
OUT="$(realpath "$OUT")"
for f in weights.mem vectors.mem model_params.vh; do
  test -f "$OUT/$f" || { echo "MISSING_ABI_FILE: $OUT/$f"; exit 3; }
done
mkdir -p "$OUT/build"
(
 cd "$OUT"
 iverilog -g2012 -Wall -I . -s tb_neuroedge_classifier \
   -o build/tb.vvp \
   "$ROOT/rtl/vertical/neuroedge_linear_classifier.sv" \
   "$ROOT/verification/rtl/vertical/tb_neuroedge_classifier.sv"
 vvp build/tb.vvp
)
