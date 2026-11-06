#!/usr/bin/env bash
set -euo pipefail
VECTORS="${1:?Usage: scripts/verify_affine_lane.sh path/to/affine_vectors.txt}"
test -s "$VECTORS"
command -v iverilog >/dev/null
command -v vvp >/dev/null
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$(mktemp)"
trap 'rm -f -- "$BIN"' EXIT
iverilog -g2012 -s tb_affine_lane -o "$BIN" \
  "$ROOT/rtl/speech/neuroedge_affine_lane.sv" \
  "$ROOT/verification/rtl/speech/tb_affine_lane.sv"
vvp "$BIN" "+VECTORS=$(realpath "$VECTORS")"
