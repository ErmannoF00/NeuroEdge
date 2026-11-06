#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v iverilog >/dev/null || { echo 'IVERILOG_MISSING'; exit 1; }
command -v vvp >/dev/null || { echo 'VVP_MISSING'; exit 1; }
OUT="${1:-results/commit}"
mkdir -p "$OUT"
iverilog -g2012 -Wall -s tb_offset_commit -o "$OUT/offset_commit.vvp" \
 rtl/control/neuroedge_offset_bank.sv \
 verification/rtl/control/tb_offset_commit.sv
vvp "$OUT/offset_commit.vvp" | tee "$OUT/rtl.log"
grep -q 'OFFSET_COMMIT_RTL_PASS' "$OUT/rtl.log"
echo 'OFFSET_COMMIT_HARDWARE_VERIFIED_PASS'
