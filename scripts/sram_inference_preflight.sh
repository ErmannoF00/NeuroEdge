#!/usr/bin/env bash
# Yosys front-end/memory-inference diagnostic only; NOT synthesis or STA signoff.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v yosys >/dev/null || { echo 'ERROR: yosys is required (sudo apt install yosys)';exit 1; }
mkdir -p results/physical
# Large unpacked memory can explode at tech mapping: STOP at memory_collect.
yosys -Q -T -l results/physical/yosys_memory_inference.log \
  -p 'read_verilog -sv rtl/speech/sync/neuroedge_sram_1rw.sv rtl/speech/sync/neuroedge_speech_decoder.sv; hierarchy -check -top neuroedge_speech_decoder; proc; opt; memory_dff; memory_collect; stat'
echo 'GENERIC_MEMORY_ELABORATION_PASS (no Liberty/STA/PPA/PDK signoff)'
