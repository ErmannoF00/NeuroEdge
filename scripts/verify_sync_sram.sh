#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
V="${1:?Usage: bash scripts/verify_sync_sram.sh <vector_dir>}"
command -v iverilog >/dev/null || { echo 'ERROR: install iverilog';exit 1; }
command -v vvp >/dev/null || { echo 'ERROR: install vvp';exit 1; }

for entry in 'all_weights.mem 44224' 'all_biases.mem 361' 'all_shifts.mem 361'; do
  read -r name expected <<< "$entry"
  test -s "$V/$name"
  count="$(wc -l < "$V/$name")"
  if ((count != expected)); then
    echo "ERROR: expected $expected rows in $V/$name, got $count"; exit 1
  fi
done
INPUTS="$(wc -l < "$V/input_frames.mem")"
if (( INPUTS % 512 != 0 )); then echo 'ERROR: 512 inputs per bin required';exit 1;fi
NF="$((INPUTS / 512))"
if (( NF < 8 || NF > 128 || NF % 4 != 0 )); then echo "ERROR: invalid frames=$NF";exit 1;fi
LOGITS="$(wc -l < "$V/expected_logits.mem")"
if (( LOGITS != (NF/4)*41 )); then echo "ERROR: invalid logits=$LOGITS";exit 1;fi
mkdir -p build/sync_sram

iverilog -g2012 -s tb_sync_sram_1rw \
  -o build/sync_sram/unit.vvp \
  rtl/speech/sync/neuroedge_sram_1rw.sv \
  verification/rtl/speech/tb_sync_sram_1rw.sv
vvp build/sync_sram/unit.vvp

iverilog -g2012 -s tb_full_decoder_sync \
  -Ptb_full_decoder_sync.NF="$NF" \
  -o build/sync_sram/full.vvp \
  rtl/speech/sync/neuroedge_sram_1rw.sv \
  rtl/speech/sync/neuroedge_speech_decoder.sv \
  verification/rtl/speech/tb_full_decoder_sync.sv
vvp build/sync_sram/full.vvp \
  "+WEIGHTS=$V/all_weights.mem" \
  "+BIASES=$V/all_biases.mem" \
  "+SHIFTS=$V/all_shifts.mem" \
  "+INPUTS=$V/input_frames.mem" \
  "+EXPECTED=$V/expected_logits.mem"
echo 'NEUROEDGE_SYNCHRONOUS_MEMORY_GATES_PASS'
