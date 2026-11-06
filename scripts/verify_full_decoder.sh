#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
V="${1:?Usage: ./scripts/verify_full_decoder.sh results/path/vector_dir}"
command -v iverilog >/dev/null || { echo 'ERROR: install Icarus Verilog (sudo apt install iverilog)'; exit 1; }
for entry in 'all_weights.mem 44224' 'all_biases.mem 361' 'all_shifts.mem 361'; do
  read -r name expected <<<"$entry"
  test -s "$V/$name"
  count="$(wc -l < "$V/$name")"
  if [[ "$count" != "$expected" ]]; then echo "BAD_VECTOR_LENGTH $name got=$count expected=$expected";exit 1;fi
done
INCOUNT="$(wc -l < "$V/input_frames.mem")"
if (( INCOUNT % 512 != 0 )); then echo "BAD_INPUT_SIZE=$INCOUNT";exit 1; fi
NF=$((INCOUNT / 512))
if (( NF < 8 || NF > 128 || NF % 4 != 0 )); then echo "BAD_FRAME_COUNT=$NF";exit 1;fi
OUTCOUNT="$(wc -l < "$V/expected_logits.mem")"
EXPECTED=$(( NF / 4 * 41 ))
if (( OUTCOUNT != EXPECTED )); then echo "BAD_LOGIT_COUNT got=$OUTCOUNT expected=$EXPECTED";exit 1;fi
mkdir -p build/speech_full_decoder
iverilog -g2012 -s tb_full_decoder -Ptb_full_decoder.NF="$NF" \
  -o build/speech_full_decoder/tb.vvp \
  rtl/speech/neuroedge_full_decoder.sv verification/rtl/speech/tb_full_decoder.sv
vvp build/speech_full_decoder/tb.vvp \
  "+WEIGHTS=$V/all_weights.mem" \
  "+BIASES=$V/all_biases.mem" \
  "+SHIFTS=$V/all_shifts.mem" \
  "+INPUTS=$V/input_frames.mem" \
  "+EXPECTED=$V/expected_logits.mem"
