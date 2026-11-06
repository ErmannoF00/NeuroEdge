#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
V="${1:?Usage: ./scripts/verify_mmio.sh results/path/vector_dir}"
command -v iverilog >/dev/null || { echo 'ERROR: apt install iverilog'; exit 1; }
for entry in 'all_weights.mem 44224' 'all_biases.mem 361' 'all_shifts.mem 361'; do
  read -r name expected <<<"$entry"
  test -s "$V/$name"
  count="$(wc -l < "$V/$name")"
  if [[ "$count" != "$expected" ]]; then echo "BAD_VECTOR_LENGTH $name got=$count expected=$expected";exit 1;fi
done
INCOUNT="$(wc -l < "$V/input_frames.mem")"
if (( INCOUNT % 512 != 0 )); then echo "BAD_INPUT_SIZE=$INCOUNT";exit 1;fi
NF=$((INCOUNT/512))
OUTCOUNT="$(wc -l < "$V/expected_logits.mem")"
if (( NF < 8 || NF > 128 || NF % 4 != 0 || OUTCOUNT != NF/4*41 )); then
  echo "BAD_MMIO_VECTORS frames=$NF logits=$OUTCOUNT";exit 1
fi
mkdir -p build/mmio
iverilog -g2012 -s tb_neuroedge_mmio -Ptb_neuroedge_mmio.NF="$NF" \
  -o build/mmio/tb.vvp \
  rtl/speech/neuroedge_full_decoder.sv \
  rtl/soc/neuroedge_mmio_wrapper.sv \
  verification/rtl/soc/tb_neuroedge_mmio.sv
vvp build/mmio/tb.vvp \
  "+WEIGHTS=$V/all_weights.mem" \
  "+BIASES=$V/all_biases.mem" \
  "+SHIFTS=$V/all_shifts.mem" \
  "+INPUTS=$V/input_frames.mem" \
  "+EXPECTED=$V/expected_logits.mem"
