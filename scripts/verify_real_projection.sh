#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VECTORS="${1:?Usage: ./scripts/verify_real_projection.sh path/to/vectors}"
test -s "$VECTORS/weights.mem"
test -s "$VECTORS/features.mem"
test -s "$VECTORS/expected.mem"
for item in 'weights.mem 32768' 'features.mem 512' 'expected.mem 64'; do
  set -- $item
  count=$(wc -l < "$VECTORS/$1")
  if [[ "$count" != "$2" ]]; then echo "BAD_VECTOR_LENGTH $1 got=$count expected=$2"; exit 1; fi
done
mkdir -p build/speech_projection
iverilog -g2012 -s tb_projection -o build/speech_projection/tb.vvp \
  rtl/speech/neuroedge_projection.sv verification/rtl/speech/tb_projection.sv
vvp build/speech_projection/tb.vvp \
  "+WEIGHTS=$VECTORS/weights.mem" \
  "+FEATURES=$VECTORS/features.mem" \
  "+EXPECTED=$VECTORS/expected.mem"
