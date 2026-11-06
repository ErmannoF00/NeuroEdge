#!/usr/bin/env bash
# Uses public synthetic 32-bin fixtures. Does not read participant data.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${NEUROEDGE_PYTHON:-python3}"
command -v gcc >/dev/null || { echo 'ERROR: gcc required'; exit 1; }
command -v iverilog >/dev/null || { echo 'ERROR: iverilog required'; exit 1; }
command -v vvp >/dev/null || { echo 'ERROR: vvp required'; exit 1; }
OUT="${NEUROEDGE_HOST_BUILD:-build/host_adapter}"
mkdir -p "$OUT"
export PYTHONPATH=.:"${PYTHONPATH:-}"
"$PY" -m pytest -q verification/python/test_host_feedback_diagnostic.py verification/python/test_host_register_contract.py |
    tee "$OUT/python.log"
gcc -std=c11 -Wall -Wextra -Werror -pedantic -Ifirmware/include \
    firmware/src/neuroedge_host.c verification/c/test_host.c -o "$OUT/test_host"
"$OUT/test_host" | tee "$OUT/firmware.log"
grep -q 'NEUROEDGE_HOST_FIRMWARE_CONTRACT_PASS' "$OUT/firmware.log"
# Optional LOCAL HDF5-derived fixture; keep its derived .mem files private.
VECTORS="$OUT/vectors"
if [[ "${1:-}" == '--vectors' ]]; then
    [[ $# -eq 2 ]] || { echo 'Usage: verify_host_adapter.sh [--vectors DIR]'; exit 2; }
    VECTORS="$2"
    for F in all_weights.mem all_biases.mem all_shifts.mem input_frames.mem expected_logits.mem; do
        [[ -f "$VECTORS/$F" ]] || { echo "ERROR: missing vector file $VECTORS/$F"; exit 1; }
    done
elif [[ $# -gt 0 ]]; then
    echo 'Usage: verify_host_adapter.sh [--vectors DIR]'; exit 2
else
    "$PY" scripts/generate_public_vectors.py --out "$VECTORS"
fi
"$PY" scripts/export_integrated_vectors.py "$VECTORS" "$OUT/oracle" |
    tee "$OUT/oracle.log"
SRC=(rtl/control/neuroedge_offset_bank.sv
     rtl/speech/sync/neuroedge_sram_1rw.sv
     rtl/speech/sync/neuroedge_speech_decoder.sv
     rtl/soc/neuroedge_offset_stream_bridge.sv
     rtl/soc/neuroedge_speech_pipeline.sv
     rtl/soc/neuroedge_host_adapter.sv)
read -r CH OFF < <("$PY" - "$OUT/oracle/oracle_evidence.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]));print(v['commit_channel'],v['commit_offset_q4'])
PY
)
for C in 0 1; do
    iverilog -g2012 -s tb_host_adapter -Ptb_host_adapter.NF=32 \
        -Ptb_host_adapter.CASE_ID="$C" \
        -o "$OUT/host_${C}.vvp" "${SRC[@]}" \
        verification/rtl/soc/tb_host_adapter.sv 2> "$OUT/compile_${C}.log" || {
            cat "$OUT/compile_${C}.log"; exit 1;
        }
    if [[ "$C" == 0 ]]; then
        EXP="$VECTORS/expected_logits.mem"
    else
        EXP="$OUT/oracle/expected_commit.mem"
    fi
    vvp "$OUT/host_${C}.vvp" \
       "+WEIGHTS=$VECTORS/all_weights.mem" \
       "+BIASES=$VECTORS/all_biases.mem" \
       "+SHIFTS=$VECTORS/all_shifts.mem" \
       "+INPUTS=$VECTORS/input_frames.mem" \
       "+EXPECTED=$EXP" \
       "+BASELINE=$VECTORS/expected_logits.mem" \
       "+CHANNEL=$CH" "+OFFSET=$OFF" | tee "$OUT/host_${C}.log"
    grep -q 'NEUROEDGE_HOST_ADAPTER_RTL_PASS bins=32 logits=328' "$OUT/host_${C}.log"
done
grep -q 'HOST_ZERO_OFFSET_EXACT_PASS logits=328' "$OUT/host_0.log"
grep -q 'HOST_UNSAFE_UPDATE_REJECTED_PASS' "$OUT/host_0.log"
grep -q 'HOST_COMMITTED_OFFSET_EXACT_PASS logits=328' "$OUT/host_1.log"
echo 'NEUROEDGE_HOST_TO_RTL_REGRESSION_PASS'
