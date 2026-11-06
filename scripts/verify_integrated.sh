#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=".:${PYTHONPATH:-}"
V="${1:?Usage: bash scripts/verify_integrated.sh results/deployment/rtl32}"
command -v iverilog >/dev/null || { echo 'ERROR: iverilog missing';exit 1; }
command -v vvp >/dev/null || { echo 'ERROR: vvp missing';exit 1; }
PY="${NEUROEDGE_PYTHON:-.venv_bci/bin/python}"
if [[ ! -x "$PY" ]]; then PY=python3; fi
OUT="${NEUROEDGE_BUILD_DIR:-results/integration}"
mkdir -p "$OUT"
# Existing verification remains authoritative; fail closed on regressions.
"$PY" -m pytest -q verification/python | tee "$OUT/python.log"
grep -Eq '[0-9]+ passed' "$OUT/python.log"
echo 'OFFSET_ARITHMETIC_PY_PASS'
bash scripts/verify_offset_commit.sh "$OUT/offset_memory" | tee "$OUT/offset_memory_regression.log"
grep -q 'OFFSET_COMMIT_RTL_PASS' "$OUT/offset_memory_regression.log"
bash scripts/verify_sync_sram.sh "$V" | tee "$OUT/decoder_regression.log"
grep -q 'NEUROEDGE_SYNCHRONOUS_MEMORY_GATES_PASS' "$OUT/decoder_regression.log"
"$PY" scripts/export_integrated_vectors.py "$V" "$OUT/oracle" | tee "$OUT/oracle.log"
grep -q 'ZERO_OFFSET_PY_ORACLE_PASS' "$OUT/oracle.log"
grep -q 'COMMIT_PY_ORACLE_PASS' "$OUT/oracle.log"
NF="$(wc -l < "$V/input_frames.mem")"; NF=$((NF/512))
if [[ "$NF" -ne 32 ]]; then echo "ERROR: current acceptance requires exactly 32 bins (got $NF)"; exit 1;fi
read -r CH OFFSET < <("$PY" - "$OUT/oracle/oracle_evidence.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]));print(j['commit_channel'], j['commit_offset_q4'])
PY
)
SRC=(rtl/control/neuroedge_offset_bank.sv
     rtl/speech/sync/neuroedge_sram_1rw.sv
     rtl/speech/sync/neuroedge_speech_decoder.sv
     rtl/soc/neuroedge_offset_stream_bridge.sv
     rtl/soc/neuroedge_speech_pipeline.sv)
iverilog -g2012 -s tb_bridge_offset -o "$OUT/bridge.vvp" \
  "${SRC[@]}" verification/rtl/soc/tb_bridge_offset.sv
vvp "$OUT/bridge.vvp" | tee "$OUT/bridge.log"
iverilog -g2012 -s tb_integrated_reset -o "$OUT/reset.vvp" \
  "${SRC[@]}" verification/rtl/soc/tb_integrated_reset.sv
vvp "$OUT/reset.vvp" | tee "$OUT/reset.log"
grep -q 'RESET_RECOVERY_RTL_PASS' "$OUT/reset.log"
grep -q 'BRIDGE_UNIT_RTL_PASS' "$OUT/bridge.log"
for C in 0 1 2; do
  iverilog -g2012 -s tb_integrated_offset \
    -Ptb_integrated_offset.NF=32 \
    -Ptb_integrated_offset.CASE_ID="$C" \
    -o "$OUT/integrated_${C}.vvp" "${SRC[@]}" \
    verification/rtl/soc/tb_integrated_offset.sv
  if [[ "$C" == 1 ]]; then
    CAL="$OUT/oracle/calibrated_commit.mem"; EXP="$OUT/oracle/expected_commit.mem"
  else
    CAL="$OUT/oracle/calibrated_zero.mem"; EXP="$V/expected_logits.mem"
  fi
  vvp "$OUT/integrated_${C}.vvp" \
    "+WEIGHTS=$V/all_weights.mem" "+BIASES=$V/all_biases.mem" \
    "+SHIFTS=$V/all_shifts.mem" "+INPUTS=$V/input_frames.mem" \
    "+CALIBRATED=$CAL" "+EXPECTED=$EXP" \
    "+BASELINE=$V/expected_logits.mem" "+CHANNEL=$CH" "+OFFSET=$OFFSET" \
    | tee "$OUT/integrated_${C}.log"
  grep -q 'BRIDGE_STREAM_RTL_PASS bins=32 features=16384' "$OUT/integrated_${C}.log"
  grep -q 'LATENCY_ACCOUNTING_PASS' "$OUT/integrated_${C}.log"
done
grep -q 'ZERO_OFFSET_328_LOGITS_PASS logits=328' "$OUT/integrated_0.log"
grep -q 'COMMIT_CHANGES_LOGITS_PASS accepted_writes=1' "$OUT/integrated_1.log"
grep -q 'ABORT_DENY_PRESERVE_LOGITS_PASS' "$OUT/integrated_2.log"
grep -q 'NO_MID_BIN_WRITE_PASS' "$OUT/integrated_2.log"
echo 'NEUROEDGE_INTEGRATED_ALL_GATES_PASS'
