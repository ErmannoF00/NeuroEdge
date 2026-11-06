#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VECTORS="${1:?Pass real-recording vectors directory}"
PY="$PWD/.venv_bci/bin/python"
test -s "$VECTORS/contract.json"
test -s "$VECTORS/features.mem"
test -s "$VECTORS/expected.txt"
read -r LOW HIGH MAG FRAMES < <("$PY" - "$VECTORS/contract.json" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
import re
contract=c['contract'].strip().lower()
contract=re.sub(r'(?i)r\d+(?:\.\d+)?-', '', contract)
assert contract=='neuroedge-int8-excursion-activity'
assert 0<=c['low_threshold']<c['high_threshold']<=512
assert 1<=c['magnitude_threshold']<=128
assert 1<=c['frames']<=2048
assert c['features_per_frame']==512
print(c['low_threshold'],c['high_threshold'],c['magnitude_threshold'],c['frames'])
PY
)
if (( $(wc -l < "$VECTORS/features.mem") != FRAMES * 512 )); then echo 'FEATURE_COUNT_FAIL';exit 1;fi
if (( $(wc -l < "$VECTORS/expected.txt") != FRAMES )); then echo 'SCORE_COUNT_FAIL';exit 1;fi
iverilog -g2012 -s tb_neuroedge_neural_monitor \
 -DNE_FRAMES="$FRAMES" -DNE_LOW="$LOW" -DNE_HIGH="$HIGH" -DNE_MAG="$MAG" \
 -o "$VECTORS/monitor.vvp" \
 rtl/control/neuroedge_budget_controller.sv \
 rtl/control/neuroedge_neural_activity_monitor.sv \
 verification/rtl/control/tb_neuroedge_neural_monitor.sv
vvp "$VECTORS/monitor.vvp" "+FEATURES=$VECTORS/features.mem" "+EXPECTED=$VECTORS/expected.txt"
