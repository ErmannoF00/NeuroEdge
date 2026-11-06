#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ngspice >/dev/null || { echo 'ERROR: ngspice not installed; no AFE simulation performed';exit 1; }
O="${1:-results/analog}"
mkdir -p "$O"
ngspice -b -o "$O/phantom.log" analog/bench/eeg_phantom_ac_transient.cir
if grep -iEq '(^|[^a-z])(error|fatal)([^a-z]|$)' "$O/phantom.log"; then
  tail -40 "$O/phantom.log";exit 1
fi
grep -iq 'peak_out' "$O/phantom.log"
grep -iq 'gain_abs_10hz' "$O/phantom.log"
echo 'EEG_PHANTOM_SPICE_ANALYSIS_GENERATED_NO_HARDWARE_VALIDATION'
