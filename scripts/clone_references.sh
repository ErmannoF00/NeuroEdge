#!/usr/bin/env bash
set -euo pipefail
mkdir -p external/checkouts
cd external/checkouts
clone(){ [ -d "$1" ] || git clone --depth 1 "$2" "$1"; }
clone SYNtzulA https://github.com/EOLAB-2025/SYNtzulA.git
clone x-heep https://github.com/x-heep/x-heep.git
clone yana https://github.com/fzi-forschungszentrum-informatik/yana.git
clone OpenCIMTC https://github.com/Tsinghua-LEMON-Lab/OpenCIMTC.git
clone ihp-ams-template https://github.com/iic-jku/ihp-sg13g2-ams-chip-template.git
cat <<'EOF'
Not cloned automatically:
- NeuroSim: research/non-commercial license; clone manually only for allowed use.
- JART: download from official FZ Juelich/RWTH page and verify redistribution terms.
- EEG-Ferro / memristor BCI code: inspect repository license before vendoring or modifying.
EOF
