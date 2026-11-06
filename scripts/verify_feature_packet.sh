#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${NEUROEDGE_PYTHON:-python}
mkdir -p build/connected_packet
PYTHONPATH=. "$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np
from neuroedge.system.packet_wire import WirePacket,encode
x=((np.arange(512,dtype=np.int32)*37+11)%256-128).astype(np.int8)
p=WirePacket(0,7,3,123456789,x)
Path('build/connected_packet/python_packet.bin').write_bytes(encode(p))
PY
cc -std=c11 -Wall -Wextra -Werror -pedantic -O2 \
  firmware/stream/neuroedge_feature_packet.c \
  verification/c/test_feature_packet.c \
  -o build/connected_packet/test_feature_packet
build/connected_packet/test_feature_packet build/connected_packet/python_packet.bin
PYTHONPATH=. "$PYTHON" -m pytest -q verification/python/test_feature_packet_wire.py
