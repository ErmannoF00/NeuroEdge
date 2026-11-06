#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CC="${CC:-cc}"
mkdir -p build/firmware
"$CC" -std=c11 -Wall -Wextra -Werror -pedantic -I firmware/include \
  firmware/src/neuroedge_mmio.c \
  verification/c/test_neuroedge_mmio.c \
  -o build/firmware/test_mmio
./build/firmware/test_mmio
