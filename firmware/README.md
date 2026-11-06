# Host-side control experiments

The callback-driven C register protocol in `src/neuroedge_mmio.c` and `include/neuroedge_mmio.h` has a standalone C regression under `verification/c`. It exercises an earlier weight/configuration protocol, **not the new speech-pipeline top**. No CPU firmware has booted with the integrated neural decoder.

An actual SoC integration needs a defined bus (e.g. AXI-Lite/APB), address map, MMIO acknowledge semantics, startup model loading, stream ingress, FIFO output, DMA or interrupts, fault reporting, and reset behavior. Do not describe the experimental host stub as a deployable driver.
