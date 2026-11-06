# Emerging-memory backends

This directory separates three levels that are often incorrectly mixed together:

1. **Algorithm/device-aware model** — quantization, update nonlinearity, variability, drift.
2. **Circuit macro model** — SPICE/Verilog-A + word-line/bit-line/peripheral circuits.
3. **Physical hard macro** — GDS/LEF/Liberty/extracted netlist from a technology that actually supports the device.

The repository contains level (1) and interface templates for (2)/(3). It deliberately does not fabricate a fake FeFET/RRAM hard macro from standard CMOS cells.

## Backends

- `FeFET`: attractive for nonvolatile weights, high on/off ratio and device-aware personalization. Use measured conductance/update traces when available.
- `RRAM/memristor`: mature research literature for analog crossbars, but variation, write verify, ADC/DAC overhead and endurance must be modeled.
- `SRAM-NMC`: digital baseline. Larger/volatile, but far easier to synthesize, verify and tape out in a standard digital process.

The software parameters in `neuroedge.device` are placeholders for smoke tests only. Replace them with measured distributions before making scientific claims.
