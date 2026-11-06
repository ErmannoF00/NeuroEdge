# Physical implementation and analog acquisition

The `physical/studies/memory_banking_sweep.py` tool calculates logical bit capacity and hypothetical SRAM bank geometries, **not post-layout area/energy or memory macro feasibility**. Stored payload: 44,224 × 8-bit weights, 361 × 37-bit metadata and 512 × 16-bit offsets, plus 512 validity bits. The total data payload excluding validity is 375,341 bits.

The SRAM behavioral model has registered access semantics, but its memory mapping, collision mode, macro compiler availability, resettable flags and repair/DFT method are not established. No RTL synthesis result here is evidence of fabrication readiness.

`analog/spice/` and `analog/bench/` contain illustrative electrical examples. They are not a fabricated or validated low-noise intracortical acquisition front-end, and the phantom source must not be connected to a human subject. A useful next technical deliverable is a dimensioned electrode + front-end + ADC noise budget, followed by transistor-level SPICE and Monte Carlo variation, then isolated bench measurements.

A credible compute-in-memory comparison must define the exact operator being moved into memory, realistic peripheral DAC/ADC quantization, array mapping/capacity, sneak paths/IR drop or device variation, sense amplifier and write endurance. It must compare accuracy, energy, latency and macro area against a standard SRAM-backed implementation under the same task and memory conditions. Storing coefficients in RAM near the decoder is not sufficient to classify the system as compute-in-memory.
