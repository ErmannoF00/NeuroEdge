# System-level architecture decisions

## Recommendation: hybrid digital SRAM + CIM test backend

The complete decoder uses 44,224 INT8 parameters across six layers. In the present implementation, weight reads happen once per MAC and the registered SRAM read and multiplication are serialized. A new row-local SRAM + MAC tile supports exact outputs, clear scheduling and simple memory ownership, and is a conservative stepping stone to multiple banks.

A 64-lane per-row banked version of the 512×64 projection is a **future architectural design**, not present in the provided RTL. For full pipeline performance, model serial depthwise, pointwise and head layers and synchronization; the projection alone does not determine whole-model latency. A shared feature buffer feeding 64 banks needs multicast routing, valid alignment and backpressure. Weight preloads need an exclusive configuration mode and documented load latency. The exact output accumulator must be shifted and biased only after the entire row completes.

## What would make an SRAM digital-CIM implementation real?

A conventional SRAM plus adjacent multiplier is not SRAM-CIM. A true digital SRAM-CIM bitcell or compiler macro must support a compute operation through the array's read/bitline structure. Specify the compute truth table, bit packing, simultaneous activated rows, bitline sense thresholds, digital bit-serial reduction, precision extension, variation margin, access disturb, collision behavior and test strategy. Then develop technology-specific memory macros and compare end-to-end inference energy, including periphery and routing.

## What would make a memristor implementation real?

A resistive CIM array needs experimentally or process-grounded `G_min/G_max`, conductance states, reset/set distributions, cell endurance, read noise, temporal drift, array dimensions, wire resistance, DAC coding, ADC bit depth/rate, reference calibration and sign handling. Positive/negative conductances or a signed bit-plane scheme must reproduce signed INT8 weights. Represent 44,224 parameters with the required number of cells and spare/reference cells. Build a *crossbar plus peripherals* area and energy estimate. Map this exact model into the existing six-layer inference with precision perturbations **before** proposing superiority. Optimize Pareto (PER, p99 latency, J/inference, mm², programming lifetime), not isolated MAC/J.

## Closed-loop control is a separate problem

The synthetic supervised replay already enforces delayed updates, but no real-time target signal or hardware policy-to-MMIO communication is verified. Use immutable event records with `trial_id`, `bin_id`, timestamp, `offset_version`, feedback availability, quality gate and safety flags. Maintain staging/commit atomicity across bins, double-buffer calibration if needed. For clinical speech tasks, distinguish externally prompted attempted speech from arbitrary thoughts; accuracy and adaptation require held-out participant/session evidence, not just self-consistency between artificial teacher and decoder.

## Relevant published research

- Nature (2025), 'A mixed-precision memristor and SRAM compute-in-memory AI processor': https://www.nature.com/articles/s41586-025-08639-2
- Nature Materials (2026), 'Strategies of high-accuracy memristor-based analogue computing in memory for artificial intelligence': https://www.nature.com/articles/s41563-026-02600-y
- npj Unconventional Computing (2026), 'Achieving high precision in analog in-memory computing systems': https://www.nature.com/articles/s44335-025-00044-2

These references motivate alternative architectures; their reported performance must not be treated as NeuroEdge measurements.
