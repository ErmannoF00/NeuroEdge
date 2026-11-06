# NeuroEdge — memory-computing architecture experiment

## Status and scope

This additive source implements a **synthesizable sequential, near-memory INT8 dot-product** using the repository's existing synchronous 1RW SRAM model. Its numerical contract is compatible with an individual affine row of the speech decoder. It does **not** replace the 44,224-weight multi-layer decoder; a future scheduler must dispatch each row and preserve causal history, per-layer integer scaling and the CTC output cadence. The currently verified full decoder remains the system reference.

## Exact arithmetic

For a row with 1–512 operands, `acc = sum(int8(x_i)*int8(w_i))`, with signed 32-bit accumulation. Maximum absolute 512-term magnitude is 8,388,608, so no accumulator overflow. A layer wrapper must then implement `sat_relu8((acc >>> shift)+bias)` (or signed saturation for depthwise and INT16 saturation for the head). Shifts are arithmetic. Do not insert activation saturation between chunks of one row.

Weights are written only at a fully idle row boundary. The input comprises LENGTH accepted ready/valid beats, not elapsed cycles. One read launches on each accepted beat; the MAC consumes the registered SRAM output on the following cycle. After LENGTH accepted beats, the tile holds the signed INT32 output under `out_valid && !out_ready`. Results and running row state survive input stalls; reset aborts incomplete work. There is no runtime weight update during inference.

## Exact versus analog backend comparison

- **Current decoder**: one physical, modelled synchronous 1RW weight store; exact serial MAC and full model semantics.
- **New near-memory tile**: an independent SRAM per row/bank and local serial MAC, with exact integer semantics. Parallel banks can process different outputs concurrently, but the supplied RTL instantiates ONE tile only.
- **Digital SRAM-CIM**: future bitline/wordline computation using specially designed 8T/10T cells, XNOR/AND/charge readout, or digital bit-serial operations; cannot be inferred from ordinary SRAM inference RTL.
- **RRAM/FeFET CIM**: future crossbars, device cells, programming/read verify, signed differential encoding, voltage/current drivers, ADC and quantization, mapping and drift/error calibration. The included bitplane emulator is idealized and models integer quantization/clipping, not Ohmic current, line resistance, device conductance variation or measured energy.

## Concrete memory hierarchy proposal

Feature buffer: 512×8 bits (4096 bits) per bin, double buffer if overlapping acquisition and compute. Offset bank: 512×16 bits plus validity/control, 8192 bits. Model weights: 44,224×8 bits = 353,792 bits. Bias/shift metadata: 361×37 bits = 13,357 bits. The projection has 64 independent output rows of length 512 (32,768 MACs). A 64-bank, single-MAC-per-row projection is theoretically bounded by 1,024 compute cycles plus configuration, buffer, bus and arbitration latency; this is NOT a complete RTL implementation or a validated 25 MHz clock. The single-bank lower bound is 65,536 compute cycles.

Each SRAM macro mapping needs explicit memory aspect ratios, banking, retention/power modes, collision behavior, DFT/MBIST and cell libraries. If a 512×8 hard macro is unavailable, model the compiler's actual macro dimensions and unused rows/columns; never infer macro area from an array of flip-flops.

## Executable validation

```bash
make verify-memory
python3 scripts/analyze_memory_banking.py
```

Run with the main repository's virtualenv active (numpy/pytest). The verification compiles the **actual** existing `neuroedge_sram_1rw` alongside this tile and exercises unsafe weight update denial, signed input arithmetic, result backpressure and exact control counters. iverilog/vvp is required. The independently distributed tile passed a developer-provided WSL simulation, but this consolidated copy requires its own new WSL acceptance run.

## Completion gates for an actual closed-loop / CIM system

1. Integrate a row dispatch interface and two-way backend selection into `neuroedge_speech_pipeline` without changing input/output semantics; compare every one of 328 synthetic logits and, locally, the 328 neural-derived logits.
2. Integrate the lagged feedback controller with a versioned MMIO protocol. Commit offsets only at full-bin boundaries, and account for rejected attempts, watchdogs and state snapshots.
3. Replay strictly held-out real neural sessions, freezing thresholds and update policy on train/development splits. Evaluate PER, blank rate, errors, update frequency, latency and failure cases against no-update and sham controls. Human feedback requires a justified supervision source.
4. For RRAM, calibrate a crossbar noise/ADC/IR-drop model against device/circuit evidence. Map the same 44,224 weights into bit-sliced arrays, model reference cells/peripheral overhead, evaluate model-level PER and compare energy/area against synthesized SRAM/NMC baselines.
5. Only then evaluate analog AFE + ADC + DMA/firmware with a measured phantom signal. Implantable/device safety, clinical outcomes and physical silicon are entirely outside the present verification.

## Caveats for publication

Do not claim an operating implant, clinical benefit, guaranteed 25 MHz, compute-in-memory silicon, or measured power. `analyze_banking.py` gives analytical ideal limits only. Maintain invention/IP review and keep neural participant recordings, PDK libraries, checkpoints and employer-controlled files private.
