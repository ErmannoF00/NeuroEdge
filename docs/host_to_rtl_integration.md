# NeuroEdge host-to-RTL register and stream contract

**Scope:** a register-transaction adapter around the existing verified synchronous speech pipeline. The original decoder, bridge and 512-entry offset bank are not changed. This interface is a synchronous, single-owner **research ready/valid bus**, not an implemented APB/AHB/AXI device, USB transport, RISC-V peripheral mapping, DMA engine, or human neural interface.

## 1. Ownership, timing and data contracts

- `req_valid`, `req_write`, `req_addr[7:0]`, `req_wdata[31:0]` are inputs; `req_ready`, `req_rdata[31:0]`, `irq` are outputs.
- A transaction is accepted **only** on `posedge clk` with `req_valid && req_ready`. Keep request fields stable while stalled. **Reads sample data on the accepted edge**; no later response phase is provided.
- All registers are 32-bit and word-aligned; accesses to undefined or read-only addresses set a sticky fault. Illegal operations complete and fault, rather than waiting indefinitely to become legal.
- Configuration is sequential, with **44,224 signed INT8 weight bytes** followed by **361 signed INT32 biases and 361 unsigned 5-bit shifts**. The entire model must load before `SEAL=1` succeeds. No model update is allowed after sealing until a full hardware reset. `WEIGHT_PTR` and `META_PTR` are read-only progress indicators. Configured parameter memories **do not clear on reset**; the `configured` flag and the write pointers do.
- `INPUT` writes submit the low 8 bits as signed INT8; there is a four-entry input FIFO in the test configuration. The verified bridge accepts exactly 512 samples per logical bin. The host does **not** apply INT16 calibration itself: the existing bridge reads the offset bank and performs exact signed saturating addition.
- Decoder output is 41 signed INT16 logits per emitted CTC timestep, 1 output timestep per four input bins (on bins 0,4,8,...). The output FIFO carries `{frame[31:0], class[5:0], logit[15:0]}`. `RESULT_FRAME` reads the FIFO head frame **without popping**. Reading `RESULT` returns `{10'b0,class[5:0],logit[15:0]}` **and pops one entry**. The caller must ensure it is the only result consumer.
- Input and output queue pointers are independent, with no overwrites. When input FIFO is full, `INPUT` waits (`req_ready=0`). When result FIFO is empty, `RESULT` read waits. The decoder is stalled whenever output FIFO is full.
- A fault or unsafe calibration request is observable in read-only counters and sticky fault bits. Faults are W1C and setting dominates clearing at the same edge. `IRQ_STATUS[0]` is result level, `[1]` fault level and `[2]` completed-bin sticky. `IRQ_ACK[2]` clears only the completed-bin sticky flag. The output and fault interrupt causes clear when the underlying condition clears.
- Stage is a one-entry candidate in the existing offset bank, not an update queue. `OFFSET_CHANNEL` selects a channel. `OFFSET_VALUE` stages an absolute signed INT16 Q4 value (the 32-bit write word must be sign-extended). `OFFSET_CTRL=1` requests a **one-shot authorization**, `=2` aborts. An unsafe request is rejected and **never queued for later retry**. The existing core checks for a complete safe bin edge, empty bridge and idle decoder. Mid-bin staging is allowed; committing mid-bin is forbidden.
- Neither the presence of a neural feature packet nor receipt of a result authorizes speech release. The trusted host must independently enforce voluntary communication release; a neural-intent classifier has **not** been demonstrated.

## 2. Register map

| Address | Register | Mode | Semantics |
|---:|---|---|---|
| `0x00` | `ID` | R | `0x4e454443` (`NEDC`) |
| `0x04` | `STATUS` | R | bit0 configured, 1 decoder busy, 2 result available, 3 sticky fault, 4 staged offset, 5 partial bin, 6 input full, 7 IRQ |
| `0x08`, `0x0c` | `WEIGHT_PTR`, `WEIGHT_DATA` | R/W | sequential parameter loading, low eight write bits |
| `0x10`, `0x14`, `0x18` | `META_PTR`, `META_BIAS`, `META_SHIFT` | R/W/W | complete bias+shift pair increments metadata pointer |
| `0x1c` | `SEAL` | W | write 1 after exactly all parameters loaded |
| `0x20` | `INPUT` | W | one signed INT8 feature per accepted write |
| `0x24`, `0x28` | `RESULT`, `RESULT_FRAME` | R/R | destructive result dequeue, non-destructive frame peek |
| `0x2c` | `BINS_DONE` | R | complete 512-feature bins |
| `0x30`, `0x34`, `0x38` | `OFFSET_CHANNEL`, `OFFSET_VALUE`, `OFFSET_CTRL` | R/W, W, W | select, stage signed absolute Q4 offset, commit/abort |
| `0x3c` | `OFFSET_STATUS` | R | bit0 pending, 1 stage-ready, 2 write pulse, 3 accepted pulse, 4 rejected pulse, 5 bin partial, 6 core busy |
| `0x40`, `0x44`, `0x48` | `WRITES`, `REJECTS`, `REQUESTS` | R | accepted writes / rejected / accepted authorizations |
| `0x4c`, `0x50` | `INPUTS_ACCEPTED`, `RESULTS_POPPED` | R | host bus transfer counters |
| `0x54`, `0x58`, `0x5c` | `IRQ_ENABLE`, `IRQ_STATUS`, `IRQ_ACK` | R/W, R, W | bit0 output, bit1 fault, bit2 completed bin |
| `0x60` | `FAULT` | R/W1C | `1` bad register, `2` parameter error, `4` invalid sequence, `8` unsafe authorization |
| `0x64` | `FIFO_LEVEL` | R | `[7:0]` input occupancy, `[15:8]` output occupancy |
| `0x68` | `LAST_LATENCY` | R | existing integrated bin latency counter in cycles |
| `0x6c`, `0x70` | `STAGES`, `ABORTS` | R | existing bank counters |
| `0x74` | `VERSION` | R | `0x00010000`: interface ABI 1.0, not a repository milestone |

## 3. Bus-to-core integration tests

The self-checking testbench loads the complete model **exclusively through the bus**, feeds **32 complete bins (16,384 feature transfers)** through the input FIFO and checks **328 logits** against independently exported Python hex files. It checks the frame index, CTC class index, signed score, full FIFO backpressure, output IRQ, all three stream counters, the 25 MHz-assumed 500,000-cycle per-bin target, and sticky fault recovery.

Two cases run: **baseline** (one offset proposal staged during a partially received bin must be rejected, then aborted, no output changes), and **committed** (one offset committed before the first bin must match the independently calculated 328-logit result and change at least one logit). No hardware result is claimed until the new testbench executes on WSL and prints `NEUROEDGE_HOST_TO_RTL_REGRESSION_PASS`.

The full private gate remains unchanged; `scripts/verify_complete_system.sh` chains that gate, this new bus regression and the read-only synthetic feedback diagnostic. Old module source files and existing passing results are preserved.

## 4. Critical remaining system boundaries

1. **Trial reset vs persistent offset calibration:** the verified decoder's temporal history clears on `rst_n`, but the same reset also clears offset validity. This adapter currently supports a continuous streaming session. It does *not* promise equivalence to the Python evaluator's fresh decoder state at every separate trial while preserving learned offsets across trials. A future separately qualified decoder-state flush preserving calibration is needed and must be explicitly verified.
2. **Physical bus and host CPU:** no AXI/APB bridge, real MMIO device tree, CPU interrupt service, DMA, isolation boundary, or RTL co-simulation with a live host process exists yet. The C callback API makes these missing integrations explicit.
3. **Speech/intent:** logit output is internal to the trusted host; there is no proven private-thought classifier, no electrical acquisition hardware, no clinical speech release permission signal and no stimulation loop.
4. **Timing:** 25 MHz is an assumed clock for simulation-based cycle accounting. No mapped timing closure, power or tapeout data is claimed.
5. **Memory-centric acceleration:** this adapter uses the unchanged serial reference inference core. Digital banked compute and analog RRAM/FeFET CIM are separate experiments, not part of this bus path.
6. **Coherent real-neural multi-trial experiment:** the 32-bin private fixture establishes fixed-point hardware equivalence for an excerpt, not a statistically meaningful future-trial adaptation benefit. Full clinical-style validation requires prospective allowed feedback and held-out sessions.
