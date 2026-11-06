# System boundaries and unimplemented subsystems

| Subsystem | Present capability | Not demonstrated |
|---|---|---|
| Intracortical feature preparation | Dataset adapter; offline features | Direct ADC/implant stream capture |
| Neural inference | Causal fixed-point CTC decoder | Useful real-world speech restoration |
| Calibration | 512 signed INT16 offset table, stage/abort/commit, bin-safe update | Accurate autonomous adaptation |
| Feedback | Lagged, bounded software proposal authorization | Human feedback pipeline and clinical benefit |
| Hardware compute | Integer MAC pipeline and SRAM-style parameter reads | True compute-in-memory multiplier/accumulator |
| Memory devices | SPICE/device research and analytical banking tools | RRAM/FeFET hard macro with measured variation and endurance |
| Analog | Idealized phantom and electrode model | Designed, taped-out, measured low-noise neural AFE/ADC |
| Host control | Earlier C register interface example | Complete current-top MMIO map, DMA and live firmware |
| ASIC physical | Constraints and logical storage budget | Foundry SRAM replacement, placement/routing, extracted STA, DRC/LVS |
| UI | Static project website and simulated arithmetic widget | Patient control application |

## Engineering-critical work

1. **A true integrated host interface:** connect parameter load, offset stage/commit, feature ingress, logit dequeue, reset and error counters to a concrete bus protocol. Verify transaction ordering, invalid writes, reset during transfer and backpressure. The older MMIO wrapper targets an older core and cannot be claimed as the current integrated interface.
2. **Real memory mapping:** replace generic array models with characterized macros; prove read/write collision semantics, reset strategy, banking, testability, timing, and realistic capacity/power.
3. **Closed-loop evaluation:** only delayed, causally available feedback may issue bounded calibration proposals. Compare against frozen decoder, sham updates and supervised oracle controls on session-disjoint data. Report error reduction and degradation with confidence intervals.
4. **Physical design:** elaborate and synthesize *the actual integrated top*, inspect inferred memories, perform timing-driven mapping, placement, CTS, routing and extracted STA. Do not extrapolate the 25 MHz assumption into a tapeout claim.
5. **Analog acquisition:** specify signal amplitude, bandwidth, electrode impedance/offset, input-referred noise, common-mode interference, saturation recovery, ADC ENOB and safety isolation. Test with safe benchtop phantoms before any human work.

These are explicit gaps, not completed functionality.
