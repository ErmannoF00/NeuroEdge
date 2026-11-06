# Architecture and integration boundaries

## Implemented feature-to-logit path

```
 external feature producer [512 signed INT8 words per bin]
        |
        | ready/valid accepted samples
        v
  neuroedge_offset_stream_bridge
        |            ^
        |            | synchronous read: address[8:0], data[15:0]
        |     neuroedge_offset_bank
        |          512 x 16 signed coefficients + 512 valid flags
        |          stage / abort / authorization / counters
        |           ^
        |           | configuration and safe-boundary control
        v           |
  neuroedge_speech_pipeline (top-level ownership / bin tracking)
        |
        v
  neuroedge_speech_decoder
     weight SRAM model: 44,224 x 8; metadata SRAM: 361 x 37
     64 feature channels; causal temporal state
        |
        v
  41 signed INT16 CTC logits, ready/valid with output stalls
```

The diagram is digital only. It starts **after** offline feature extraction. The actual electrode front-end, analog filtering, sampling, event generation and digital feature computation are not inside this top-level RTL.

## Modules

- `rtl/soc/neuroedge_speech_pipeline.sv` owns configuration gating and authorization accounting, connects all three datapath blocks, and exposes per-bin completion and latency counters.
- `rtl/soc/neuroedge_offset_stream_bridge.sv` latches raw features on upstream handshakes, requests registered coefficient reads, saturates the sum, and sends exactly one calibrated value to the decoder. `bin_in_flight` stays asserted until decoder completion.
- `rtl/control/neuroedge_offset_bank.sv` contains one staged candidate and an active table. A commit request is accepted only at a safe boundary; no automatic retry is permitted. Physical offset contents are not automatically reset, but validity bits clear.
- `rtl/speech/sync/neuroedge_speech_decoder.sv` implements the ordered layer execution with synchronous RAM reads and causal history. It has its own input/weight configuration protocol.
- `rtl/speech/sync/neuroedge_sram_1rw.sv` is a behavioral synchronous 1RW memory model, **not a qualified foundry SRAM macro**.

The earlier `rtl/soc/neuroedge_mmio_wrapper.sv` does not instantiate this integrated top. It is an **independent, older control-register experiment** and must not be described as the production host interface. `rtl/top/neuroedge_top.sv` likewise demonstrates a separate digital front-end.

## Critical safety invariant

The decoder's `busy` may be low after receiving between 1 and 511 of the 512 required input words. Thus the whole-system safe boundary must additionally check upstream bin ownership, bridge emptiness, no outstanding frame completion, and competing write/feature traffic. A configuration or offset write must not be counted merely because a request pulse was issued.

## What is not integrated

No real AFE/ADC, digitized 256-electrode stream, CPU boot, driver, physical DMA, MAC near-memory hard macro, electrode-safe implant circuitry, or human-in-the-loop application forms part of the integrated top. Research examples in other directories are independent unless explicitly connected and verified.
