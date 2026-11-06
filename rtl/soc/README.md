# Speech pipeline and control

**Current integrated top:** `neuroedge_speech_pipeline.sv`, combining `neuroedge_offset_stream_bridge.sv`, `rtl/control/neuroedge_offset_bank.sv` and `rtl/speech/sync/neuroedge_speech_decoder.sv`. Verification is under `verification/rtl/soc`.

`neuroedge_mmio_wrapper.sv` is retained as a separate, earlier host-register experiment and does not instantiate the integrated pipeline. A real host interface to the current top is outstanding.
