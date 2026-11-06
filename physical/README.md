# ASIC implementation research

The present integrated digital top is `rtl/soc/neuroedge_speech_pipeline.sv`. No available physical flow has completed place-and-route or extracted timing for this top. Existing `physical/librelane/config.json` instead targets a **separate exploratory delta encoder** and should not be used to claim that the speech decoder was synthesized or routed.

The `studies/` subdirectory provides hypothetical SRAM banking and a 25 MHz *target* SDC. A full accelerator implementation requires characterized standard cells and SRAM macros, correct read/write collision semantics, memory mapping/DFT, top-level integration, timing closure and physical signoff. See [physical and analog limitations](../docs/physical_and_analog.md).
