# SG13G2 real full-chip methodology handoff (pending execution)

Use the maintained [IHP SG13 LibreLane full-chip template](https://github.com/IHP-GmbH/ihp-sg13-librelane-template). Do not copy random PDK files into Git and do not assume JART Verilog-A means the same process includes physical RRAM cells.

1. Make vertical slice score equivalence PASS under Icarus.
2. Add an explicit weight-loading memory interface (not a `$readmemh`-only mock ROM) and a proper top-level bus/stream wrapper.
3. Choose SRAM macro type, size, port and timing with the chosen PDK. Create accurate Verilog blackbox, Liberty, LEF and GDS views.
4. Run standalone synthesis, inspect mapped cell counts and clock definitions. Establish target Fmax with a real SDC.
5. Clone and pin IHP official template revision; replace the reference `chip_core` with the verified block and align pad count, clock/reset, pin placements and PDN.
6. Run physical implementation at specified PVT. Sign off setup/hold, CTS, routing, antennas, DRC/LVS, extraction and pad/ESD constraints; archive reports and hashes.
7. Run gate-level regression and verify functionality using the same weight/test-vector ABI (memory init or firmware programming now real).
8. Document die size, actual implemented macro area, clock, activity assumptions and estimated power by component; explicitly separate simulation and post-layout estimates from silicon measurements.

Never fabricate an NVM macro from a behavioral RTL wrapper. In the absence of compatible NVM process integration, the tapeout baseline is digital SRAM/NMC only.
