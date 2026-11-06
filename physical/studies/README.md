# Logical memory budgeting, not silicon characterization

Run `python3 physical/studies/memory_banking_sweep.py --out build/banking.json` for capacity and hypothetical macro geometry. This tool does not infer a manufacturable SRAM macro or measure power, area, throughput or frequency. The 25 MHz SDC is an assumed timing target. Foundry libraries, real SRAM collision semantics, synthesis, extraction, and timing/DRC/LVS results are still required before any silicon claim.
