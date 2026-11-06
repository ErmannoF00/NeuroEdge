# SPICE integration contract

A real emerging-memory macro should provide, at minimum:

- bit-cell / device compact model (SPICE or Verilog-A)
- program/read pulse conditions
- conductance-state distributions
- read noise and retention/drift model
- endurance assumptions
- word-line / bit-line parasitics
- sense amplifier / ADC or charge-domain readout
- write driver / verify loop
- PVT + Monte-Carlo characterization
- extracted post-layout netlist

For the digital SoC, harden the array and expose abstract views:

```text
NVM macro
├── .gds          physical geometry
├── .lef          place-and-route abstract
├── .lib          timing/power
├── .spice/.cdl   LVS/circuit
└── behavioral SV functional model
```

Do not infer those views from the illustrative Python model.
