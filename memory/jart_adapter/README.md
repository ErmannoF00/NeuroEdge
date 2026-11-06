# JART physical-device research workflow — external dependency

JART (Jülich Aachen Resistive Switching Tools), from the FZ Jülich/RWTH research ecosystem, includes physically motivated VCM/OxRAM models. **No Verilog-A JART source is vendored here**: it is not automatically open-redistributable merely because it is available to download. Refer to official https://www.fz-juelich.de/en/pgi/pgi-7/research-groups-1/ag-menzel/jart-model, identify the exact release/associated publication and request commercial terms separately if needed.

## What a scientifically valid integration actually involves

1. Select the exact VCM model/version, verify its device physics/ranges and simulator compatibility. Record author/model revision/license/DOI and SHA256.
2. Compile/load Verilog-A using a simulator that supports the model's constructs. A plain `.include model.va` in ngspice is **not** sufficient unless the model was compiled/transformed appropriately. OpenVAF/OSDI, Xyce/ADMS or compatible proprietary engines are possible depending on syntax/version; validate rather than assume.
3. Make a 1T1R test cell with MOS access device, programming drivers and sensible compliance/current limits. Characterize SET, RESET and read at several temperatures and voltage corners; check current limits and programming windows against source authors' characterization. Model variability only if supported by the particular model and with documented seeds/distributions.
4. For 2/4/8 conductance states, sweep program-and-verify, retention, D2D/C2C (if supported), read disturb. **Simulate the peripheral circuits** (WL, DAC/driver, sense/ADC, reset/program pumps) if claiming macro energy.
5. Export a raw CSV with one row per measurement (not hand-picked summary):

```
state,conductance_s,read_energy_j,program_energy_j,read_latency_s,program_latency_s
0, ... (actual SPICE results only)
...
```

6. Summarize with the converter included here:

```bash
python scripts/characterize_nvm_csv.py \
  --csv memory/generated/jart_measured_runs.csv \
  --source-model 'JART VCM <exact variant+version>' \
  --model-sha256 '<64-hex actual downloaded model digest>' \
  --out memory/generated/jart_characterized.json
```

7. Propagate conductance and energy distributions into a **model-to-CIM mapper** and compare baseline classification with device-aware classification using identical held-out subjects and weight tensors.

The converter validates units/positivity, reports state statistics and records source hashes. It **does not magically produce JART results**. The unit tests use fake *test fixtures* only. Do not cite the fixture as measured RRAM.

## Differential coding and physical accounting

For signed weights: \(w_{ij}=\alpha(G_{ij}^{+}-G_{ij}^{-})\). Then the column computes \(I_j=\sum_i V_i(G_{ij}^{+}-G_{ij}^{-})\), subject to line resistance and device nonlinearity. Include both arrays, driver/DAC energy, sense/ADC, write verification, refresh/calibration, spare cells and wire overhead when making fair area/power comparisons. Analog CIM is not equivalent to the digital INT8 v1 RTL classifier without a calibrated quantization map.
