# Private engineering review checklist

This is a **private working tree**. No GitHub publication is authorized by completing individual items. No renaming of proven core interfaces is required in the original WSL baseline.

## Short, high-value steps

- [x] Keep one accurate source-of-truth architecture image (SVG and PNG); remove earlier labels implying unfinished integration.
- [x] Keep the README explicit about RTL, software replay, actual neural PER and unverified claims.
- [x] Exercise a bounded prior-trial supervised feedback loop on the real integer Python decoder without future targets; compare to fixed offsets and missing feedback.
- [x] Expose the *generated synthetic* replay trace interactively on the website, without pretending it represents a patient or trained neural accuracy.
- [x] Add synthetic generator/website checks to CI and publication-audit code.
- [x] Reorganized RTL passed the full synthetic regression in the developer's WSL terminal.
- [ ] Run the **consolidated** source through `make private-acceptance` in a fresh private WSL directory.
- [ ] Review the browser UI at desktop and mobile widths; check navigation, contrast, form/button and accessibility behavior.
- [ ] Run the complete read-only source, privacy, licensing and IP audit on the actual working directory. No public push until authorized.

## Next implementation candidates

- [ ] **Host-to-RTL transaction controller:** hook configuration/offset stage/commit, streaming and status registers to the *current* integrated speech top. Supply a mock host and backpressure/regression test. The older MMIO example is not that controller.
- [ ] **Recorded-data replay with causal feedback:** define independently obtainable supervision, enforce availability timestamps, maintain separate no-update/sham controls and report prospective PER, saturation and memory commits. The December shift-development session must not be recycled as a pristine holdout.
- [ ] **Memory-centric physical evidence:** quantify weight SRAM traffic and operations, establish an SRAM-based mapped area/timing baseline, and compare characterized CIM devices/macro nonidealities (ADC, programming, retention, endurance, drift) rather than extrapolating from abstract array sizes.
- [ ] **Analog acquisition validation:** validate electrode impedance/noise, input safety/isolation, protection, gain, bandwidth and ADC with a physical bench before describing any live acquisition path.

**Critical distinction:** transactional offset SRAM is genuine hardware memory but **not** in-memory compute. Human thought-to-speech restoration requires training, validated labels, complete sensing, communication UI and clinical evidence absent today.
