# Reproducibility and verification

## What was independently observed

**Source of prior integration evidence:** terminal log supplied from Ubuntu WSL, running the unrenamed, integrated source with user-held real-neural-derived vectors. The transcript explicitly reported successful completion of all its integration gates. It reported:

| Test | Observed outcome |
|---|---|
| Python regressions | 97 passed |
| Legacy synchronous decoder | 32 bins, 328 exact logits, 44,224 weights, 15 output stalls |
| Zero-offset full integrated path | 32 bins, 16,384 feature transfers, 328 exact logits |
| Offset commit | 1 accepted update; 328 logits changed as predicted for an intentionally extreme correction |
| Abort and unsafe denial | Original logits preserved; no mid-bin write |
| Reset | Pending/valid/partial state cleared |
| Integrated latency | Worst 90,195 cycles; assumed 25 MHz; 500,000-cycle budget |

The committed-offset test proves an exact functional output change, **not** a decrease in phoneme errors. The extreme correction is deliberately unsuitable as an accuracy claim.

**Earlier private candidate:** locally 106 Python tests and a C firmware protocol regression pass. This is a copied, renamed private review candidate; **its SystemVerilog module renames subsequently passed in the developer's WSL terminal, but the consolidated source has not been rerun with Icarus in the packaging environment**. Therefore the new tree's RTL remains a private acceptance requirement.

## Public reproduction procedure

```bash
make test
make vectors
make verify-firmware
make website-check
make verify-rtl
```

`make vectors` builds sparse deterministic weights, biases, shifts, 32 bins of signed inputs and 328 expected outputs. The script includes SHA256 hashes and a `provenance = synthetic` marker. Nothing about this fixture measures clinical or BCI classification performance.

`make verify-rtl` invokes the existing unit tests, the synchronous decoder regression, the Python oracle and three integrated cases: zero offset, committed offset, abort/unsafe denial. It expects the simulator to compare every output against the reference. **Do not mark a passing CI badge until GitHub Actions really completes this job.**

## After installing in a new directory

Before publishing, rerun the local 32-bin recorded-vector gate with its private files using `bash scripts/verify_integrated.sh <path-to-private-rtl32-vectors>`. The scripts use the corrected, public naming, but no real neural input files are bundled. Keep output logs under ignored `build/` or `results/`; preserve important logs privately.

## No claims of

Foundry macro timing; physical clock closure; dynamic hardware power; measured SRAM energy or NVM endurance; clinical speech quality; autonomous closed-loop adaptation; patient safety; integrated analog sensing.


## Causal synthetic control replay (this candidate)

`make replay` runs `neuroedge/control/closed_loop_replay.py` on the same class of deterministic non-learned 512-feature decoder fixture as the public vector test. The trial-supervision interface supplies reference logits **only after the trial has completed**. Any update is authorized at a subsequent safe boundary by the real Python transactional `OffsetTable` and `LaggedFeedbackGate`.

The local run produced fixed-offset / sham / supervised-control integer-logit L1 totals of **135 / 163 / 42**, with **4** accepted supervised-control writes and **7** sham writes. This is a synthetic test of control sequencing and numerical correction, **not** human or real-neural closed-loop efficacy. Five new tests check lagged behavior, missing feedback, zero drift, delayed targets, and malformed trial targets.

The website replay contains an exact JSON copy of this fixture; the Python static website test compares the replay against its evidence metadata. No behavioral simulation of the reorganized RTL has been run **in the packaging environment**; the developer's independent WSL execution passed the named synthetic integration gates.

## Updated source and WSL evidence

The developer has subsequently completed the reorganized synthetic WSL RTL regression, including `NEUROEDGE_PUBLIC_REGRESSION_PASS`, and the connected-stream add-on re-ran `NEUROEDGE_EXISTING_RTL_REGRESSION_PASS`. See [current evidence](evidence_and_limitations.md). The new consolidated source still needs its own independent WSL acceptance using [one private gate](private_acceptance.md).
