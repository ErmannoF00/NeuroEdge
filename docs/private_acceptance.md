# Private acceptance and publication hold

## Single working repository, single test entry point

All modules are in the same NeuroEdge source tree. Do not copy successive overlay ZIPs on top of the verified WSL directory once this consolidated candidate is used. Work inside a separate private development checkout. The original directory `~/projects/neuroedge-bci-asic` is the known-good evidence anchor and must stay unchanged.

From a clean WSL copy of this candidate:

```bash
cd ~/projects/neuroedge-private/neuroedge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
sudo apt-get install -y iverilog build-essential
python scripts/check_manifest.py
make PYTHON=python private-acceptance
```

This runs source audit, Python tests, C tests, website integrity, C↔Python framing, synthetic connected streaming, synthetic delayed feedback, standalone SRAM-near-memory dot RTL, and the complete integrated SystemVerilog synthetic regression. The final marker `NEUROEDGE_PRIVATE_ACCEPTANCE_PASS` must **actually appear** in WSL; it has not been claimed for this new consolidated candidate.

To validate a locally generated, ignored, private recorded-neural report:

```bash
python scripts/inspect_neural_replay.py build/local_neural_stream_report.json \
  --out build/local_neural_summary.json
```

Do not copy local participant data or logs into this package.

## Required engineering signoffs before a public link

| Gate | Current meaning | Public acceptance criterion |
|---|---|---|
| Functional decoder and memory | WSL simulator gates already reported PASS | Re-run consolidated source in fresh WSL; preserve exact log/commit hash |
| Host software | New stream and C packet layer tested in isolation | End-to-end deterministic stream gate and error injection on clean clone |
| Real neural task | 4-trial PER ~59.48%, no supervised writes | Chronological multi-session evaluation, frozen hyperparameters, per-trial and aggregate statistics |
| Output permission | Independent external arming | Document privacy threat model; test fail-closed faults and operator confirmation |
| Memory compute | One standalone digital tile; Python bank model | Integrate full projection RTL, compare exact activations and complete logits |
| CIM | Idealized bit-plane software emulator | Device-/peripheral-calibrated nonideality model and measured area/energy methodology |
| Analog/biointerface | Exploratory phantom only | Bench-validated electrical specifications, safety review, no human deployment claims |
| Physical ASIC | Constraints and analytical estimates | Technology-mapped netlist with library provenance, real macro model and STA reports |
| Web/README | Technical claims tied to evidence ledger | Verify mobile browser behavior, links, accessibility, ownership, and licenses |
| Git and legal | Current private source may have no Git history | Inspect all staged files/secrets/third-party licenses/IP before remote creation |

**Do not put publication on a deadline.** It is appropriate to keep it private until a unified clean-clone demonstration, rights review and source-level verification are completed. This is a functioning research platform; it is not a validated medical interface.
