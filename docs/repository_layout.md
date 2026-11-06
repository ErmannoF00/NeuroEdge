# Repository layout and ownership of the verified path

The top-level digital path starts at `rtl/soc/neuroedge_speech_pipeline.sv`; do not confuse it with other demonstration tops. The Python oracle is `neuroedge/speech/fixed_decoder.py`. New or refactored code should preserve the mathematical interface between them.

```text
neuroedge/
  speech/            dataset, CompactCTC, train/eval helpers, fixed integer decoder,
                     offset arithmetic and transactional software reference
  system/            C/Python packet framing, strict trial receiver, streaming host
  memory/            logical banked projection and idealized compute models
  control/           offset memory state model, activity budget logic,
                     strictly lagged external feedback authorization
  drift/             retrospective activity/risk studies and causal-audit tools
  datasets/          external dataset adapters
  vertical/          independent EEG-style 8-feature teaching example
  models.py          exploratory QNN/SNN/SSM models, not integrated speech RTL
rtl/
  memory/            independent digital near-memory dot tile (not full projection)
  speech/sync/       memory-based causal speech decoder and 1RW SRAM model
  soc/               current full decoder top + registered correction bridge;
                     separate older MMIO wrapper explicitly non-integrated
  control/           512x16 transactional offset store, neural monitor, budget
  speech/            older decoder and projection/affine blocks
  frontend/          separate event encoder experiment
  backend/           digital SRAM / NVM placeholder interfaces
  nmc/               arithmetic lanes and matrix tile experiment
  snn/, ssm/         independent spiking / state-space blocks
  vertical/          8-feature classifier demonstration
firmware/            C callback-driven register/protocol example (not live SoC)
verification/
  python/            signed arithmetic, data handling, reproducibility, policy tests
  rtl/speech/        synchronous SRAM and all-layer decoder equivalence
  rtl/soc/           byte-exact integration and transaction safety tests
  rtl/control/       offset transaction and neural-monitor unit tests
  rtl/vertical/      standalone classifier regression
  c/                 callback-driven firmware protocol tests
scripts/             fixture export, tests, dataset evaluation, research commands
research/neural/     experimental causal GRU alternative, not RTL mapped
analog/             SPICE electrode examples and isolated phantom
physical/           hypothetical memory banking, constraints and flow notes
memory/             external model interface documentation, no NVM hard macro
website/            responsive static research site; no patient application
external/           upstream research references and licensing notes
docs/               operative contracts and scientific limits
.github/workflows/  full synthetic RTL CI and static site deployment
```

The full exact file inventory and SHA256 manifest are bundled at the repository root. Generated data, private recordings, Python environments, build output and PDK-dependent results do not belong in this tree. The installed WSL repository with original baseline testing is intentionally separate.
