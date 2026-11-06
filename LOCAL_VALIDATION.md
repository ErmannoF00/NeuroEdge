# Local source validation — consolidated private candidate

This file records **only checks actually run against this source candidate in the packaging environment**. The packaging environment has Python and a C compiler, but no `iverilog` or `vvp`; therefore **no new RTL simulation PASS can be asserted here**.

- Python: **147 passed** across `verification/python` (includes connected streaming, C packet model, standalone near-memory arithmetic and four report-audit tests).
- Python/C 540-byte packet: `NEUROEDGE_PYTHON_C_FEATURE_PACKET_PASS bytes=540 features=512 crc=PASS`.
- C register protocol: `NEUROEDGE_HOST_FIRMWARE_TEST_PASS`.
- Static website: `STATIC_WEBSITE_EVIDENCE_CHECK_PASS`.
- Source/publication heuristic: `PUBLICATION_SOURCE_AUDIT_PASS` (rights and secrets still require manual review).
- Python connected-stream synthetic compare: 8 trials / 32 bins / 328 scalar logits. A controlled fixture differs from frozen/sham, as expected for synthetic teacher data. **No real-neural benefit inferred.**
- Private replay report auditor: `NEUROEDGE_LOCAL_REPLAY_AUDIT_PASS` with aligned arm denominators; tests reject missing arms, fake PER, mixed trial counts and unintended output release.
- Bash scripts: `bash -n` passed on consolidated acceptance scripts.
- Standalone near-memory module: the developer's *earlier WSL run* reported `NEUROEDGE_NEAR_MEMORY_DOT_RTL_PASS`, but this repackaged path has **not** been re-simulated in the packaging environment.
- Complete reorganized decoder: the developer's *earlier WSL run* reported `NEUROEDGE_PUBLIC_REGRESSION_PASS` and `NEUROEDGE_EXISTING_RTL_REGRESSION_PASS`; the consolidated checkout requires a new WSL regression.

The four-trial real-neural software replay is sourced **only** from a user-provided WSL terminal log: frozen PER 0.5948275862; sham 0.6034482759 (3 writes); supervised 0.5948275862 (0 writes). Neither neural HDF5 nor trained checkpoint is available in this source package. See [evidence ledger](docs/evidence_and_limitations.md).

No source was pushed to GitHub, no clinical/ASIC hardware test was run, and no website was deployed.

- Headless Chromium screenshot attempt in the packaging container did not complete. Browser visual QA is **not newly verified** for the consolidated candidate; check the local site manually at desktop and mobile widths before publication.
- Local Markdown links in the root README, `docs/` and `website/` README files were checked for missing relative targets; none were found.
