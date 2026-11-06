# Evidence ledger and engineering interpretation

**Status:** private research source. Evidence below comes from (a) executed local source tests and (b) explicit Ubuntu WSL terminal transcripts supplied by the project developer. It does not imply tool access to the developer's local datasets or hardware environment.

## Integrated digital simulation — developer's Ubuntu WSL transcript

| Check | Observed output | Interpretation |
|---|---|---|
| Public synthetic Python suite | `106 passed` before connected-stream add-on | Reorganized source numerical tests passed on WSL |
| Decoder | `FULL_QUANTIZED_DECODER_RTL_PASS frames=32 outputs=328 weights=44224` | Complete synchronous fixed-point model matches fixture |
| Integrated zero offset | `ZERO_OFFSET_328_LOGITS_PASS` | 328/328 golden comparisons with 512-feature correction bridge |
| Commit change | `COMMIT_CHANGES_LOGITS_PASS ... changed_logits=8` | Exact synthetic fixture expected changed logits, not improved PER |
| Abort/denial | `ABORT_DENY_PRESERVE_LOGITS_PASS`, `NO_MID_BIN_WRITE_PASS` | Unsafe memory updates rejected in tested circumstances |
| Integrated cycles | `LATENCY_ACCOUNTING_PASS worst_bin_cycles=90195 budget=500000` | Simulation cycle accounting, not physical clock measurement |
| Regression | `NEUROEDGE_PUBLIC_REGRESSION_PASS` | Reorganized RTL verified in WSL |
| Connected add-on rerun | `NEUROEDGE_EXISTING_RTL_REGRESSION_PASS` | Unchanged RTL gate rerun after host-source installation |
| Standalone near-memory dot tile | `NEUROEDGE_NEAR_MEMORY_DOT_RTL_PASS sum=-9676 inputs=8 writes=8` | Eight-tap near-memory digital MAC tested independently |

These outcomes do **not** establish actual processor timing, switched SRAM power, a CIM macro, a complete 64-bank projection, signal acquisition, speech synthesis, or a clinical BCI.

## Real-neural connected-stream replay — four offline validation trials

The developer executed the script with one local August 2023 validation HDF5, frozen checkpoint normalization and a frozen INT8 quantization profile. The script reported:

| Control arm | Trials | Bins | CTC timesteps | Scalar logits | Greedy PER | Authorized offsets |
|---|---:|---:|---:|---:|---:|---:|
| Frozen/off | 4 | 3,123 | 783 | 32,103 | 0.5948275862 | 0 |
| Sham | 4 | 3,123 | 783 | 32,103 | 0.6034482759 | 3 |
| Delayed supervised | 4 | 3,123 | 783 | 32,103 | 0.5948275862 | 0 |

**Readout:** supervised control did not authorize an update on any tested trial; PER therefore did not improve. The sham arm was slightly worse in this tiny sample. The data are not a prospective independent held-out test, and the labels used by the feedback search were read from recorded prompted trials after each trial. No claim of adaptation efficacy, clinical performance, private thought recognition or system-level speed should be made.

Use `python scripts/inspect_neural_replay.py build/local_neural_stream_report.json --out build/local_neural_summary.json` on the developer's machine to recheck denominators, arm equivalence and write accounting without disclosing identifiable contents. The source report must remain under the ignored `build/` directory.

## Synthetic feedback comparison

The generated 32-bin, eight-trial non-learned fixture produced fixed/sham/adaptive synthetic logit L1 values of 135/163/42 in the earlier deterministic replay, with four accepted supervised writes. This proves the **control logic can apply delayed updates in a constructed setting**. It does not prove the algorithm is useful on neural recordings. This result must always be labeled *synthetic*.

## Reproduction and evidence provenance

Commands for private acceptance are in [private acceptance](private_acceptance.md). The developer's independent private recorded-data replay requires licensed, locally held recordings and checkpoints. Public synthetic tests require none of these. The website evidence ledger is in `website/evidence.json` and intentionally contains only aggregate, non-sensitive values from supplied logs.

No raw HDF5, trained checkpoint, electrode waveforms, private neural `.mem` files, participant-specific transcripts, personal secrets or foundry assets are included.
