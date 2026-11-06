# Closed-loop research contract

The current source implements **a transactional memory update mechanism** and an offline proposal gate (`neuroedge/control/feedback_policy.py`). It does not infer desired calibration corrections directly from raw thoughts or achieve speech restoration.

A real feedback experiment needs a causal trace:

```
completed neural trial → frozen decoder output → external feedback becomes available
                                ↓
                 source_trial, availability timestamp, provenance
                                ↓
                 bounded update proposal (external estimator)
                                ↓
             authorization at a complete bin boundary only
                                ↓
                      next trial uses committed offsets
                                ↓
                    held-out evaluation of subsequent output
```

No future labels may be used to propose an update for an earlier trial. `LaggedFeedbackGate` rejects unavailable feedback, duplicate evidence, excessive per-update delta, offset overflow, channel cooldown violations and unsafe bin boundaries. It does not decide which offset would improve accuracy; that needs a separately trained, evaluated estimator with appropriate controls. Hardware logic has a single pending memory entry and no automatic retry.

The strongest minimum experiment compares: fixed network with zero corrections; fixed network with randomized or sham corrections; externally supervised proposals based only on previously available feedback; and an oracle upper bound clearly identified as noncausal. Report trial-level paired changes in PER, saturation fraction, blank rate, request frequency and energy *proxies* separately. Never claim endurance, energy, or safety from counting update requests instead of actual accepted memory writes.

## Implemented synthetic replay of the supervised loop

Run `make replay` (or `PYTHONPATH=. python3 scripts/run_closed_loop_replay.py`).

`neuroedge/control/closed_loop_replay.py` executes **the actual fixed-point Python decoder**, using a deterministic, *non-learned* 44,224-weight fixture and 512-feature input bins. It does not use private neural data. Each trial is four bins and yields one 41-logit timestep. An injected constant +9-Q4 drift on synthetic feature 0 causes a reproducible discrepancy from a drift-free synthetic teacher.

The replay is causal by construction:

1. Before trial `t`, snapshot the current decoder state, and apply only memory commits previously authorized at safe boundaries.
2. Run all four bins of `t` through the actual `OffsetStreamingDecoder`. The memory is marked `bin_in_flight` during each bin.
3. **Only after trial `t` completes**, release its externally supplied synthetic clean-target logits into a timestamped feedback queue.
4. At a later trial boundary, select eligible prior-trial evidence; evaluate bounded positive and negative offset steps by *replaying the completed prior trial from its captured pre-trial decoder state*. Never read a future trial's features or target.
5. Reject stale evidence if the active offset has changed since the candidate's source trial. Otherwise, pass the proposal through `LaggedFeedbackGate` and the existing transactional `OffsetTable`. The committed offset affects only a later trial.

The optional `--feedback-delay-trials N` argument enforces an additional N-trial feedback delay. `N=100` produces no commits in an eight-trial fixture. The unit tests check the delay, zero-drift no-op, source-trial causality, bounds and shape validation.

The generated `build/closed_loop_replay.json` contains every trial's fixed/sham/control logit L1 score, active offset, bounded proposal, authorization decision and accepted-write count. The website's bundled trace is regenerated from the same code. Any reported reduction is an **offline synthetic supervised-control metric**; the teacher logits are privileged simulation data, not thoughts that can be observed in vivo.

**Missing for genuine neural closed-loop evidence:** a separately justified source of post-trial supervision, live or replayed session-consistent feature acquisition, error estimator with held-out-session validation, sham/random/no-update controls, and measurement of patient-relevant outcomes. None of these can be inferred from this synthetic numerical experiment alone.

A deterministic **sham controller** also commits bounded offsets at the same trial boundaries, but its proposal signs are predeclared and never depend on the teacher. In the default fixture the accumulated integer-logit L1 values are fixed=135, sham=163, supervised=42. This is a negative control for the software test, not evidence of neural decoding improvement.
