# Diagnosing rejected neural calibration updates

The recorded four-trial replay reported frozen PER `0.5948275862`, sham PER `0.6034482759` (three writes), supervised PER `0.5948275862` (**zero writes**). This is a negative adaptation outcome: it neither proves beneficial updates nor that the classifier detected thoughts.

Use `scripts/diagnose_feedback_candidates.py` on local training/development data, with frozen checkpoint normalization and the same exact INT8 decoder as the original experiment. Each completed trial is decoded once with zero offsets, then again with bounded candidate single-channel ±steps. No candidate is committed. The script counts how many calibrated input values changed, how many output scalar logits changed, whether the CTC sequence changed and whether edit distance on the **same completed trial** decreased. Outputs include only numeric counts, not HDF5 arrays, paths, identities or phoneme strings.

Cause taxonomy:
- `no_input_effect_after_saturation`: the offset did not change any sampled INT8 input value (possibly Q4 saturation).
- `no_logit_effect`: changed features did not change decoder logits.
- `logits_changed_ctc_unchanged`: changes did not alter the CTC greedy sequence.
- `ctc_changed_no_edit_improvement`: emitted sequence changed without improving edit distance.
- `retrospective_edit_improvement_only`: retrospective candidate improved the same trial. **This is not evidence that a future trial improves.**

Default two channels (0,304), one step (4 Q4 integer units), maximum two trials keeps the experiment bounded. Use more channels or steps only with **predeclared development hypotheses**, never for post hoc test-set optimization. Report per-session PER with frozen, sham and prospective causal controls after policy selection. The script explicitly does not provide future-trial quality claims.

```bash
python scripts/diagnose_feedback_candidates.py --hdf5 "$HDF5" \
  --checkpoint "$CKPT" --quant-profile "$PROFILE" \
  --max-trials 2 --channels 0,304 --steps 4 \
  --out build/private_feedback_diagnostic.json
```
