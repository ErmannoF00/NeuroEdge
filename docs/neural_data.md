# Recorded neural data and experimental validity

This repository contains source code and synthetically generated integration tests. It deliberately excludes human intracortical recordings, individual trial text, model checkpoints, derived neural features and full exported trained weight arrays.

The experimental pipeline uses 512 features per 20 ms bin derived from attempted-speech sessions with 256 electrode channels. **512 is the feature dimensionality, not a claim of 512 physical recording electrodes.** Neural preprocessing is outside the current digital accelerator. See `neuroedge/speech/dataset.py` and the training/evaluation scripts for the local data contract.

The historical trained INT8 model's reported August validation phoneme error rate was 58.732% across 35 trials. A December shift/development session gave 73.739% across 34 trials. Neither number implies intelligible synthesized speech. December data have informed engineering choices, so an untouched future test set is still required for an unbiased comparison of improved decoders/adaptation.

For reproducibility of quality metrics, record dataset DOI/version, individual sessions (where access conditions permit), trial inclusion rules, train/dev/test partition, normalization-fit partition, model SHA256, checkpoint provenance, decoding settings, errors and confidence intervals. Do not fit normalization on validation/test data, optimize on the nominal test set, or leak future-trial information into causal feedback experiments.

Before releasing participant-derived artifacts, verify the dataset license and consent-compatible usage; a public dataset link does not justify mirroring private local derivatives.
