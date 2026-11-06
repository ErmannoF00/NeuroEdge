# Research experiments that answer the open scientific questions

The primary application is **voluntary attempted-speech communication**. A future dataset may distinguish *rest, private imagined speech, deliberate intended communication*; the current attempted-speech labels do **not** support training an actual private-thought detector. NeuroEdge must not infer authorization merely because a neural pattern is linguistically decodable.

## Neural decoding benchmark

Use chronological train/development/test session separation; fit normalization, feature scales, model selection, and adaptation rules on train/development only. Report denominator and 95% uncertainty intervals alongside per-trial, per-session PER and WER. Baselines: the existing causal fixed-point CNN, stronger causal recurrent/TCN model, a comparable streaming transformer/state-space alternative, and an external language correction stage evaluated separately. Track trainable parameters, receptive field, memory footprints, quantization loss, and performance on the *same held-out data*.

## Feedback efficacy

Run **frozen / sham / delayed supervised / strictly inference-only unsupervised** policies on chronological trials. Annotate precisely when supervision becomes available. Predeclare channels and update rules on development trials, never tune them using held-out outcomes. Record proposals, denials, accepted writes, stale-version events, cooldowns, saturation and future-trial PER deltas. Include a negative control where targets are shuffled across trials. A policy that proposes zero writes is a useful safety result but no evidence of adaptation gains. To claim improvement, show prospective benefits on unseen trials with uncertainty intervals and without retrospective target leakage.

## Privacy and voluntary intent

A separate, consented research dataset would need explicit labels for rest, imagined private speech, attempted deliberate communication and independent release intention. Evaluate false release per hour during private/no-communicate conditions, omission of intended releases, selective-risk/coverage under uncertainty and out-of-distribution performance. The present external trial-arming gate is the correct safety default; it is **not** a neural intention decoder.

## Hardware/energy co-design

Preserve the exact INT8 reference. Build and check a full 512×64 projection using a real row scheduler/weight banking. Compare 1,2,4,8,16,32,64 banks with **measured RTL cycles**, actual memory transactions, saturation counts, and area/timing from explicitly named technology and memory models. For a CIM alternative, include bit precision, two's-complement/signed mapping, ADC/DAC, references, wire resistance, device variation, drift and peripheries. Track PER degradation on the same held-out neural set. A pure ideal dot-product equality is not evidence of analog-CIM feasibility.

## Acquisition and feedback beyond software

Build an electrical phantom and board-level interface before any biological connection. Specify electrode impedance envelope, input-referred noise, amplifier saturation and recovery, anti-alias band, sample rate, ENOB, clock drift, channel power, isolation, data integrity, missing packets and fault response. Define reproducible spike-band/threshold-crossing feature extraction. Keep chemistry-based sensors as a separate transduction modality. Add text-to-speech or other output only with an explicit consent/authorization mechanism and visible uncertainty.
