# Contributing

Keep contributions testable and tied to an explicit numerical, temporal or experimental contract. Describe what a change does, how it was tested, and which assumptions remain unverified.

Before submitting a pull request, run `make test`, `make verify-firmware`, `make website-check`, `make verify-rtl` and `python scripts/check_publication.py`. Icarus Verilog is required for RTL verification. Never include original neural data, derived input dumps, checkpoints, personal information, PDKs or confidential source. Treat a claimed accuracy improvement as a controlled experiment requiring held-out trials and provenance, not a single synthetic test.

A passing simulation is not a timing-closed ASIC, a verified analog acquisition system or a validated medical device. Do not make such claims in the website or documentation.
