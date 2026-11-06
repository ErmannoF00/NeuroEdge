PYTHON ?= python3
.PHONY: help test vectors replay verify verify-rtl verify-firmware verify-vertical website-check verify-stream verify-memory private-acceptance clean
help:
	@printf 'make test             Python numerical and protocol regressions\n'
	@printf 'make replay           Deterministic synthetic supervised closed-loop replay\n'
	@printf 'make vectors          Deterministic public synthetic decoder vectors\n'
	@printf 'make verify-rtl       Complete synthetic 32-bin SystemVerilog integration\n'
	@printf 'make verify-vertical  Synthetic 8-feature classifier RTL regression\n'
	@printf 'make verify-stream    Frame transport, causal stream and control checks\n'
	@printf 'make verify-memory    Standalone near-memory dot tile and CIM numerical model\n'
	@printf 'make private-acceptance  One-stop complete private acceptance (RTL required)\n'
	@printf 'make verify-firmware  C register-protocol test\n'
	@printf 'make website-check    Static website/evidence content checks\n'

test:
	PYTHONPATH=. $(PYTHON) -m pytest -q verification/python

replay:
	PYTHONPATH=. $(PYTHON) scripts/run_closed_loop_replay.py --out build/closed_loop_replay.json

vectors:
	PYTHONPATH=. $(PYTHON) scripts/generate_public_vectors.py --out build/public_vectors

verify-rtl:
	NEUROEDGE_PYTHON=$(PYTHON) bash scripts/verify_public.sh

verify-vertical:
	PYTHONPATH=. $(PYTHON) scripts/run_vertical.py --dataset synthetic --out build/vertical
	bash scripts/rtl_equivalence.sh build/vertical/rtl_vectors

verify-firmware:
	bash scripts/verify_firmware.sh

website-check:
	PYTHONPATH=. $(PYTHON) scripts/check_website.py

verify-stream:
	NEUROEDGE_PYTHON=$(PYTHON) bash scripts/verify_connected_stream.sh

verify-memory:
	NEUROEDGE_PYTHON=$(PYTHON) bash scripts/verify_memory_compute.sh

private-acceptance:
	NEUROEDGE_PYTHON=$(PYTHON) bash scripts/verify_private_acceptance.sh

verify: test replay verify-firmware website-check verify-stream verify-memory verify-rtl

clean:
	rm -rf build/ dist/ .pytest_cache/
