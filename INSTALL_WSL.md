# Install NeuroEdge as one private working repository (Ubuntu WSL)

**Do not overwrite the existing verified `~/projects/neuroedge-bci-asic` directory.** This package is a consolidated, separate private development candidate; publication remains on hold. Do not apply prior overlay bundles on top of it.

Download `NeuroEdge_Unified_Private_Engineering.zip` into Windows Downloads, then:

```bash
mkdir -p ~/projects/neuroedge-private
unzip -q /mnt/c/Users/ermix/Downloads/NeuroEdge_Unified_Private_Engineering.zip \
  -d ~/projects/neuroedge-private
cd ~/projects/neuroedge-private/neuroedge

sudo apt update
sudo apt install -y python3-venv iverilog build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python scripts/check_manifest.py
make PYTHON=python private-acceptance
```

The final marker `NEUROEDGE_PRIVATE_ACCEPTANCE_PASS` is issued only after the script completes. **This exact consolidated ZIP still requires the WSL run.** The earlier unmodified RTL gates and standalone dot tile have already been reported passing on the developer's machine; previous evidence is not silently reassigned to new files.

To inspect the local website, run `python3 -m http.server 8080 --directory website` and open `http://localhost:8080`. The site is intentionally `noindex,nofollow` while private.

A private recorded-data replay can be launched with `scripts/run_connected_stream.py` using the locally held HDF5, trained checkpoint and quantization profile, as documented in [system integration](docs/system_integration.md). This generates an **ignored local JSON report** in `build/`. Audit it with:

```bash
python scripts/inspect_neural_replay.py build/local_neural_stream_report.json \
  --out build/local_neural_summary.json
```

No participant recordings, trained weights, local absolute file paths or neural text outputs are packaged. `make verify` runs source tests, but only `make private-acceptance` exercises the full consolidated set including the independent near-memory tile and packet exchange.
