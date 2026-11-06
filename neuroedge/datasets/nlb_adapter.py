"""Neural Latents Benchmark integration notes.

NLB datasets are ideal for the intracortical branch. Use nlb_tools/DANDI to load
spike counts, then return float32 binned counts with shape [trial,time,neuron].
This file intentionally avoids hard-coding an obsolete download API.
"""
def require_nlb_tools():
    try:
        import nlb_tools  # noqa
    except ImportError as e:
        raise RuntimeError('Install nlb-tools and follow the current NLB dataset loader documentation') from e
