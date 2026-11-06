import numpy as np
import pytest
from scripts.audit_deployment import aggregate,bootstrap_delta_ci
from pathlib import Path


def test_micro_per_weights_phoneme_counts_not_trial_averages():
    rows=[{'target_tokens':100,'float_edits':20,'int8_edits':10},
          {'target_tokens':2,'float_edits':1,'int8_edits':2}]
    a=aggregate(rows)
    assert a['target_phonemes']==102
    assert a['float_PER']==21/102
    assert a['int8_PER']==12/102
    assert a['int_minus_float_PER']==-9/102


def test_bootstrap_reproducible_and_bounded():
    rows=[{'target_tokens':10,'float_edits':6,'int8_edits':4},
          {'target_tokens':15,'float_edits':7,'int8_edits':6},
          {'target_tokens':20,'float_edits':8,'int8_edits':8}]
    a=bootstrap_delta_ci(rows,seed=9,repetitions=1000)
    b=bootstrap_delta_ci(rows,seed=9,repetitions=1000)
    assert a==b and a[0]<=a[1]<=0
    with pytest.raises(ValueError):bootstrap_delta_ci(rows,repetitions=1)
    with pytest.raises(ValueError):aggregate([])


def test_extended_rtl_scripts_and_mmio_register_map_exist():
    root=Path(__file__).resolve().parents[2]
    assert (root/'rtl/soc/neuroedge_mmio_wrapper.sv').is_file()
    assert 'parameter integer NF=8' in (root/'verification/rtl/speech/tb_full_decoder.sv').read_text()
    assert 'neuroedge_mmio.h' in str(root/'firmware/include/neuroedge_mmio.h')
