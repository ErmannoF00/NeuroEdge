import importlib.util
from pathlib import Path
import pytest

def load(name):
    loc=Path(__file__).resolve().parents[2]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,loc)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_afe_budget_physically_positive():
    v=load('afe_budget').budget()
    assert v['combined_idealized_noise_uv_rms']>v['input_amp_white_noise_uv_rms']
    assert v['raw_adc_payload_bps']==61440000
    assert v['feature_stream_payload_bps']==204800
    assert v['nominal_raw_to_feature_bit_rate_ratio']==300
    assert v['analog_channel_power_mw']==2.56

def test_afe_reject_invalid_nyquist():
    with pytest.raises(ValueError):load('afe_budget').budget(analog_bw=10100)

def test_cycle_budget_known_totals():
    b=load('core_cycle_budget').estimate(4,25)
    assert b['total_ideal_cycles']==149417
    assert b['worst_case_cycles_per_bin']==44905
    assert b['peak_bin_processing_ms']<2
    assert b['timing_feasible_under_assumed_1cycle_mac']

def test_cycle_budget_reject_invalid():
    with pytest.raises(ValueError):load('core_cycle_budget').estimate(0,25)
