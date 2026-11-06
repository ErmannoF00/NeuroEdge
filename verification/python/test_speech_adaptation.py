"""Tests for hardware-sized input-calibration contract, independent of patient data."""
import torch
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.adaptation import ChannelAffineDecoder


def test_initial_calibration_is_exact_identity():
    torch.manual_seed(1)
    model=CompactCTC(hidden=64)
    model.eval()
    wrapper=ChannelAffineDecoder(model)
    x=torch.randn(2,33,512)
    torch.testing.assert_close(wrapper(x),model(x))
    assert sum(p.numel() for p in wrapper.parameters() if p.requires_grad)==1024
    assert not any(p.requires_grad for p in wrapper.base.parameters())


def test_only_calibration_parameters_receive_gradient():
    torch.manual_seed(2)
    wrapper=ChannelAffineDecoder(CompactCTC(hidden=64))
    z=wrapper(torch.randn(2,27,512)).square().mean()
    z.backward()
    assert wrapper.gain.grad is not None
    assert wrapper.offset.grad is not None
    assert wrapper.gain.grad.isfinite().all()
    assert wrapper.offset.grad.isfinite().all()
    assert all(p.grad is None for p in wrapper.base.parameters())


def test_frozen_base_stays_in_eval_during_training():
    wrapper=ChannelAffineDecoder(CompactCTC(hidden=64))
    wrapper.train(True)
    assert wrapper.training
    assert not wrapper.base.training
    assert wrapper.calibration_state()['gain'].shape==(512,)
    assert wrapper.calibration_state()['offset'].shape==(512,)


def test_calibration_rejects_wrong_feature_count():
    wrapper=ChannelAffineDecoder(CompactCTC(hidden=64))
    try:
        wrapper(torch.zeros(1,4,64))
        raise AssertionError('Expected a dimension error')
    except ValueError:
        pass
