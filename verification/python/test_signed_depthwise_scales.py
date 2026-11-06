"""Directed architectural regression that existing missed.
Depthwise convolution is SIGNED linear and is followed by pointwise + ReLU.
"""
import numpy as np
import torch
import pytest
from neuroedge.speech.fixed_decoder import (
    FixedParameters, FixedStreamingDecoder, quantize_model,
    quantize_input, _layer, TOTAL_WEIGHTS, TOTAL_META, WEIGHT_OFFSETS,
    META_OFFSETS, LAYER_ORDER,
)
from neuroedge.speech.model import CompactCTC
from scripts.calibrate_int8_scales import select_bits


def test_signed_depthwise_relu_location():
    w=np.array([[1]], dtype=np.int8)
    x=np.array([-24], dtype=np.int8)
    bias=np.array([0],dtype=np.int32)
    shift=np.array([0],dtype=np.uint8)
    assert _layer(x,w,bias,shift,signed_hidden=True)[0]==-24
    assert _layer(x,w,bias,shift)[0]==0
    assert _layer(np.array([-128],dtype=np.int8),
                  np.array([[2]],dtype=np.int8),bias,shift,signed_hidden=True)[0]==-128


def test_negative_depthwise_must_pass_through_negative_pointwise_weight():
    # A carefully constructed all-six-layer inference with negative DW outputs.
    # With the old, incorrect DW-ReLU, the result would have been zero.
    weights=np.zeros(TOTAL_WEIGHTS,dtype=np.int8)
    biases=np.zeros(TOTAL_META,dtype=np.int32)
    shifts=np.zeros(TOTAL_META,dtype=np.uint8)
    weights[WEIGHT_OFFSETS['project']]=1  # projection channel 0 takes input 0
    weights[WEIGHT_OFFSETS['dw1']+4]=-1   # newest tap, channel 0
    weights[WEIGHT_OFFSETS['pw1']]=-1     # negative DW times negative PW
    weights[WEIGHT_OFFSETS['dw2']+4]=-1
    weights[WEIGHT_OFFSETS['pw2']]=-1
    weights[WEIGHT_OFFSETS['head']]=1
    dec=FixedStreamingDecoder(FixedParameters(weights,biases,shifts))
    x=np.zeros((8,512),dtype=np.int8)
    x[:,0]=32
    out=dec.run(x)
    assert out.shape==(2,41)
    assert out[0,0]==32 and out[1,0]==32
    assert (out[:,1:]==0).all()
    assert dec.saturation_hits['dw1']==0


def test_custom_per_layer_power_of_two_scales():
    torch.manual_seed(17)
    m=CompactCTC(hidden=64)
    default=quantize_model(m)
    profile={
        'input_frac_bits':4,
        'output_frac_bits':{'project':4,'dw1':5,'pw1':4,'dw2':5,'pw2':4,'head':8}
    }
    p=quantize_model(m,profile)
    np.testing.assert_array_equal(default.weights,p.weights)
    np.testing.assert_array_equal(default.shifts[META_OFFSETS['project']:META_OFFSETS['dw1']],
                                  p.shifts[META_OFFSETS['project']:META_OFFSETS['dw1']])
    # The depthwise output has double the integer precision -> shift reduces 1.
    np.testing.assert_array_equal(p.shifts[64:128].astype(np.int32),
                                  default.shifts[64:128].astype(np.int32)-1)
    # Its subsequent pointwise layer needs one extra shift to return to 1/16.
    np.testing.assert_array_equal(p.shifts[128:192].astype(np.int32),
                                  default.shifts[128:192].astype(np.int32)+1)
    assert p.biases.shape==(TOTAL_META,)


def test_train_only_quantile_fraction_bits():
    assert select_bits(1.0)==6
    assert select_bits(2.0)==5
    assert select_bits(8.0)==3
    assert select_bits(0.0)==9
    with pytest.raises(ValueError):select_bits(float('nan'))
