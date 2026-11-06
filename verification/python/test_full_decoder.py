"""Independent functional checks for existing fixed-point temporal inference."""
import numpy as np
import torch
import pytest
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.fixed_decoder import (
    FixedParameters,FixedStreamingDecoder,quantize_input,quantize_model,
    _layer,_qweight,TOTAL_WEIGHTS,TOTAL_META,WEIGHT_OFFSETS,META_OFFSETS,
    WEIGHT_COUNTS,write_rtl_vectors,
)


def deterministic_model():
    torch.manual_seed(341)
    m=CompactCTC(hidden=64)
    m.eval()
    return m


def test_layout_and_shapes():
    p=quantize_model(deterministic_model())
    assert len(p.weights)==44224
    assert len(p.biases)==361
    assert len(p.shifts)==361
    assert all(int(p.shifts.max())<=31 for _ in [0])
    assert p.row_weights('project').shape==(64,512)
    assert p.row_weights('dw1').shape==(64,5)
    assert p.row_weights('pw1').shape==(64,64)
    assert p.row_weights('dw2').shape==(64,5)
    assert p.row_weights('pw2').shape==(64,64)
    assert p.row_weights('head').shape==(41,64)
    assert sum(WEIGHT_COUNTS.values())==TOTAL_WEIGHTS
    assert WEIGHT_OFFSETS['head']+WEIGHT_COUNTS['head']==TOTAL_WEIGHTS
    assert META_OFFSETS['head']+41==TOTAL_META


def test_numeric_contract_floor_relu_and_negative_logit():
    w=np.array([[-1,-1]],dtype=np.int8)
    x=np.array([1,0],dtype=np.int8)
    bias=np.array([0],dtype=np.int32)
    shift=np.array([1],dtype=np.uint8)
    assert _layer(x,w,bias,shift,head=True)[0]==-1  # -1 >>> 1 is -1
    assert _layer(x,w,bias,shift,head=False)[0]==0
    assert _layer(np.array([127,127],dtype=np.int8),
                  np.array([[127,127]],dtype=np.int8),bias,
                  np.array([0],dtype=np.uint8))[0]==127


def test_activation_quantization_is_bounded():
    x=np.full((2,512),100,dtype=np.float32)
    q=quantize_input(x)
    assert q.shape==(2,512) and q.dtype==np.int8
    assert np.all(q==127)
    x[:]=-100
    assert np.all(quantize_input(x)==-128)
    with pytest.raises(ValueError):quantize_input(np.full((2,512),np.nan))


def test_deterministic_across_chunk_boundaries():
    p=quantize_model(deterministic_model())
    rng=np.random.default_rng(77)
    x=rng.integers(-128,127,size=(13,512),dtype=np.int16).astype(np.int8)
    a=FixedStreamingDecoder(p).run(x)
    dec=FixedStreamingDecoder(p)
    b=np.concatenate([dec.run(x[:5]),dec.run(x[5:])],axis=0)
    assert a.shape==(4,41)
    np.testing.assert_array_equal(a,b)


def test_initial_zero_pad_and_causality():
    p=quantize_model(deterministic_model())
    rng=np.random.default_rng(1)
    x=rng.integers(-4,4,size=(8,512),dtype=np.int16).astype(np.int8)
    x2=x.copy()
    x2[4:]=127
    first=FixedStreamingDecoder(p).run(x)
    changed=FixedStreamingDecoder(p).run(x2)
    np.testing.assert_array_equal(first[0],changed[0])


def test_temporal_output_count_matches_pytorch():
    m=deterministic_model()
    p=quantize_model(m)
    for t in (1,2,3,4,5,8,9):
        x=np.zeros((t,512),dtype=np.float32)
        with torch.no_grad():floatout=m(torch.from_numpy(x)[None])
        y=FixedStreamingDecoder(p).run(quantize_input(x))
        assert y.shape==tuple(floatout.shape[1:])
        assert y.shape==((t+3)//4,41)


def test_bitwise_quantization_repeatability():
    p1=quantize_model(deterministic_model())
    p2=quantize_model(deterministic_model())
    np.testing.assert_array_equal(p1.weights,p2.weights)
    np.testing.assert_array_equal(p1.biases,p2.biases)
    np.testing.assert_array_equal(p1.shifts,p2.shifts)


def test_full_memory_image_and_vectors(tmp_path):
    p=quantize_model(deterministic_model())
    x=np.zeros((8,512),dtype=np.int8)
    y=FixedStreamingDecoder(p).run(x)
    write_rtl_vectors(tmp_path,p,x,y)
    assert len((tmp_path/'all_weights.mem').read_text().splitlines())==44224
    assert len((tmp_path/'all_biases.mem').read_text().splitlines())==361
    assert len((tmp_path/'all_shifts.mem').read_text().splitlines())==361
    assert len((tmp_path/'input_frames.mem').read_text().splitlines())==4096
    assert len((tmp_path/'expected_logits.mem').read_text().splitlines())==82
    with pytest.raises(ValueError):write_rtl_vectors(tmp_path,p,x,y[:1])
