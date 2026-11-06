"""Independent existing arithmetic/transaction tests; no training data needed."""
import numpy as np
import pytest
from neuroedge.speech.offset_reference import (
    saturating_offset_add, apply_offsets, OffsetTable, OffsetStreamingDecoder)
from neuroedge.speech.fixed_decoder import FixedParameters, TOTAL_WEIGHTS, TOTAL_META


@pytest.mark.parametrize('offset',[-32768,-129,-128,-127,-1,0,1,127,128,32767])
def test_exhaustive_input_domain(offset):
    for x in range(-128,128):
        expect=min(127,max(-128,x+offset))
        assert saturating_offset_add(x,offset)==expect


def test_fuzz_vectorized_crosscheck():
    gen=np.random.default_rng(25052026)
    for _ in range(256):
        x=gen.integers(-128,128,size=512,dtype=np.int16).astype(np.int8)
        o=gen.integers(-32768,32768,size=512,dtype=np.int32).astype(np.int16)
        result=apply_offsets(x,o)
        assert result.dtype==np.int8
        assert all(int(v)==saturating_offset_add(int(a),int(b))
                   for a,b,v in zip(x,o,result))


def test_zero_is_identity_and_saturates_extremes():
    x=np.arange(-128,128,dtype=np.int16).repeat(2).astype(np.int8)
    assert np.array_equal(apply_offsets(x,np.zeros(512,dtype=np.int16)),x)
    off=np.zeros(512,dtype=np.int16);off[0]=-32768;off[1]=32767
    y=apply_offsets(x,off)
    assert y[0]==-128 and y[1]==127


def test_staging_commit_abort_denied_and_partial_bin():
    tab=OffsetTable()
    assert tab.stage(7,-30)
    assert not tab.stage(8,50)
    assert tab.active[7]==0
    tab.begin_bin()
    assert not tab.authorize()
    assert tab.pending==(7,-30) and tab.active[7]==0
    assert tab.calibrate(np.zeros(512,dtype=np.int8))[7]==0
    tab.end_bin()
    assert tab.authorize()
    assert tab.active[7]==-30 and tab.accepted_writes==1
    assert not tab.authorize()
    assert tab.rejected_requests==2
    assert tab.stage(7,55)
    tab.abort()
    assert tab.active[7]==-30 and tab.aborted_stages==1
    tab.reset()
    assert tab.active[7]==-30  # unchanged physical array contents
    assert tab.read_channel(7)==0 and tab.accepted_writes==0
    assert tab.calibrate(np.zeros(512,dtype=np.int8))[7]==0


def test_causal_decoder_does_not_reset_history_when_offset_changes():
    # Explicit low-dimensional useful paths within the REAL unchanged 512x64
    # decoder graph; synthetic coefficients, no model-quality claim.
    w=np.zeros(TOTAL_WEIGHTS,dtype=np.int8)
    b=np.zeros(TOTAL_META,dtype=np.int32)
    shifts=np.zeros(TOTAL_META,dtype=np.uint8)
    w[0]=1                                # input ch0 -> project0
    w[32768+4]=1                         # newest tap DW1 ch0
    w[33088]=1                           # DW1 ch0 -> PW1 ch0
    w[37184+4]=1                         # newest tap DW2 ch0
    w[37504]=1                           # DW2 ch0 -> PW2 ch0
    w[41600]=1                           # PW2 ch0 -> head logit0
    model=FixedParameters(w,b,shifts)
    decoder=OffsetStreamingDecoder(model)
    src=np.zeros(512,dtype=np.int8)
    src[0]=1
    y0=decoder.push(src)
    assert int(y0[0])==1 and decoder.decoder.frame_count==1
    assert decoder.table.stage(0,3)
    decoder.table.begin_bin()
    assert not decoder.table.authorize()
    decoder.table.end_bin()
    assert decoder.table.authorize()
    for _ in range(3):
        assert decoder.push(src) is None
    y4=decoder.push(src)
    assert int(y4[0])==4
    assert decoder.decoder.frame_count==5


def test_range_invalid():
    for arg in [(-129,0),(128,0),(0,-32769),(0,32768)]:
        with pytest.raises(ValueError):saturating_offset_add(*arg)


def test_exporter_rejects_missing_neural_vectors(tmp_path):
    from scripts.export_integrated_vectors import make_vectors
    with pytest.raises(FileNotFoundError):
        make_vectors(tmp_path/'missing',tmp_path/'output')


def test_exporter_detects_real_logit_change_on_synthetic_fixture(tmp_path):
    # This validates the exporter machinery only; NOT a real-neural result.
    from scripts.export_integrated_vectors import make_vectors
    from neuroedge.speech.fixed_decoder import write_rtl_vectors,FixedStreamingDecoder
    w=np.zeros(TOTAL_WEIGHTS,dtype=np.int8)
    b=np.zeros(TOTAL_META,dtype=np.int32)
    s=np.zeros(TOTAL_META,dtype=np.uint8)
    w[0]=1;w[32768+4]=1;w[33088]=1;w[37184+4]=1;w[37504]=1;w[41600]=1
    p=FixedParameters(w,b,s)
    bins=np.zeros((8,512),dtype=np.int8)
    bins[:,0]=1
    expected=FixedStreamingDecoder(p).run(bins)
    folder=tmp_path/'synthetic';out=tmp_path/'export'
    write_rtl_vectors(folder,p,bins,expected)
    info=make_vectors(folder,out)
    assert info['changed_logits']>0
    assert info['baseline_logits']==82
    assert (out/'expected_commit.mem').is_file()
