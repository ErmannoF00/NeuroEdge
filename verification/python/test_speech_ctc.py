from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
from torch.nn.functional import ctc_loss

from neuroedge.speech.ctc import NUM_CLASSES, TOKENS, ctc_collapse, ctc_min_length, edit_distance
from neuroedge.speech.dataset import (
    list_labeled_trials, read_trial, training_channel_stats, SpeechTrials, collate_ctc
)
from neuroedge.speech.model import CompactCTC
from neuroedge.speech.quant import quantize_weights, quantize_activation, project_int32


def make_file(path, *, n=3, level=0.0, ids=(1,2,3)):
    rng=np.random.default_rng(17)
    with h5py.File(path,'w') as f:
        for i in range(n):
            g=f.create_group(f'trial_{i:04d}')
            x=rng.normal(size=(72+4*i,512)).astype('float32')+level
            g.create_dataset('input_features',data=x)
            g.create_dataset('seq_class_ids',data=np.array([*ids,0,0],dtype=np.int32))
            g.attrs['n_time_steps']=len(x)
            g.attrs['seq_len']=len(ids)
            g.attrs['session']='test_session'
    return path


def test_vocab_ctc_repetition_and_edit():
    assert NUM_CLASSES==41
    assert TOKENS[0]=='BLANK' and TOKENS[-1]==' | '
    assert ctc_collapse([1,1,0,1,2,2,0]) == [1,1,2]
    assert ctc_min_length([1,1,2,2])==6
    assert edit_distance([1,2,3],[1,4,3])==1
    assert edit_distance([],[1,2])==2


def test_no_data_leak_and_target_padding(tmp_path):
    tr=make_file(tmp_path/'train.h5',level=0)
    va=make_file(tmp_path/'val.h5',level=99)
    a=list_labeled_trials(tr)
    b=list_labeled_trials(va)
    mean,std=training_channel_stats(a)
    assert abs(float(mean.mean())) < .2
    assert np.all(std > 0)
    x,y=read_trial(a[0]); assert x.shape==(72,512)
    assert list(y)==[1,2,3]  # trailing zeros were padding
    ds=SpeechTrials(a,mean,std)
    batches=collate_ctc([ds[0],ds[1]])
    X,lengths,targets,target_lengths=batches
    assert X.shape==(2,76,512) and lengths.tolist()==[72,76]
    assert target_lengths.tolist()==[3,3]
    assert targets.tolist()==[1,2,3,1,2,3]
    Xval=SpeechTrials(b,mean,std)[0][0]
    assert float(Xval.mean())>7  # out of distribution; val cannot alter normalization


def test_refuses_zero_blank_in_true_targets(tmp_path):
    f=make_file(tmp_path/'bad.h5',ids=(0,1,2))
    with pytest.raises(ValueError,match='Invalid phoneme'):
        read_trial(list_labeled_trials(f)[0])


def test_causal_temporal_conv_is_future_blind():
    torch.manual_seed(91)
    model=CompactCTC(hidden=64).eval()
    x=torch.randn(1,50,512)
    with torch.no_grad():
        all_out=model(x)
        prefix_out=model(x[:,:25])
    assert all_out.shape==(1,13,41)
    assert prefix_out.shape==(1,7,41)
    torch.testing.assert_close(all_out[:,:prefix_out.size(1)],prefix_out,atol=2e-6,rtol=2e-6)
    assert model.output_lengths(torch.tensor([1,2,3,4,5,50])).tolist()==[1,1,1,1,2,13]


def test_ctc_loss_finite_and_backward(tmp_path):
    tr=make_file(tmp_path/'train.h5')
    refs=list_labeled_trials(tr)
    mean,std=training_channel_stats(refs)
    x,lens,targets,tlens=collate_ctc([SpeechTrials(refs,mean,std)[0]])
    model=CompactCTC(hidden=16)
    logits=model(x)
    logprobs=logits.log_softmax(-1).transpose(0,1)
    outlens=model.output_lengths(lens)
    loss=ctc_loss(logprobs,targets,outlens,tlens,blank=0,zero_infinity=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.project.weight.grad is not None
    assert torch.isfinite(model.project.weight.grad).all()


def test_signed_projection_exact_int32():
    rng=np.random.default_rng(2)
    w=rng.normal(size=(64,512)).astype('float32')
    x=rng.normal(size=(512,)).astype('float32')
    wq,sw=quantize_weights(w)
    xq=quantize_activation(x)
    ans=project_int32(wq,xq)
    assert ans.shape==(64,) and ans.dtype==np.int32
    assert np.array_equal(ans,wq.astype(np.int64)@xq.astype(np.int64))
    assert len(sw)==64 and np.all(sw>0)
