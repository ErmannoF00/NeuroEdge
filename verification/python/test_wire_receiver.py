"""Binary acquisition-to-host receiver and fail-closed trial ownership."""
from __future__ import annotations
import numpy as np
import pytest
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder
from neuroedge.speech.synthetic_fixture import synthetic_features, synthetic_parameters
from neuroedge.system.packet_wire import WirePacket, encode, FLAG_BEGIN_TRIAL, FLAG_END_TRIAL
from neuroedge.system.wire_receiver import WireStreamReceiver
from neuroedge.system.streaming import ConnectedStream


def wire(t,i,ts,arr,flags=0):
    return encode(WirePacket(flags,t,i,ts,arr))


def test_wire_receiver_full32_packets_all_328_logits():
    p=synthetic_parameters();x=synthetic_features(32)
    host=WireStreamReceiver(ConnectedStream(p,banks=64))
    visible=[]
    for i,row in enumerate(x):
        flags=(FLAG_BEGIN_TRIAL if i==0 else 0)|(FLAG_END_TRIAL if i==31 else 0)
        r=host.receive(wire(0,i,i*20_000,row,flags),externally_armed=(i==0))
        if r.released_logits is not None: visible.append(r.released_logits)
        if i==31:
            assert r.completed_trial.emitted_steps==8
            assert r.completed_trial.released
        else:
            assert r.completed_trial is None
    assert host.packet_count==32
    assert np.array_equal(np.stack(visible),FixedStreamingDecoder(p).run(x))


def test_wire_receiver_private_trial_reveals_no_logits():
    p=synthetic_parameters();x=synthetic_features(4)
    host=WireStreamReceiver(ConnectedStream(p))
    for i,row in enumerate(x):
        f=(FLAG_BEGIN_TRIAL if i==0 else 0)|(FLAG_END_TRIAL if i==3 else 0)
        r=host.receive(wire(0,i,i*20_000,row,f))
        assert r.released_logits is None
    assert r.completed_trial.released_phonemes is None
    assert r.completed_trial.emitted_steps==1


def test_corruption_and_bad_sequence_fail_without_advancing_state():
    p=synthetic_parameters();x=synthetic_features(8)
    host=WireStreamReceiver(ConnectedStream(p))
    bad=bytearray(wire(0,0,0,x[0],FLAG_BEGIN_TRIAL));bad[123]^=1
    with pytest.raises(ValueError):host.receive(bytes(bad))
    assert host.packet_count==0 and not host.active
    with pytest.raises(ValueError):host.receive(wire(0,0,0,x[0],0))
    host.receive(wire(0,0,0,x[0],FLAG_BEGIN_TRIAL))
    with pytest.raises(ValueError):host.receive(wire(0,2,40_000,x[2],0))
    with pytest.raises(ValueError):host.receive(wire(0,1,20_000,x[1],FLAG_BEGIN_TRIAL))
    with pytest.raises(ValueError):host.receive(wire(0,1,20_000,x[1],0),externally_armed=True)
    host.abort()
    assert host.aborted_trials==1 and not host.active
    host.receive(wire(1,0,80_000,x[4],FLAG_BEGIN_TRIAL))
    for i in range(1,4):
        f=FLAG_END_TRIAL if i==3 else 0
        end=host.receive(wire(1,i,80_000+i*20_000,x[4+i],f))
    assert end.completed_trial.trial_id==1


def test_unsolicited_labels_are_rejected_before_inference():
    p=synthetic_parameters(); x=synthetic_features(4)
    host=WireStreamReceiver(ConnectedStream(p,feedback_mode='supervised'))
    with pytest.raises(ValueError):
        host.receive(wire(0,0,0,x[0],FLAG_BEGIN_TRIAL),revealed_target=[1])
    assert not host.active and host.packet_count==0
