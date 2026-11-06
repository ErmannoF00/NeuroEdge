from dataclasses import replace
import numpy as np
import pytest
from neuroedge.system.packet_wire import (
    WirePacket,encode,decode,to_feature_packet,FLAG_BEGIN_TRIAL,FLAG_END_TRIAL,
)


def test_wire_roundtrip_and_feature_order_signed():
    x=np.arange(512,dtype=np.int16)%256-128
    p=WirePacket(FLAG_BEGIN_TRIAL|FLAG_END_TRIAL,7,3,123456789,x.astype(np.int8))
    b=encode(p)
    assert len(b)==540
    read=decode(b)
    assert read.flags==3 and read.timestamp_us==p.timestamp_us
    assert np.array_equal(read.features,x.astype(np.int8))
    assert to_feature_packet(read).bin_index==3


def test_wire_rejects_corruption_version_flags_and_length():
    p=WirePacket(0,0,0,0,np.zeros(512,dtype=np.int8))
    blob=encode(p)
    for j in (0,4,9,24,400,536):
        bad=bytearray(blob);bad[j]^=1
        with pytest.raises(ValueError):decode(bytes(bad))
    with pytest.raises(ValueError):decode(blob[:-1])
    with pytest.raises(ValueError):encode(replace(p,flags=0x4))
    with pytest.raises(ValueError):encode(replace(p,features=np.zeros(511,dtype=np.int8)))


def test_wire_does_not_contain_release_authorization():
    p=WirePacket(FLAG_BEGIN_TRIAL,0,0,0,np.zeros(512,dtype=np.int8))
    assert decode(encode(p)).flags==FLAG_BEGIN_TRIAL
    # Even forged reserved flags are rejected rather than interpreted as
    # consent to decode/release private attempted/inner speech.
    with pytest.raises(ValueError):encode(replace(p,flags=0x8000))


def test_binary_packet_through_host_and_complete_ctc_matches_328_logit_reference():
    from neuroedge.speech.synthetic_fixture import synthetic_parameters, synthetic_features
    from neuroedge.speech.fixed_decoder import FixedStreamingDecoder
    from neuroedge.system.streaming import ConnectedStream
    weights=synthetic_parameters(); data=synthetic_features(32)
    host=ConnectedStream(weights,banks=64)
    host.begin_trial(trial_id=0,first_timestamp_us=0,allow_output=True)
    visible=[]
    for i,row in enumerate(data):
        flags=(FLAG_BEGIN_TRIAL if i==0 else 0)|(FLAG_END_TRIAL if i==31 else 0)
        p=WirePacket(flags,0,i,i*20_000,row)
        out=host.push(to_feature_packet(decode(encode(p))))
        if out is not None: visible.append(out)
    result=host.finish_trial()
    assert result.bins==32 and result.emitted_steps==8
    assert np.array_equal(np.stack(visible),FixedStreamingDecoder(weights).run(data))
