"""Integration tests for future-only feedback and explicit communication consent."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder, CLASSES
from neuroedge.speech.ctc import ctc_collapse
from neuroedge.speech.synthetic_fixture import synthetic_features, synthetic_parameters
from neuroedge.system.streaming import (ConnectedStream, FeaturePacket,
                                        predicted_phonemes, proposal_from_completed_trial,
                                        run_trial_series)
from neuroedge.control.feedback_policy import FeedbackProposal


def test_strict_20ms_clock_no_duplicate_and_ctc_cadence():
    p = synthetic_parameters(); x = synthetic_features(32)
    sys = ConnectedStream(p)
    sys.begin_trial(trial_id=0, first_timestamp_us=0, allow_output=True)
    for i in range(32):
        y = sys.push(FeaturePacket(0, i, 20_000*i, x[i]))
        assert (y is not None) == (i % 4 == 0)
    result = sys.finish_trial()
    assert result.emitted_steps == 8 and result.bins == 32
    assert result.released and result.released_phonemes is not None


def test_reject_duplicate_missing_future_and_bad_shapes():
    p = synthetic_parameters(); x = synthetic_features(4)
    sys = ConnectedStream(p)
    sys.begin_trial(trial_id=0, first_timestamp_us=0)
    with pytest.raises(ValueError):
        sys.push(FeaturePacket(0, 0, 0, x[0].astype(np.int16)))
    sys.push(FeaturePacket(0, 0, 0, x[0]))
    for packet in (FeaturePacket(0, 0, 0, x[0]), FeaturePacket(0, 1, 19_999, x[1]),
                   FeaturePacket(1, 1, 20_000, x[1])):
        with pytest.raises(ValueError): sys.push(packet)
    for i in range(1,4): sys.push(FeaturePacket(0,i,i*20_000,x[i]))
    sys.finish_trial()
    with pytest.raises(ValueError): sys.begin_trial(trial_id=1, first_timestamp_us=60_000)


def test_no_mid_trial_update_even_with_old_trial_supervision():
    p=synthetic_parameters();x=synthetic_features(8)
    sys=ConnectedStream(p,feedback_mode='supervised')
    sys.begin_trial(trial_id=0,first_timestamp_us=0)
    sys.push(FeaturePacket(0,0,0,x[0]))
    proposal=FeedbackProposal(source_trial=0,available_after_trial=0,channel=0,
                              delta_q4=-4,provenance='posttrial-test')
    assert not sys.attempt_unsafe_commit(proposal)
    assert sys.table.accepted_writes == 0
    for i in range(1,4):sys.push(FeaturePacket(0,i,i*20_000,x[i]))
    sys.finish_trial()
    sys.begin_trial(trial_id=1,first_timestamp_us=80_000)
    assert sys.table.accepted_writes==0


def test_banked_projection_exact_through_complete_ctc():
    p=synthetic_parameters();x=synthetic_features(32)
    expected=FixedStreamingDecoder(p).run(x)
    for banks in (1,2,8,16,64):
        result=run_trial_series(p,[(x[i:i+4],None) for i in range(0,32,4)],
                                banks=banks)
        assert result['bins']==32 and result['outputs']==8
        # Direct core comparison proves ALL 328 logits, not just CTC argmax.
        from neuroedge.memory.banked_projection import BankedStreamingDecoder
        actual=BankedStreamingDecoder(p,banks=banks).run(x)
        assert np.array_equal(actual, expected)


def test_precommitted_q4_offset_has_exact_integer_effect():
    p=synthetic_parameters();x=synthetic_features(4)
    sys=ConnectedStream(p)
    assert sys.table.stage(0, -32768)
    assert sys.table.authorize()
    sys.begin_trial(trial_id=0,first_timestamp_us=0,allow_output=True)
    produced=[]
    for i in range(4):
        q=sys.push(FeaturePacket(0,i,i*20_000,x[i]))
        if q is not None: produced.append(q)
    sys.finish_trial()
    ref=x.copy();ref[:,0]=-128
    expected=FixedStreamingDecoder(p).run(ref)
    assert np.array_equal(np.stack(produced),expected)


def test_never_release_unarmed_neural_output():
    p=synthetic_parameters();x=synthetic_features(8)
    result=run_trial_series(p,[(x[:4], None),(x[4:],None)],armed_trials=frozenset({1}))
    assert result['records'][0]['released_phonemes'] is None
    assert result['records'][1]['released']
    assert result['release_control'].startswith('explicit')


def test_lagged_sham_not_updated_during_source_trial():
    p=synthetic_parameters();x=synthetic_features(12)
    report=run_trial_series(p,[(x[i:i+4],None) for i in range(0,12,4)],mode='sham')
    assert report['committed_writes']==2
    assert report['records'][0]['offset_writes']==0
    assert report['records'][1]['offset_writes']==1
    assert report['records'][2]['offset_writes']==2


def test_lagged_supervised_no_future_target_leakage():
    p=synthetic_parameters();x=synthetic_features(12)
    labels=[ctc_collapse(y.argmax(axis=1).tolist()) for y in np.split(FixedStreamingDecoder(p).run(x),3)]
    sys=ConnectedStream(p, feedback_mode='supervised', delay_trials=1)
    for t in range(3):
        sys.begin_trial(trial_id=t,first_timestamp_us=t*4*20_000)
        assert sys.table.accepted_writes == 0  # fixture has no reducible errors
        for i,v in enumerate(x[t*4:t*4+4]):sys.push(FeaturePacket(t,i,t*80_000+i*20_000,v))
        sys.finish_trial(revealed_target=labels[t])
    assert sys.table.accepted_writes==0


def test_candidate_search_fails_closed_on_invalid_target():
    p=synthetic_parameters();x=synthetic_features(4)
    decoder=FixedStreamingDecoder(p)
    with pytest.raises(ValueError):
        proposal_from_completed_trial(initial_state=decoder,raw_bins=x,
             known_target=[42],offsets=np.zeros(512,dtype=np.int16),channel=0)


def test_report_never_claims_clinical_or_physical_measurement():
    p=synthetic_parameters();x=synthetic_features(4)
    r=run_trial_series(p,[(x,None)])
    assert r['physical_measurements'] is False
    assert 'NOT arbitrary thoughts' in r['classification_task']


def test_banked_random_full_density_signed_weights_matches_all_328_logits():
    from neuroedge.speech.fixed_decoder import FixedParameters, TOTAL_WEIGHTS, TOTAL_META
    from neuroedge.memory.banked_projection import BankedStreamingDecoder, exact_banked_projection
    rng=np.random.default_rng(20261105)
    p=FixedParameters(
        weights=rng.integers(-128,128,TOTAL_WEIGHTS,dtype=np.int16).astype(np.int8),
        biases=rng.integers(-64,65,TOTAL_META,dtype=np.int32),
        shifts=rng.integers(1,10,TOTAL_META,dtype=np.uint8))
    x=rng.integers(-128,128,(32,512),dtype=np.int16).astype(np.int8)
    x[0,:]=np.int8(-128)
    x[1,:]=np.int8(127)
    expected=FixedStreamingDecoder(p).run(x)
    assert expected.shape==(8,41)
    assert expected.size==328
    for banks in (1,4,16,64):
        core=BankedStreamingDecoder(p,banks=banks)
        actual=core.run(x)
        assert np.array_equal(actual,expected)
        assert core.projection_counters.macs==32*32768
        assert core.projection_counters.weight_reads==32*32768
        assert core.projection_counters.compute_only_cycles==32*(64//banks)*512*2
    y0=exact_banked_projection(x[0],p,banks=64)
    assert y0.shape==(64,) and y0.dtype==np.int8


def test_supervised_policy_requires_label_and_does_not_invent_it():
    p=synthetic_parameters();x=synthetic_features(8)
    s=ConnectedStream(p,feedback_mode='supervised')
    for t in range(2):
        s.begin_trial(trial_id=t,first_timestamp_us=t*80_000)
        for i,feat in enumerate(x[4*t:4*t+4]):
            s.push(FeaturePacket(t,i,t*80_000+i*20_000,feat))
        s.finish_trial()  # NO oracle feedback
    assert s.table.accepted_writes == 0 and not s.queue


def test_invalid_bank_count_and_candidate_step_rejected():
    from neuroedge.memory.banked_projection import BankedStreamingDecoder
    p=synthetic_parameters()
    with pytest.raises(ValueError):BankedStreamingDecoder(p,banks=3)
    with pytest.raises(ValueError):ConnectedStream(p,step_q4=0)
    with pytest.raises(ValueError):ConnectedStream(p,feedback_channel=512)


def test_host_32_bin_packets_exactly_match_rtl_golden_328_logits(tmp_path):
    """Same params/bins that the existing complete RTL bench consumes."""
    from neuroedge.speech.synthetic_fixture import synthetic_parameters, synthetic_features
    from neuroedge.speech.fixed_decoder import FixedStreamingDecoder
    from scripts.generate_public_vectors import generate
    from scripts.run_connected_stream import load_hex_array, from_rtl_weights
    generate(tmp_path, bins=32)
    p=from_rtl_weights(tmp_path)
    x=synthetic_features(32)
    golden=load_hex_array(tmp_path/'expected_logits.mem', 328, 16, True).reshape(8,41)
    assert np.array_equal(FixedStreamingDecoder(p).run(x),golden)
    for banks in (None,8,64):
        host=ConnectedStream(p,banks=banks)
        host.begin_trial(trial_id=0,first_timestamp_us=0,allow_output=True)
        outputs=[]
        for i,vec in enumerate(x):
            y=host.push(FeaturePacket(0,i,i*20_000,vec))
            if y is not None: outputs.append(y)
        host.finish_trial()
        assert np.array_equal(np.stack(outputs), golden)


def test_unarmed_step_never_reveals_logits_even_to_caller():
    p=synthetic_parameters();x=synthetic_features(4)
    sys=ConnectedStream(p)
    sys.begin_trial(trial_id=0,first_timestamp_us=0,allow_output=False)
    for i,v in enumerate(x):
        assert sys.push(FeaturePacket(0,i,i*20_000,v)) is None
    result=sys.finish_trial()
    assert result.emitted_steps==1  # Internal decoding did occur
    assert result.released_phonemes is None


def test_synthetic_hdf5_adapter_uses_frozen_checkpoint_normalizer(tmp_path):
    """Schema test only: generated HDF5 is not a neural accuracy benchmark."""
    import torch, h5py
    from neuroedge.speech.model import CompactCTC
    from scripts.run_connected_stream import load_real_trials
    model=CompactCTC(hidden=64)
    ck=tmp_path/'untrained.pt'
    torch.save({'architecture':{'hidden':64},'state_dict':model.state_dict(),
                'mean':torch.full((512,),2.0),
                'std':torch.full((512,),2.0)},ck)
    f=tmp_path/'schema.hdf5'
    with h5py.File(f,'w') as h:
        g=h.create_group('trial_a')
        g.attrs['n_time_steps']=8
        g.attrs['seq_len']=1
        g.create_dataset('input_features',data=np.full((8,512),4.0,dtype=np.float32))
        g.create_dataset('seq_class_ids',data=np.array([1],dtype=np.int64))
    params, items = load_real_trials(hdf5=f,checkpoint=ck,profile_path=None,
                                     weights_dir=None,max_trials=1)
    assert len(items)==1
    assert items[0][0].dtype==np.int8
    assert np.all(items[0][0]==16)  # (4.0 - train_mean 2.0)/std 2.0 * 16
    assert items[0][1]==[1]
