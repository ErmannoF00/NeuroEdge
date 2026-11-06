from pathlib import Path
import json
import h5py
import numpy as np
import pytest
import torch
from neuroedge.drift.monitor import (
    quantize_neural, excursion_scores, calibrated_thresholds,
    trial_bootstrap_difference, evaluate_trials,
)
from scripts.run_neural_monitor import run
from argparse import Namespace
import hashlib


def _write_hdf5(path: Path, shift: float):
    with h5py.File(path,'w') as f:
        for i in range(3):
            g=f.create_group(f'trial_{i:04d}')
            x=np.zeros((24,512),dtype=np.float32)
            x[:, :16] = shift
            if i==2:
                x[:, :4] = -shift
            g.create_dataset('input_features',data=x)
            g.attrs['n_time_steps']=24


def test_int8_signed_min_magnitude_and_ties_even():
    x=np.zeros((2,512),np.float32)
    x[0,0]=-8.0
    x[0,1]=2.0
    x[0,2]=-2.0
    x[0,3]=0.5/16
    q=quantize_neural(x,np.zeros(512,np.float32),np.ones(512,np.float32),4)
    assert list(q[0,:4]) == [-128,32,-32,0]
    assert excursion_scores(q,32).tolist()==[3,0]


def test_validation_and_train_calibration():
    assert calibrated_thresholds(np.zeros(20,dtype=np.int32))==(0,1)
    with pytest.raises(ValueError):
        quantize_neural(np.zeros((1,511),np.float32),np.zeros(512),np.ones(512),4)
    with pytest.raises(ValueError):
        excursion_scores(np.zeros((1,512),np.int16),32)
    with pytest.raises(ValueError):
        calibrated_thresholds(np.ones(2,dtype=np.int32))


def test_trial_cluster_ci_repeatability():
    x=(0., 1., 2.)
    y=(5., 6., 7.)
    assert trial_bootstrap_difference(x,y,42,200)==trial_bootstrap_difference(x,y,42,200)


def test_end_to_end_fixture(tmp_path):
    train,ref,shift=(tmp_path/n for n in ('train.hdf5','ref.hdf5','shift.hdf5'))
    _write_hdf5(train,0.)
    _write_hdf5(ref,0.)
    _write_hdf5(shift,3.5)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    ckpt=tmp_path/'model.pt'
    torch.save({'mean':torch.zeros(512),'std':torch.ones(512),
                'data_manifest':{'train_sha256':sha(train)}},ckpt)
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps({'input_frac_bits':4,'checkpoint_sha256':sha(ckpt),
                                   'calibration_hdf5_sha256':sha(train)}))
    out=tmp_path/'out'
    run(Namespace(train=train,reference_val=ref,shift_val=shift,checkpoint=ckpt,
                  profile=profile,out=out,max_cal_trials=3,max_evaluation_trials=0,
                  export_bins=12,bootstrap=200,seed=2026))
    result=json.loads((out/'activity_audit.json').read_text())
    assert result['hysteresis_low']==0
    assert result['hysteresis_high']==1
    assert result['reference']['requested_actions']==0
    assert result['shift_development']['requested_actions']>0
    assert result['difference_bin_weighted_mean_excursions']>0
    lines=(out/'rtl_vectors/features.mem').read_text().splitlines()
    assert len(lines)==12*512
    assert (out/'rtl_vectors/expected.txt').is_file()
    assert 'transcription' not in (out/'activity_audit.json').read_text()


def test_trial_reset_policy(tmp_path):
    p=tmp_path/'trials.hdf5'
    _write_hdf5(p,3.5)
    result=evaluate_trials(p,np.zeros(512,np.float32),np.ones(512,np.float32),
                           fraction_bits=4,magnitude_threshold=32,
                           low_threshold=0,high_threshold=1)
    assert result.trials==3
    assert result.requested_actions==sum(result.trial_requests)
    assert all(v>0 for v in result.trial_requests)
