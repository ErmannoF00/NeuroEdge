"""Offline tests use synthetic structural fixtures, never pretend to be BCI data."""
import pytest
from neuroedge.datasets.b2t25_contract import inspect_file, load_trial_features

h5py = pytest.importorskip('h5py')
np = pytest.importorskip('numpy')


def _example(path, shape=(7, 512), count=5, with_labels=True):
    with h5py.File(path, 'w') as f:
        g = f.create_group('trial_001')
        g.create_dataset('input_features', data=np.zeros(shape, dtype=np.float32))
        g.attrs['n_time_steps'] = count
        g.attrs['session'] = 'fixture_2026_10'
        g.attrs['block_num'] = 1
        g.attrs['trial_num'] = 1
        if with_labels:
            g.create_dataset('seq_class_ids', data=np.array([1, 2], dtype=np.int32))
            g.create_dataset('transcription', data=np.array([ord('h'), ord('i'), 0], dtype=np.uint8))


def test_shape_and_metadata(tmp_path):
    path = tmp_path/'data_train.hdf5'
    _example(path)
    t = list(inspect_file(path))
    assert len(t) == 1 and t[0].feature_shape == (7, 512)
    assert t[0].n_time_steps == 5 and t[0].has_phoneme_labels
    assert load_trial_features(path, 'trial_001').shape == (5, 512)


def test_unlabeled_test_split_is_legal(tmp_path):
    path = tmp_path/'data_test.hdf5'
    _example(path, with_labels=False)
    t = next(inspect_file(path))
    assert not t.has_phoneme_labels and not t.has_transcription


def test_bad_feature_width_rejected(tmp_path):
    path = tmp_path/'wrong.hdf5'
    _example(path, shape=(7, 511))
    with pytest.raises(ValueError, match='expected'):
        list(inspect_file(path))


def test_bad_time_count_rejected(tmp_path):
    path = tmp_path/'wrongtime.hdf5'
    _example(path, count=99)
    with pytest.raises(ValueError, match='invalid n_time_steps'):
        list(inspect_file(path))


def test_nonfinite_input_rejected(tmp_path):
    path = tmp_path/'nan.hdf5'
    _example(path)
    with h5py.File(path, 'r+') as f:
        f['trial_001/input_features'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        load_trial_features(path, 'trial_001')
