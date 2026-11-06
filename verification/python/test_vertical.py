from pathlib import Path
import json
import numpy as np
import pytest
from neuroedge.vertical.dataset import synthetic_fixture
from neuroedge.vertical.quantized_linear import IntegerLinear
from neuroedge.vertical.export import export_abi
from scripts.run_vertical import parse_ids, main


def test_no_overlapping_subjects(tmp_path):
    with pytest.raises(ValueError, match='Subject leakage'):
        main(['--train-subjects','1,2', '--val-subjects','2,3',
              '--test-subjects','4', '--out',str(tmp_path)])


def test_synthetic_reproducibility():
    a = synthetic_fixture([1,2]); b = synthetic_fixture([1,2])
    assert np.array_equal(a.X,b.X)
    assert np.array_equal(a.y,b.y)
    assert a.provenance == 'synthetic_demo_only'


def test_int8_sign_handling_and_score_export(tmp_path):
    X=np.array([[0,-1,2,-3,4,-5,6,-7]], dtype=float)
    m=IntegerLinear(np.array([-5,4,-3,2,-1,5,-2,6], dtype=np.int8),
                    -77, np.zeros(8),np.ones(8))
    xq=m.quantize_inputs(X)
    expected=int((xq.astype(np.int64)@m.w.astype(np.int64))[0]-77)
    assert m.scores(xq)[0] == expected
    out=export_abi(m,X,str(tmp_path),max_vectors=1)
    word=int((tmp_path/'vectors.mem').read_text().strip(),16)
    score=(word & 0xffffffff)
    if score >= 2**31: score-=2**32
    assert score == expected
    assert out['test_vector_count']==1
    assert "-32'sd77" in (tmp_path/'model_params.vh').read_text()


def test_full_export_matches_python(tmp_path):
    root=tmp_path/'vertical'
    assert main(['--out',str(root)])==0
    metrics=json.loads((root/'metrics.json').read_text())
    assert metrics['scientific_results_valid'] is False
    p=root/'rtl_vectors'
    params=json.loads((p/'model.json').read_text())
    weights=[int(x.strip(),16) for x in (p/'weights.mem').read_text().splitlines()]
    assert len(weights)==params['input_features']==8
    assert len((p/'vectors.mem').read_text().splitlines())==params['test_vector_count']
    assert (p/'manifest.sha256').exists()
