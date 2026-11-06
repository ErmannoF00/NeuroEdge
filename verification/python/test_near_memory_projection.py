"""Verify per-row near-memory integer dot mirrors project-stage arithmetic.

Requires numpy and the existing NeuroEdge source on PYTHONPATH.
"""
import sys, random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from neuroedge.memory.near_memory_reference import int8_dot
from neuroedge.speech.fixed_decoder import _layer
from neuroedge.speech.synthetic_fixture import synthetic_parameters, synthetic_features

def test_projection_exact_requantization():
    p=synthetic_parameters(); w=p.row_weights('project'); bias,shift=p.layer_metadata('project')
    x=synthetic_features(4)[0]
    oracle=_layer(x,w,bias,shift)
    calc=[]
    for i in range(64):
        acc,_=int8_dot([int(v) for v in x],[int(v) for v in w[i]])
        quant=(acc >> int(shift[i])) + int(bias[i])
        calc.append(max(0,min(127,quant)))
    assert np.array_equal(np.asarray(calc,dtype=np.int8),oracle)

def test_dense_random_requantization():
    rng=random.Random(182)
    x=np.array([rng.randint(-128,127) for _ in range(512)],dtype=np.int8)
    weights=np.array([[rng.randint(-128,127) for _ in range(512)] for _ in range(8)],dtype=np.int8)
    biases=np.array([rng.randint(-200,200) for _ in range(8)],dtype=np.int32)
    shifts=np.array([rng.randint(10,18) for _ in range(8)],dtype=np.uint8)
    oracle=_layer(x,weights,biases,shifts)
    result=[]
    for i in range(8):
        dot,_=int8_dot([int(v) for v in x],[int(v) for v in weights[i]])
        result.append(max(0,min(127,(dot >> int(shifts[i])) + int(biases[i]))))
    assert np.array_equal(np.array(result,dtype=np.int8),oracle)
