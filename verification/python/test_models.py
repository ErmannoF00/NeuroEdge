import torch
from neuroedge.models import TinyQNN,TinySNN,DiagonalSSM
from neuroedge.data import synthetic_bci,delta_events

def test_shapes():
    x,_=synthetic_bci(8,32,12,4).tensors
    assert TinyQNN(12,16,4)(x).shape==(8,4)
    assert TinySNN(12,16,4)(delta_events(x)).shape==(8,4)
    assert DiagonalSSM(12,16,4)(x).shape==(8,4)
