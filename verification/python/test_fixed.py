import torch
from neuroedge.fixed import symmetric_quantize,dequantize,qmul_acc

def test_quant_roundtrip():
    x=torch.tensor([-1.0,-0.2,0.0,0.7,1.0]); q,s=symmetric_quantize(x,8); y=dequantize(q,s)
    assert torch.max(torch.abs(x-y)) < 0.02

def test_mac():
    a=torch.tensor([[1,2,3]],dtype=torch.int32); b=torch.tensor([[4,5,6]],dtype=torch.int32)
    assert int(qmul_acc(a,b))==32
