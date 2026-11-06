import torch
from neuroedge.backends import SRAMNMCBackend, StatisticalNVMBackend

def test_sram_backend_exact():
    x=torch.tensor([[1.,2.,-1.]])
    w=torch.tensor([[2.,1.],[0.,3.],[4.,-2.]])
    y=SRAMNMCBackend().matmul(x,w)
    assert torch.allclose(y, x@w)

def test_stat_nvm_is_deterministic_for_seed():
    x=torch.randn(2,4); w=torch.randn(4,3)
    a=StatisticalNVMBackend(seed=9).matmul(x,w)
    b=StatisticalNVMBackend(seed=9).matmul(x,w)
    assert torch.allclose(a,b)
