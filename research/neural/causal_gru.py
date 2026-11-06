"""existing experimental GRU teacher/candidate; not the verified INT8 decoder.
Outputs logit at bins 0,4,8,..., with strictly causal recurrence.
The training/inference compute budget is NOT asserted equivalent to CompactCTC.
"""
import torch
from torch import nn

class CausalGRUCTC(nn.Module):
    def __init__(self, hidden=128, depth=2, classes=41, dropout=0.1):
        super().__init__()
        if hidden<=0 or depth<=0 or classes<=1 or not 0<=dropout<1:
            raise ValueError('invalid GRU dimensions')
        self.hidden=hidden
        self.proj=nn.Sequential(nn.Linear(512,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.gru=nn.GRU(hidden,hidden,num_layers=depth,batch_first=True,
                        dropout=dropout if depth>1 else 0.0,bidirectional=False)
        self.readout=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,classes))

    @staticmethod
    def output_lengths(n:torch.Tensor)->torch.Tensor:
        return (n+3)//4

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=3 or x.size(-1)!=512:
            raise ValueError('expected B,T,512')
        z=self.proj(x)
        h,_=self.gru(z)
        return self.readout(h[:,::4,:])
