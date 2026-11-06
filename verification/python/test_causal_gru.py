import pytest

torch=pytest.importorskip('torch')
from research.neural.causal_gru import CausalGRUCTC

def test_causal_gru_shapes_and_length():
    m=CausalGRUCTC(hidden=16,depth=1,dropout=0).eval()
    x=torch.randn(2,17,512)
    with torch.no_grad():
        y=m(x)
    assert y.shape==(2,5,41)
    assert m.output_lengths(torch.tensor([1,4,5,17])).tolist()==[1,1,2,5]

def test_causal_gru_prefix_invariance():
    torch.manual_seed(42)
    m=CausalGRUCTC(hidden=16,depth=2,dropout=0).eval()
    a=torch.randn(1,9,512)
    b=torch.cat([a,torch.randn(1,11,512)],dim=1)
    with torch.no_grad():
        short=m(a);long=m(b)
    assert torch.allclose(short,long[:,:3],atol=1e-5,rtol=1e-5)
