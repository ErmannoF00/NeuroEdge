import random
import pytest
from neuroedge.memory.near_memory_reference import int8_dot

def test_extreme_products():
    assert int8_dot([-128]*512,[-128]*512)[0]==512*16384
    assert int8_dot([127]*512,[-128]*512)[0]==-512*16256

def test_counter_contract():
    result,c=int8_dot([1,2,-3],[4,-5,6])
    assert result==-24
    assert (c.macs,c.weight_reads,c.input_handshakes,c.minimum_compute_cycles)==(3,3,3,6)

@pytest.mark.parametrize('length',[1,5,8,64,512])
def test_seeded(length):
    rng=random.Random(length)
    x=[rng.randint(-128,127) for _ in range(length)]
    w=[rng.randint(-128,127) for _ in range(length)]
    assert int8_dot(x,w)[0]==sum(a*b for a,b in zip(x,w))

def test_reject_invalid():
    for a,b in [([],[]),([2],[128]),([0],[]),([0]*513,[0]*513)]:
        with pytest.raises(ValueError): int8_dot(a,b)
