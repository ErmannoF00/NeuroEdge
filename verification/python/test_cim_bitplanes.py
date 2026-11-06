import random
from neuroedge.memory.cim_bitplane_model import bitplane_dot

def test_perfect_ideal_bitplanes():
    for length in [1,8,64,512]:
        rng=random.Random(length)
        for _ in range(12):
            x=[rng.randint(-128,127) for i in range(length)]
            w=[rng.randint(-128,127) for i in range(length)]
            e=bitplane_dot(x,w)
            assert e.dot==e.exact_reference
            assert e.column_conversions==16

def test_adc_coarsening_is_observable():
    x=[1,3,5,8]; w=[127,-128,17,-5]
    ideal=bitplane_dot(x,w)
    coarse=bitplane_dot(x,w,column_step=32)
    assert ideal.dot==ideal.exact_reference
    assert coarse.quantized_conversions>0

def test_clipping_is_observable():
    x=[127]*32;w=[127]*32
    limited=bitplane_dot(x,w,column_limit=100)
    assert limited.clipped_conversions>0
    assert limited.dot!=limited.exact_reference
