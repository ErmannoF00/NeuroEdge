import pytest
from physical.studies.memory_banking_sweep import STORES,candidates

def test_logical_storage_accounting():
    assert sum(v['depth']*v['width'] for k,v in STORES.items() if k!='offset_valid_mask')==375341
    assert STORES['offset_valid_mask']['depth']==512

def test_bank_capacity_ceil_not_power_assumption():
    x=candidates(44224,8,1024,8)
    assert x['macro_count']==44
    assert x['logical_bits']==353792
    assert x['capacity_bits']==44*1024*8
    assert 0<x['capacity_utilization']<=1
    with pytest.raises(ValueError):candidates(12,8,0,8)
