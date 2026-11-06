from pathlib import Path
import pytest
from scripts.characterize_nvm_csv import characterize


def test_characterization_requires_real_input_provenance(tmp_path):
    p=tmp_path/'characterization.csv'
    p.write_text('state,conductance_s,read_energy_j,program_energy_j,read_latency_s,program_latency_s\n'
                 '0,1e-6,1e-12,2e-9,2e-8,5e-7\n'
                 '0,1.1e-6,1.1e-12,2.1e-9,2.1e-8,5.1e-7\n'
                 '1,4e-6,1.5e-12,2.2e-9,2.4e-8,6e-7\n')
    with pytest.raises(ValueError):
        characterize(str(p),str(tmp_path/'bad.json'),'fixture','none')
    d=characterize(str(p),str(tmp_path/'out.json'),'SYNTHETIC_TEST_ONLY','0'*64)
    assert d['state_statistics']['0']['samples']==2
    assert d['state_statistics']['1']['samples']==1
    assert d['state_statistics']['1']['conductance_s']['mean']==4e-6
