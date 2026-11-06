"""Prevent Python/C/SystemVerilog host ABI register drift."""
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SV=(ROOT/'rtl/soc/neuroedge_host_adapter.sv').read_text()
HEADER=(ROOT/'firmware/include/neuroedge_host.h').read_text()
ALIASES={'WPTR':'WEIGHT_PTR','WDATA':'WEIGHT_DATA',
         'MPTR':'META_PTR','MBIAS':'META_BIAS','MSHIFT':'META_SHIFT',
         'BINS_DONE':'BINS_DONE','REQUESTS':'REQUESTS',
         'REJECTS':'REJECTS','WRITES':'WRITES','STAGES':'STAGES',
         'ABORTS':'ABORTS'}

def test_register_offsets_identical_and_unique():
    sv={k:int(v,16) for k,v in re.findall(r'\bREG_([A-Z_]+)\s*=\s*8\'h([0-9a-fA-F]+)',SV)}
    c={k:int(v,16) for k,v in re.findall(r'^#define\s+NEH_([A-Z_]+)\s+0x([0-9a-fA-F]+)u',HEADER,re.M)}
    assert len(sv)>=25 and len(set(sv.values()))==len(sv)
    for key,val in sv.items():
        assert c[ALIASES.get(key,key)]==val,(key,val)
    assert all(0<=x<128 and x%4==0 for x in sv.values())

def test_no_redefinition_of_golden_rtl():
    assert SV.count('neuroedge_speech_pipeline u_pipeline')==1
    assert 'module neuroedge_speech_decoder (' not in SV
    assert 'module neuroedge_offset_bank ' not in SV
