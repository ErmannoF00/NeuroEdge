"""Nonfunctional source-level checks; runtime RTL Icarus gate required separately."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'rtl/speech/sync/neuroedge_speech_decoder.sv'
SRAM = ROOT / 'rtl/speech/sync/neuroedge_sram_1rw.sv'
OLD = ROOT / 'rtl/speech/neuroedge_full_decoder.sv'


def test_sram_is_synchronous():
    text = SRAM.read_text()
    assert 'always_ff @(posedge clk)' in text
    assert 'if(we) mem[addr] <= wdata' in text
    assert 'else rdata <= mem[addr]' in text
    assert 'assign rdata = mem[' not in text


def test_full_core_has_fetch_stage():
    text = CORE.read_text()
    assert 'S_FETCH: state<=S_MAC;' in text
    assert text.count('state<=S_MAC;') == 1
    assert 'weight_read' in text and 'meta_read' in text
    assert 'neuroedge_sram_1rw' in text
    assert 'weights[weight_index]' not in text
    assert 'shift_mem[meta_index]' not in text
    assert 'S_EMIT: if(logit_valid && logit_ready)' in text


def test_parameter_dimensions_and_graph_match():
    s = CORE.read_text()
    r = OLD.read_text()
    for token in (
        'localparam integer WEIGHT_COUNT=44224;',
        'localparam integer META_COUNT=361;',
        'localparam integer INPUTS=512;',
        'localparam integer CLASSES=41;',
        'PROJECT: projected[row]<=clamp_relu8(biased_sum);',
        'DW1: depth1[row]<=clamp_signed8(biased_sum);',
        'PW1: point1[row]<=clamp_relu8(biased_sum);',
        'DW2: depth2[row]<=clamp_signed8(biased_sum);',
        'PW2: point2[row]<=clamp_relu8(biased_sum);',
        'logit_data<=clamp_int16(biased_sum);',
    ):
        assert token in s and token in r


def test_per_bin_cycle_guard_exists():
    t = (ROOT / 'verification/rtl/speech/tb_full_decoder_sync.sv').read_text()
    assert 'CYCLES_PER_BIN_AT_25MHZ=500000' in t
    assert 'SYNC_SRAM_FULL_DECODER_RTL_PASS' in t
    assert 'SYNC_SRAM_BIN_DEADLINE_FAIL' in t
