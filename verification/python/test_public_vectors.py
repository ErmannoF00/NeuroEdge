"""No external checkpoint or dataset needed for the public fixture."""
from pathlib import Path
import numpy as np
from scripts.generate_public_vectors import generate
from scripts.export_integrated_vectors import load_hex, make_vectors
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder, FixedParameters, TOTAL_WEIGHTS, TOTAL_META


def test_fixture_is_reproducible_and_closed_under_integer_oracle(tmp_path: Path):
    a = tmp_path / 'first'
    b = tmp_path / 'second'
    d1 = generate(a)
    d2 = generate(b)
    assert d1 == d2
    assert d1['bins'] == 32 and d1['logits'] == 328
    w = load_hex(a / 'all_weights.mem', TOTAL_WEIGHTS, 8, np.int8)
    bias = load_hex(a / 'all_biases.mem', TOTAL_META, 32, np.int32)
    shifts = load_hex(a / 'all_shifts.mem', TOTAL_META, 8, np.uint8)
    x = load_hex(a / 'input_frames.mem', 32 * 512, 8, np.int8).reshape(32, 512)
    expected = load_hex(a / 'expected_logits.mem', 328, 16, np.int16).reshape(8, 41)
    actual = FixedStreamingDecoder(FixedParameters(w, bias, shifts)).run(x)
    assert np.array_equal(actual, expected)
    evidence = make_vectors(a, tmp_path / 'calibrated')
    assert evidence['zero_offset_exact']
    assert evidence['changed_logits'] > 0
    assert 'synthetic' in evidence['source']
