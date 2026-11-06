"""Produce exactly the memory/parameter/vector files consumed by RTL."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .quantized_linear import N, X_SCALE, W_SCALE, IntegerLinear


def export_abi(model: IntegerLinear, features: np.ndarray, output: str,
               max_vectors: int = 32) -> dict:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    qx = model.quantize_inputs(features)[:max_vectors]
    scores = model.scores(qx)
    (root/'weights.mem').write_text(''.join(f'{int(v) & 0xff:02x}\n' for v in model.w))
    # 8 input bytes (high bits, natural feature order) + int32 signed score
    with (root/'vectors.mem').open('w') as f:
        for row, score in zip(qx, scores):
            pack = 0
            for feature in row:
                pack = (pack << 8) | (int(feature) & 0xff)
            pack = (pack << 32) | (int(score) & 0xffffffff)
            f.write(f'{pack:024x}\n')
    bias_literal = (f"-32'sd{-model.b}" if model.b < 0 else f"32'sd{model.b}")
    (root/'model_params.vh').write_text(
        '// Generated; do not edit. For synthesizable RTL and TB.\n'
        f'`define MODEL_N {N}\n'
        f'`define MODEL_BIAS {bias_literal}\n'
        f'`define MODEL_NUM_VECTORS {len(qx)}\n'
    )
    export = {
        'abi': 'neuroedge-linear-v1',
        'input_features': N, 'input_format': 'signed int8',
        'weight_format': 'signed int8', 'output_format': 'signed int32',
        'weight_scale': W_SCALE, 'input_scale': X_SCALE,
        'bias': model.b, 'weight_values': [int(x) for x in model.w],
        'train_mean': model.mean.tolist(), 'train_std': model.scale.tolist(),
        'test_vector_count': len(qx),
        'latency': '8 accepted input transfers plus registered output',
        'interface': 'one feature per accepted clock, backpressured result',
    }
    (root/'model.json').write_text(json.dumps(export, indent=2) + '\n')
    sha = hashlib.sha256()
    for name in ('weights.mem', 'vectors.mem', 'model_params.vh', 'model.json'):
        sha.update((root/name).read_bytes())
    (root/'manifest.sha256').write_text(sha.hexdigest() + '  exported_bundle\n')
    return export
