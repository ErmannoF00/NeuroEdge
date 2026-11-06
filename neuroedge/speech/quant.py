"""Explicit symmetric INT8 projection quantization and INT32 reference."""
import numpy as np


def quantize_weights(weight: np.ndarray):
    """Per-output-channel symmetric signed INT8 weights, [O,512]."""
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] != 512 or not np.isfinite(w).all():
        raise ValueError('Weights must be finite [O,512]')
    scales = np.maximum(np.max(np.abs(w), axis=1) / 127.0, 1e-12)
    q = np.clip(np.rint(w / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def quantize_activation(x: np.ndarray, scale: float = 1.0 / 16.0):
    """Fixed scale across frames; inputs already train-normalized and clipped."""
    v = np.asarray(x, dtype=np.float32)
    if v.shape != (512,) or not np.isfinite(v).all() or scale <= 0:
        raise ValueError('Invalid activation')
    return np.clip(np.rint(v/scale), -128, 127).astype(np.int8)


def project_int32(weight_q: np.ndarray, feature_q: np.ndarray):
    """Products sum in INT64 reference, proven to fit signed INT32 for 512xINT8."""
    w = np.asarray(weight_q)
    x = np.asarray(feature_q)
    if w.ndim != 2 or w.shape[1] != 512 or x.shape != (512,):
        raise ValueError('Invalid projection shape')
    if w.dtype != np.int8 or x.dtype != np.int8:
        raise ValueError('Expected signed INT8 operands')
    sums = w.astype(np.int64) @ x.astype(np.int64)
    if np.any(sums < -(2**31)) or np.any(sums > 2**31-1):
        raise OverflowError('INT32 overflow')
    return sums.astype(np.int32)
