"""Integer inference ABI for the FPGA/ASIC linear-classifier milestone."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

N = 8
X_SCALE = 16                # quantized standardized feature step = 1/16
W_SCALE = 32                # quantized logistic coefficient step = 1/32
SCORE_SCALE = X_SCALE * W_SCALE


@dataclass
class IntegerLinear:
    w: np.ndarray            # int8[8]
    b: int                   # signed int32 bias
    mean: np.ndarray         # float64[8] train-only normalization
    scale: np.ndarray        # float64[8] train-only normalization

    def quantize_inputs(self, raw_features: np.ndarray) -> np.ndarray:
        z = (np.asarray(raw_features, dtype=np.float64) - self.mean) / self.scale
        return np.clip(np.rint(z * X_SCALE), -128, 127).astype(np.int8)

    def scores(self, xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=np.int32)
        assert xq.shape[-1] == N
        # int64 in Python reference avoids overflow before checking 32-bit ABI.
        s = (xq.astype(np.int64) @ self.w.astype(np.int64)) + self.b
        if np.any(s < -(2**31)) or np.any(s >= 2**31):
            raise OverflowError('Signed 32-bit ASIC score overflow')
        return s.astype(np.int32)

    def predict(self, xq: np.ndarray) -> np.ndarray:
        return (self.scores(xq) >= 0).astype(np.int64)

    @classmethod
    def from_fitted(cls, sklearn_model, train_x: np.ndarray):
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (train_x - mean) / scale
        # Model is fit to normalized data already; this method only exports.
        _ = normalized
        coef = np.asarray(sklearn_model.coef_[0], dtype=np.float64)
        bias = float(sklearn_model.intercept_[0])
        if coef.shape != (N,):
            raise ValueError(f'Expected {N} coefficients')
        w = np.clip(np.rint(coef * W_SCALE), -128, 127).astype(np.int8)
        b = int(np.rint(bias * SCORE_SCALE))
        if abs(b) >= 2**31:
            raise OverflowError('Quantized bias overflows int32')
        return cls(w, b, mean, scale)
