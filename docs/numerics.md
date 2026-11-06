# Fixed-point numerical and timing contract

## Input correction

For `j` in `[0, 511]`, with signed integers `x[j]` in `[-128, 127]` and a committed coefficient `d[j]` in `[-32768, 32767]`:

```
sum17[j] = sign_extend_17(x[j]) + sign_extend_17(d[j])
cal[j]   = min(127, max(-128, sum17[j]))
```

`d` shares the original feature's Q4 integer units; a value of +16 represents +1 input unit. The coefficient is NOT a learned model weight, and the correction is not an SRAM-based MAC. Registered memory address/data timing is exercised by the RTL bridge tests.

## Decoder contract

| Stage | Dimensions | Weights | Flat weight base | Metadata base | Clamp |
|---|---|---:|---:|---:|---|
| Projection | 512 → 64 | 32,768 | 0 | 0 | ReLU [0,127] |
| Depthwise 1 | 64 × 5 taps | 320 | 32768 | 64 | signed INT8 |
| Pointwise 1 | 64 → 64 | 4,096 | 33088 | 128 | ReLU [0,127] |
| Depthwise 2 | 64 × 5 taps | 320 | 37184 | 192 | signed INT8 |
| Pointwise 2 | 64 → 64 | 4,096 | 37504 | 256 | ReLU [0,127] |
| CTC head | 64 → 41 | 2,624 | 41600 | 320 | signed INT16 |

For each output row, `acc = sum(signed_int8_activation * signed_int8_weight)` with signed 32-bit accumulator; output = arithmetic-right-shift by the row's 5-bit shift followed by signed INT32 bias, then layer-specific clipping. Negative arithmetic shift follows two's-complement floor behavior. There is **no ReLU between a depthwise layer and its pointwise layer**.

Causal depthwise convolutions use length-five history, zero initialized, with stride two at each depthwise layer. One 41-logit vector is emitted at input indices 0,4,8,... . Backpressure must hold valid output data stable until accepted.

## Time accounting

Input bin cadence is **20 ms**, a data-format expectation. The tests apply a hypothetical **25 MHz** clock. A 500,000-cycle deadline corresponds to 20 ms at that *assumed* clock. The provided WSL log reported a worst 90,195-cycle bin under one specific RTL test sequence (3.6078 ms if the design could physically meet 25 MHz). This is a **pre-synthesis simulation result**, not proven real-time hardware frequency or physical timing closure.
