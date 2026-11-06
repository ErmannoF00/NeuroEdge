"""Hardware-contract INT8 streaming implementation of CompactCTC.

SCOPE: inference-only, frame/bin accurate, no language model, no AFE, no
hardware adaptation and no claim of numerical equivalence to floating PyTorch.
Quantized output is exactly reproducible by the companion SystemVerilog core.

Input neural feature bins: 512 signed INT8, represented with scale 1/16.
Weights: signed INT8, power-of-two per-output-channel weight scale 2**-e.
Biases: signed INT32 in output units of 1/16.
Accumulate: signed INT32 (bounded by max fan-in 512 * 128 * 128).
Requantize: arithmetic floor division by 2**e, then add output-unit bias.
Hidden layers: project/pw1/pw2 use ReLU + clamp [0,127];
DW1/DW2 are SIGNED linear outputs, clamp [-128,127], with NO ReLU.
This matches torch CompactCTC: ReLU(pw(dw(...))) rather than pw(ReLU(dw(...))).
Head: no ReLU, saturate to signed INT16 (logit scale 1/16).
DW kernels: chronological tap [oldest..newest]; left padding zeros.
DW1 executes for input bins 0,2,4,... DW2 for DW1 bins 0,2,4,...
Outputs 41 logits at input bins 0,4,8,... .

A fixed activation scale sacrifices quality; compare PER to float before making
any accuracy, energy or novelty claims. No quantization-aware training yet.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

HIDDEN=64
INPUTS=512
CLASSES=41
TAPS=5
ACT_SCALE=1/16  # existing compatibility default, frac bits = 4
LAYER_ORDER=("project", "dw1", "pw1", "dw2", "pw2", "head")
# Flat row-major INT8 memory layout: 44,224 bytes total.
WEIGHT_OFFSETS={"project":0, "dw1":32768, "pw1":33088, "dw2":37184,
                "pw2":37504, "head":41600}
WEIGHT_COUNTS={"project":32768, "dw1":320, "pw1":4096, "dw2":320,
               "pw2":4096, "head":2624}
META_OFFSETS={"project":0, "dw1":64, "pw1":128, "dw2":192, "pw2":256, "head":320}
TOTAL_WEIGHTS=44224
TOTAL_META=361


def _qweight(rows: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    rows=np.asarray(rows,dtype=np.float64)
    if rows.ndim!=2 or not np.all(np.isfinite(rows)):
        raise ValueError("weights must be finite 2D")
    peak=np.max(np.abs(rows),axis=1)
    # Zero-weight rows use an unambiguous all-zero representation.
    exponent=np.clip(np.floor(np.log2(127 / np.maximum(peak, 1e-30))), 0,15).astype(np.uint8)
    q=np.clip(np.rint(rows*np.exp2(exponent.astype(np.float64))[:,None]),-127,127).astype(np.int8)
    return q,exponent


def _q_bias(arr: np.ndarray) -> np.ndarray:
    arr=np.asarray(arr,dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("nonfinite bias")
    val=np.rint(arr/ACT_SCALE)
    if np.any(np.abs(val)>2**31-1):
        raise OverflowError("bias exceeds signed 32-bit")
    return val.astype(np.int32)


def quantize_input(x_normalized: np.ndarray, frac_bits: int = 4) -> np.ndarray:
    x=np.asarray(x_normalized,dtype=np.float32)
    if x.shape[-1] != INPUTS or not np.isfinite(x).all():
        raise ValueError("features must be finite [...,512]")
    if not isinstance(frac_bits, int) or not 0 <= frac_bits <= 12:
        raise ValueError("Input fractional bits must be 0..12")
    return np.clip(np.rint(x * float(2**frac_bits)),-128,127).astype(np.int8)


@dataclass(frozen=True)
class FixedParameters:
    weights: np.ndarray
    biases: np.ndarray
    shifts: np.ndarray

    def __post_init__(self):
        if self.weights.dtype != np.int8 or self.weights.shape!=(TOTAL_WEIGHTS,):
            raise ValueError("INT8 flattened weights required")
        if self.biases.dtype != np.int32 or self.biases.shape!=(TOTAL_META,):
            raise ValueError("INT32 flattened biases required")
        if self.shifts.dtype != np.uint8 or self.shifts.shape!=(TOTAL_META,) or np.any(self.shifts>31):
            raise ValueError("UINT8 [0,31] row shifts required")

    def row_weights(self,phase: str)->np.ndarray:
        if phase not in LAYER_ORDER: raise KeyError(phase)
        nout=CLASSES if phase=='head' else HIDDEN
        nin=INPUTS if phase=='project' else (TAPS if phase in ('dw1','dw2') else HIDDEN)
        start=WEIGHT_OFFSETS[phase]
        return self.weights[start:start+nout*nin].reshape(nout,nin)

    def layer_metadata(self,phase: str)->tuple[np.ndarray,np.ndarray]:
        n=CLASSES if phase=='head' else HIDDEN
        a=META_OFFSETS[phase]
        return self.biases[a:a+n],self.shifts[a:a+n]


def quantize_model(model, profile: dict | None = None)->FixedParameters:
    """Extract exactly the six learned CompactCTC layers (hidden=64)."""
    if int(model.hidden)!=HIDDEN or int(model.n_classes)!=CLASSES:
        raise ValueError("Only trained CompactCTC(hidden=64,n_classes=41) supported")
    components={
        'project':model.project, 'dw1':model.dw1.conv, 'pw1':model.pw1,
        'dw2':model.dw2.conv, 'pw2':model.pw2, 'head':model.head,
    }
    profile = profile or {"input_frac_bits": 4,
                          "output_frac_bits": {n: 4 for n in LAYER_ORDER}}
    input_bits = int(profile["input_frac_bits"])
    out_bits = {n: int(profile["output_frac_bits"][n]) for n in LAYER_ORDER}
    if not 0 <= input_bits <= 12 or any(not 0 <= v <= 12 for v in out_bits.values()):
        raise ValueError("fractional bit positions must be 0..12")
    prior_bits = input_bits
    weights=[]; biases=[]; shifts=[]
    for name in LAYER_ORDER:
        op=components[name]
        w=op.weight.detach().cpu().numpy().astype(np.float64)
        if name in ('dw1','dw2'):
            w=w.reshape(HIDDEN,TAPS)
        elif name in ('pw1','pw2'):
            w=w.reshape(HIDDEN,HIDDEN)
        q,e=_qweight(w)
        # Weight represents qweight * 2**(-e). Input represents qin *
        # 2**(-prior_bits). Output is qout * 2**(-out_bits[name]).
        # Therefore qout = floor(acc / 2**(e+prior_bits-out_bits)) + bias_q.
        shift = e.astype(np.int32) + prior_bits - out_bits[name]
        if np.any(shift < 0) or np.any(shift > 31):
            raise ValueError(f"{name}: unsupported right-shift range {shift.min()}..{shift.max()}")
        b_float = op.bias.detach().cpu().numpy().astype(np.float64)
        b_val = np.rint(b_float * float(2**out_bits[name]))
        if not np.isfinite(b_val).all() or np.any(np.abs(b_val)>2**31-1):
            raise OverflowError(f"{name}: bias exceeds INT32")
        weights.append(q.reshape(-1))
        biases.append(b_val.astype(np.int32))
        shifts.append(shift.astype(np.uint8))
        prior_bits = out_bits[name]
    return FixedParameters(np.concatenate(weights),np.concatenate(biases),np.concatenate(shifts))


def _layer(x: np.ndarray, w: np.ndarray, bias:np.ndarray,shift:np.ndarray,
           head=False, signed_hidden=False)->np.ndarray:
    """One MAC/requantization layer; numpy matmul is INT64 reference."""
    if x.dtype != np.int8:
        raise TypeError("layer expected signed INT8")
    result=w.astype(np.int64) @ x.astype(np.int64)
    result=np.right_shift(result,shift.astype(np.int64)) + bias.astype(np.int64)
    if head:
        return np.clip(result,-32768,32767).astype(np.int16)
    if signed_hidden:
        return np.clip(result,-128,127).astype(np.int8)
    return np.clip(result,0,127).astype(np.int8)


class FixedStreamingDecoder:
    """Streaming, causal, no-lookahead inference reference matching RTL ordering."""
    def __init__(self,p:FixedParameters):
        self.p=p
        self.hist0=np.zeros((TAPS,HIDDEN),dtype=np.int8)
        self.hist1=np.zeros((TAPS,HIDDEN),dtype=np.int8)
        self.frame_count=0
        self.saturation_hits={name:0 for name in LAYER_ORDER}
        self.layer_outputs={name:0 for name in LAYER_ORDER}

    def _eval(self,name,x,head=False):
        b,s=self.p.layer_metadata(name)
        result=_layer(x,self.p.row_weights(name),b,s,head=head)
        self.saturation_hits[name]+=int(np.count_nonzero((result==32767)|(result==-32768))) if head else int(np.count_nonzero(result==127))
        self.layer_outputs[name]+=len(result)
        return result

    def push(self,features:np.ndarray)->np.ndarray|None:
        x=np.asarray(features)
        if x.shape!=(INPUTS,) or x.dtype!=np.int8:
            raise ValueError("push expects signed INT8 shape (512,)")
        project=self._eval('project',x)
        self.hist0[:-1]=self.hist0[1:].copy()
        self.hist0[-1]=project
        result=None
        if self.frame_count%2==0:
            dw1=self._eval_dw('dw1',self.hist0)
            pw1=self._eval('pw1',dw1)
            self.hist1[:-1]=self.hist1[1:].copy()
            self.hist1[-1]=pw1
            if self.frame_count%4==0:
                dw2=self._eval_dw('dw2',self.hist1)
                pw2=self._eval('pw2',dw2)
                result=self._eval('head',pw2,head=True)
        self.frame_count+=1
        return result

    def _eval_dw(self,name,hist):
        w=self.p.row_weights(name).astype(np.int64)
        b,s=self.p.layer_metadata(name)
        # Elementwise channel-wise five-tap convolution with causal history.
        acc=np.sum(w*hist.T.astype(np.int64),axis=1,dtype=np.int64)
        # Depthwise output is signed linear, NOT ReLU. ReLU only after PW.
        y=np.clip(np.right_shift(acc,s.astype(np.int64))+b.astype(np.int64),-128,127).astype(np.int8)
        self.saturation_hits[name]+=int(np.count_nonzero((y==127)|(y==-128)))
        self.layer_outputs[name]+=len(y)
        return y

    def run(self,bins:np.ndarray)->np.ndarray:
        x=np.asarray(bins)
        if x.ndim!=2 or x.shape[1]!=INPUTS or x.dtype!=np.int8:
            raise ValueError("run expects int8 [T,512]")
        outputs=[y for binval in x if (y:=self.push(binval)) is not None]
        return np.stack(outputs).astype(np.int16) if outputs else np.empty((0,CLASSES),dtype=np.int16)


def write_hex(path:Path,arr:np.ndarray,digits:int)->None:
    vals=np.asarray(arr).reshape(-1)
    path.parent.mkdir(parents=True,exist_ok=True)
    mask=(1<<(4*digits))-1
    with path.open('w') as f:
        for val in vals:
            f.write(f'{int(val)&mask:0{digits}x}\n')


def write_rtl_vectors(directory:Path,p:FixedParameters,features:np.ndarray,expected:np.ndarray)->None:
    if features.dtype!=np.int8 or features.ndim!=2 or features.shape[1]!=INPUTS:
        raise ValueError('bad features')
    if expected.dtype!=np.int16 or expected.shape!=((len(features)+3)//4,CLASSES):
        raise ValueError('bad expected logit sequence')
    directory.mkdir(parents=True,exist_ok=True)
    write_hex(directory/'all_weights.mem',p.weights,2)
    write_hex(directory/'all_biases.mem',p.biases,8)
    write_hex(directory/'all_shifts.mem',p.shifts,2)
    write_hex(directory/'input_frames.mem',features,2)
    write_hex(directory/'expected_logits.mem',expected,4)
