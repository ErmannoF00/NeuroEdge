#!/usr/bin/env python3
"""Generate a directed 8-bin test that FAILS if DW-ReLU is reintroduced."""
from pathlib import Path
import argparse
import numpy as np
from neuroedge.speech.fixed_decoder import (
    TOTAL_WEIGHTS,TOTAL_META,WEIGHT_OFFSETS,
    FixedParameters,FixedStreamingDecoder,write_rtl_vectors,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    w=np.zeros(TOTAL_WEIGHTS,dtype=np.int8)
    b=np.zeros(TOTAL_META,dtype=np.int32)
    sh=np.zeros(TOTAL_META,dtype=np.uint8)
    # Drive +32 into project, then TWO negative DW results followed by
    # negative pointwise coefficients. Each negative pair restores +32.
    w[WEIGHT_OFFSETS['project']]=1
    w[WEIGHT_OFFSETS['dw1']+4]=-1
    w[WEIGHT_OFFSETS['pw1']]=-1
    w[WEIGHT_OFFSETS['dw2']+4]=-1
    w[WEIGHT_OFFSETS['pw2']]=-1
    w[WEIGHT_OFFSETS['head']]=1
    params=FixedParameters(w,b,sh)
    x=np.zeros((8,512),dtype=np.int8)
    x[:,0]=32
    expected=FixedStreamingDecoder(params).run(x)
    assert expected.shape==(2,41)
    assert expected[0,0]==32 and expected[1,0]==32
    assert not np.any(expected[:,1:])
    write_rtl_vectors(args.out,params,x,expected)
    print('NEGATIVE_DEPTHWISE_DIRECTED_VECTORS_PASS output=32 outputs=82')

if __name__=='__main__':main()
