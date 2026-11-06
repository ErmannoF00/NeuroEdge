#!/usr/bin/env python3
"""Analytical neural analog front-end noise/power/bitrate BUDGET, not SPICE signoff.

Noise model assumes flat input-referred amplifier white noise and ideal ADC.
Excludes flicker noise, electrode impedance, lead motion, CM interference,
anti-alias design, ADC ENOB, nonlinearities, real power and safety compliance.
Used only to establish interface targets for a future transistor-level/PCB design.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path


def budget(*,channels:int=256,fs:int=20000,adc_bits:int=12,
           analog_bw:float=7500,noise_nv_sqrt_hz:float=25.0,
           gain:float=1000,adc_vpp:float=1.0,channel_power_uw:float=10.0,
           feature_dims:int=512,feature_period_ms:float=20.0)->dict:
    if not (channels>0 and fs>0 and adc_bits>0 and analog_bw>0 and analog_bw<fs/2
            and noise_nv_sqrt_hz>0 and gain>0 and adc_vpp>0 and channel_power_uw>0
            and feature_dims>0 and feature_period_ms>0):
        raise ValueError('invalid analogue system budget')
    amp_noise_uv=noise_nv_sqrt_hz*1e-3*math.sqrt(analog_bw)
    # Ideal uniform quantization noise: full-scale Vpp / (2^N * sqrt(12)).
    adc_ir_noise_uv=(adc_vpp/(2**adc_bits*math.sqrt(12)*gain))*1e6
    combined=math.hypot(amp_noise_uv,adc_ir_noise_uv)
    raw_bps=channels*fs*adc_bits
    feature_bps=feature_dims*8/(feature_period_ms/1000)
    return {
        'channels':channels,'sampling_hz':fs,'analog_bandwidth_hz':analog_bw,
        'input_amp_white_noise_uv_rms':amp_noise_uv,
        'adc_ideal_input_referred_noise_uv_rms':adc_ir_noise_uv,
        'combined_idealized_noise_uv_rms':combined,
        'analog_channel_power_mw':channels*channel_power_uw/1000,
        'raw_adc_payload_bps':raw_bps,
        'feature_stream_payload_bps':feature_bps,
        'nominal_raw_to_feature_bit_rate_ratio':raw_bps/feature_bps,
        'limitations':'ANALYTICAL ONLY; does not model electrode, circuit, clock, digital, wireless or implant safety',
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--channels',type=int,default=256)
    p.add_argument('--fs',type=int,default=20000)
    p.add_argument('--bw',type=float,default=7500.0)
    p.add_argument('--noise-nv',type=float,default=25.0)
    p.add_argument('--gain',type=float,default=1000.0)
    p.add_argument('--bits',type=int,default=12)
    p.add_argument('--channel-uw',type=float,default=10)
    p.add_argument('--out',type=Path,default=Path('results/afe/budget.json'))
    a=p.parse_args()
    b=budget(channels=a.channels,fs=a.fs,adc_bits=a.bits,
             analog_bw=a.bw,noise_nv_sqrt_hz=a.noise_nv,
             gain=a.gain,channel_power_uw=a.channel_uw)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(b,indent=2)+'\n')
    print(json.dumps(b,indent=2))
    print('AFE_ANALYTICAL_BUDGET_PASS (not circuit validation)')
if __name__=='__main__':main()
