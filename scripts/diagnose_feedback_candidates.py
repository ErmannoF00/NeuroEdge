#!/usr/bin/env python3
"""Read-only, retrospective trial calibration diagnostics.

No new offset is committed, no learned model is changed, and no private neural
arrays/phoneme strings/paths are written to the report. Candidate improvements
on the *same trial* are NOT evidence of prospective adaptation efficacy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder, FixedParameters
from neuroedge.speech.offset_reference import apply_offsets
from neuroedge.speech.ctc import edit_distance
from neuroedge.system.streaming import predicted_phonemes
from neuroedge.speech.synthetic_fixture import synthetic_features, synthetic_parameters


def diagnose_trial(params: FixedParameters, raw: np.ndarray, target: list[int],
                   *, channels: tuple[int, ...], steps: tuple[int, ...]) -> dict:
    if raw.dtype != np.int8 or raw.ndim != 2 or raw.shape[1] != 512 or not len(raw):
        raise ValueError('trial must be INT8[T,512]')
    if not channels or not steps or any(not 0 <= c < 512 for c in channels):
        raise ValueError('bad feature channels')
    if any(not 1 <= s <= 16 for s in steps):
        raise ValueError('steps must be 1..16 Q4 units')
    if not target or any(not 0 < int(x) < 41 for x in target):
        raise ValueError('invalid completed-trial supervision')
    offsets = np.zeros(512, dtype=np.int16)
    calibrated = apply_offsets(raw, offsets)
    baseline = FixedStreamingDecoder(params).run(calibrated)
    baseline_prediction = predicted_phonemes(baseline)
    baseline_edits = edit_distance(baseline_prediction,target)
    candidates=[]
    for ch in channels:
        for step in steps:
            for delta in (-step,step):
                candidate_offsets = offsets.copy()
                candidate_offsets[ch]=np.int16(delta)
                adjusted=apply_offsets(raw,candidate_offsets)
                feature_changes=int(np.count_nonzero(adjusted!=calibrated))
                logits=FixedStreamingDecoder(params).run(adjusted)
                logit_changes=int(np.count_nonzero(logits!=baseline))
                new_prediction=predicted_phonemes(logits)
                edits=edit_distance(new_prediction,target)
                if feature_changes==0:
                    reason='no_input_effect_after_saturation'
                elif logit_changes==0:
                    reason='no_logit_effect'
                elif new_prediction==baseline_prediction:
                    reason='logits_changed_ctc_unchanged'
                elif edits>=baseline_edits:
                    reason='ctc_changed_no_edit_improvement'
                else:
                    reason='retrospective_edit_improvement_only'
                candidates.append({'channel':ch,'step_q4':delta,
                    'feature_values_changed':feature_changes,
                    'scalar_logits_changed':logit_changes,
                    'predicted_phoneme_sequence_changed':new_prediction!=baseline_prediction,
                    'baseline_edits':baseline_edits,'candidate_edits':edits,
                    'retrospective_edit_delta':baseline_edits-edits,
                    'reason':reason})
    return {'bins':len(raw),'ctc_steps':int(baseline.shape[0]),
            'target_phonemes':len(target),'baseline_edits':baseline_edits,
            'candidates':candidates,
            'strictly_improving_candidates':sum(c['candidate_edits']<baseline_edits for c in candidates)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sources=ap.add_mutually_exclusive_group(required=True)
    sources.add_argument('--synthetic',action='store_true')
    sources.add_argument('--hdf5',type=Path)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--quant-profile',type=Path)
    ap.add_argument('--max-trials',type=int,default=2)
    ap.add_argument('--channels',default='0,304')
    ap.add_argument('--steps',default='4')
    ap.add_argument('--out',type=Path,default=Path('build/feedback_diagnostic.json'))
    args=ap.parse_args()
    channels=tuple(int(x.strip()) for x in args.channels.split(',') if x.strip())
    steps=tuple(int(x.strip()) for x in args.steps.split(',') if x.strip())
    if len(channels)>8 or len(steps)>3 or not 1<=args.max_trials<=8:
        ap.error('diagnostic bounded to 8 channels, 3 step sizes, 8 trials')
    if args.synthetic:
        if args.checkpoint or args.quant_profile:
            ap.error('checkpoint only valid with --hdf5')
        from neuroedge.speech.ctc import ctc_collapse
        p=synthetic_parameters()
        x=synthetic_features(4)
        target=ctc_collapse(FixedStreamingDecoder(p).run(x).argmax(axis=1).tolist())
        if not target:
            target=[1]
        trials=[(x,target)]
        provenance='synthetic_nonlearned'
    else:
        if not args.checkpoint or not args.quant_profile:
            ap.error('real input requires frozen checkpoint and quantization profile')
        from scripts.run_connected_stream import load_real_trials
        p,trials=load_real_trials(hdf5=args.hdf5,checkpoint=args.checkpoint,
                                  profile_path=args.quant_profile,
                                  weights_dir=None,max_trials=args.max_trials)
        provenance='locally_held_attempted_speech_validation'
    result={'task':'retrospective_diagnostic_only',
            'provenance':provenance,
            'adaptation_improvement_proven':False,
            'active_offsets_q4':'zero_for_each_trial_no_commits',
            'candidate_supervision':'completed trial labels used only after baseline inference',
            'channels':list(channels),'steps':list(steps),'trials':[]}
    for i,(features,labels) in enumerate(trials):
        item=diagnose_trial(p,np.asarray(features),list(labels),
                            channels=channels,steps=steps)
        result['trials'].append({'local_trial_ordinal':i,**item})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('NEUROEDGE_FEEDBACK_DIAGNOSTIC_PASS')
    for t in result['trials']:
        reasons={}
        for c in t['candidates']:
            reasons[c['reason']]=reasons.get(c['reason'],0)+1
        print(f"  trial={t['local_trial_ordinal']} edits={t['baseline_edits']} "
              f"strictly_improving={t['strictly_improving_candidates']} causes={reasons}")
    print('  local_report=',args.out)

if __name__=='__main__':
    main()
