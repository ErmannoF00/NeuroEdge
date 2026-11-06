"""Causal *synthetic* supervised feedback around the existing integer decoder.

A known calibration target is revealed only after its trial completes. An
offline estimator replays that completed trial against bounded offset
candidates using a snapshot of the decoder's *past* causal state, then
requests a real transactional OffsetTable commit before the NEXT trial.

This establishes temporal control semantics, NOT an autonomous thought decoder,
patient feedback system or real-data improvement. Human or live targets are
neither inferred nor present in this simulation.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

from neuroedge.control.feedback_policy import FeedbackProposal, LaggedFeedbackGate
from neuroedge.speech.fixed_decoder import FixedStreamingDecoder, FixedParameters
from neuroedge.speech.offset_reference import OffsetStreamingDecoder
from neuroedge.speech.synthetic_fixture import synthetic_features, synthetic_parameters


@dataclass(frozen=True)
class TrialRecord:
    trial: int
    baseline_logit_l1: int
    controlled_logit_l1: int
    sham_logit_l1: int
    target_available_after_trial: int
    active_offset_q4: int
    proposal_delta_q4: int
    proposed_improvement_l1: int
    accepted: bool
    decision: str


def _decode_past_trial(decoder_at_start: FixedStreamingDecoder,
                       trial_input: np.ndarray, channel: int, offset_q4: int) -> np.ndarray:
    # Copy only a state captured BEFORE this trial, never a decoder state from
    # the next trial. Candidate evaluation does not modify the live decoder.
    decoder = deepcopy(decoder_at_start)
    features = np.array(trial_input, copy=True)
    x = features[:, channel].astype(np.int32) + int(offset_q4)
    features[:, channel] = np.clip(x, -128, 127).astype(np.int8)
    return decoder.run(features)


def _l1(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def select_lagged_proposal(*, decoder_before_trial: FixedStreamingDecoder,
                           completed_raw_trial: np.ndarray, completed_logits: np.ndarray,
                           revealed_target_logits: np.ndarray, current_offset_q4: int,
                           source_trial: int, channel: int = 0,
                           max_step_q4: int = 4) -> tuple[FeedbackProposal | None, int]:
    """Supervised candidate selection with *previous-trial-only* targets.

    The caller is responsible for revealing the target after trial completion.
    This procedure cannot infer desired speech from neural data on its own.
    """
    if not 1 <= max_step_q4 <= 16:
        raise ValueError('unsupported bounded step')
    if completed_raw_trial.dtype != np.int8 or completed_raw_trial.shape != (4, 512):
        raise ValueError('expected one completed 4-bin trial')
    if completed_logits.dtype != np.int16 or completed_logits.shape != (1, 41):
        raise ValueError('expected one completed trial output')
    if revealed_target_logits.dtype != np.int16 or revealed_target_logits.shape != (1, 41):
        raise ValueError('expected supervised 41-logit target')
    base_loss = _l1(completed_logits, revealed_target_logits)
    best_loss = base_loss
    best_delta = 0
    for step in (-max_step_q4, max_step_q4):
        candidate = _decode_past_trial(decoder_before_trial, completed_raw_trial,
                                       channel, current_offset_q4 + step)
        loss = _l1(candidate, revealed_target_logits)
        if loss < best_loss:
            best_loss, best_delta = loss, step
    if not best_delta:
        return None, 0
    proposal = FeedbackProposal(source_trial=source_trial,
                                available_after_trial=source_trial,
                                channel=channel, delta_q4=best_delta,
                                provenance=f'synthetic-supervised-trial-{source_trial}')
    return proposal, base_loss - best_loss


def run_synthetic_replay(*, drift_q4: int = 9, bins: int = 32,
                         feedback_delay_trials: int = 0,
                         parameters: FixedParameters | None = None) -> dict:
    """Compare fixed offsets, bounded lagged control, and clean teacher.

    `feedback_delay_trials` enforces additional whole-trial delay AFTER a
    completed trial. No target from a future trial is used in a proposal.
    """
    if bins < 16 or bins % 4 or not -32 <= drift_q4 <= 32:
        raise ValueError('unsupported demonstration shape or drift')
    if feedback_delay_trials < 0:
        raise ValueError('feedback delay must be nonnegative')
    p = parameters if parameters is not None else synthetic_parameters()
    clean = synthetic_features(bins)
    raw = clean.copy()
    raw[:, 0] = np.clip(clean[:, 0].astype(np.int32) + drift_q4, -128, 127).astype(np.int8)
    ideal = FixedStreamingDecoder(p).run(clean)
    fixed = FixedStreamingDecoder(p).run(raw)
    controlled = OffsetStreamingDecoder(p)
    sham = OffsetStreamingDecoder(p)
    sham_gate = LaggedFeedbackGate(max_step_q4=4, max_abs_offset_q4=32,
                                   min_trials_between_writes=1)
    sham_observed: list[np.ndarray] = []
    gate = LaggedFeedbackGate(max_step_q4=4, max_abs_offset_q4=32,
                              min_trials_between_writes=1)
    records: list[TrialRecord] = []
    observed: list[np.ndarray] = []
    # Only completed trial data/teacher labels can enter this FIFO.
    queued: list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, FixedStreamingDecoder]] = []
    trials = bins // 4
    for t in range(trials):
        start = t * 4
        # Apply eligible proposals at a boundary only, before the next bin.
        if controlled.table.bin_in_flight:
            raise AssertionError('unexpected active bin at trial boundary')
        decision = 'no available feedback'
        accepted = False
        proposal_delta = 0
        proposed_improvement = 0
        eligible = [q for q in queued if q[0] < t]
        queued = [q for q in queued if q[0] >= t]
        if eligible:
            # FIFO of delayed observations: one update request per boundary.
            _, source, prior_offset, prior_raw, prior_output, prior_target, prior_state = eligible.pop(0)
            queued = eligible + queued
            if controlled.table.read_channel(0) != prior_offset:
                # No stale optimizer response may be applied against a newer
                # coefficient version, including when feedback is delayed.
                decision = 'stale feedback: coefficient version changed'
            else:
                proposal, proposed_improvement = select_lagged_proposal(
                    decoder_before_trial=prior_state, completed_raw_trial=prior_raw,
                    completed_logits=prior_output, revealed_target_logits=prior_target,
                    current_offset_q4=prior_offset, source_trial=source)
                if proposal:
                    response = gate.try_apply(proposal, next_trial=t,
                                              table=controlled.table, safe_boundary=True)
                    decision, accepted = response.reason, response.accepted
                    proposal_delta = proposal.delta_q4
                else:
                    decision = 'past-trial candidate not better'
        # Deterministic control without any target inspection. It uses the
        # SAME safe transactional mechanism, but its updates carry no signal.
        if t > 0:
            sham_proposal = FeedbackProposal(
                source_trial=t-1, available_after_trial=t-1, channel=0,
                delta_q4=4 if t % 2 else -4,
                provenance=f'predeclared-sham-step-{t}')
            if not sham_gate.try_apply(sham_proposal, next_trial=t,
                                       table=sham.table, safe_boundary=True).accepted:
                raise AssertionError('deterministic sham write unexpectedly rejected')
        sham_outputs = []
        for b in range(start, start + 4):
            value = sham.push(raw[b])
            if value is not None:
                sham_outputs.append(value)
        sham_produced = np.stack(sham_outputs).astype(np.int16)
        sham_observed.append(sham_produced[0])
        active = controlled.table.read_channel(0)
        prior_state = deepcopy(controlled.decoder)
        outputs = []
        for b in range(start, start + 4):
            y = controlled.push(raw[b])
            if y is not None:
                outputs.append(y)
        produced = np.stack(outputs).astype(np.int16)
        observed.append(produced[0])
        # The teacher is disclosed only at trial completion, never before.
        # A production system would need an independently supplied, audited label.
        queued.append((t + feedback_delay_trials, t, active, raw[start:start + 4].copy(),
                       produced.copy(), ideal[t:t+1].copy(), prior_state))
        records.append(TrialRecord(
            trial=t, baseline_logit_l1=_l1(fixed[t:t+1], ideal[t:t+1]),
            controlled_logit_l1=_l1(produced, ideal[t:t+1]),
            sham_logit_l1=_l1(sham_produced, ideal[t:t+1]),
            target_available_after_trial=t + feedback_delay_trials,
            active_offset_q4=active, proposal_delta_q4=proposal_delta,
            proposed_improvement_l1=proposed_improvement,
            accepted=accepted, decision=decision))
    controlled_logits = np.stack(observed).astype(np.int16)
    sham_logits = np.stack(sham_observed).astype(np.int16)
    evidence = {
        'provenance': 'completely synthetic, non-learned model and deterministic injected drift',
        'feedback': 'retrospective supervised clean-target logits; supplied only after trial completion',
        'outputs': int(ideal.size), 'bins': bins, 'trials': trials,
        'drift_feature_index': 0, 'drift_q4': drift_q4,
        'feedback_delay_trials': feedback_delay_trials,
        'fixed_logit_l1': _l1(fixed, ideal),
        'controlled_logit_l1': _l1(controlled_logits, ideal),
        'sham_logit_l1': _l1(sham_logits, ideal),
        'sham_accepted_writes': sham.table.accepted_writes,
        'accepted_memory_writes': controlled.table.accepted_writes,
        'rejected_memory_requests': controlled.table.rejected_requests,
        'final_offset_q4': controlled.table.read_channel(0),
        'trials_log': [asdict(x) for x in records],
        'no_supervised_target_for_sham': True,
        'no_real_data_accuracy_claim': True,
        'no_hardware_closed_loop_claim': True,
    }
    return evidence


def write_report(path: Path, **kwargs) -> dict:
    report = run_synthetic_replay(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    return report
