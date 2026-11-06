"""Completed-trial supervised replay around the VERIFED NeuroEdge arithmetic.

Contract:
 - input is 512 INT8 feature values at monotonic, explicit 20 ms bin boundaries;
 - CTC scores emitted at input bins 0,4,8,... (ceil(T/4) steps);
 - a trial starts with zero causal history, as the current dataset evaluator does;
 - offsets persist across trials, but cannot commit inside an active trial/bin;
 - labels become available AFTER a trial ends; edits affect only later trials;
 - output release is opt-in; neural output is never treated as arbitrary thought;
 - software virtual time is NOT measured wall-time or demonstrated real-time RTL.

The selected calibration policy is a bounded *retrospective* CTC edit-distance
candidate search. Its choices may OVERFIT past trials; no efficacy is assumed.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from typing import Literal
import time
import numpy as np
from neuroedge.speech.fixed_decoder import FixedParameters, FixedStreamingDecoder, INPUTS, CLASSES
from neuroedge.speech.ctc import ctc_collapse, edit_distance, TOKENS
from neuroedge.speech.offset_reference import OffsetStreamingDecoder, apply_offsets, OffsetTable
from neuroedge.control.feedback_policy import FeedbackProposal, LaggedFeedbackGate
from neuroedge.memory.banked_projection import BankedStreamingDecoder

BIN_PERIOD_US = 20_000


@dataclass(frozen=True)
class FeaturePacket:
    trial_id: int
    bin_index: int
    timestamp_us: int
    feature_q4: np.ndarray

    def validate(self) -> None:
        if self.trial_id < 0 or self.bin_index < 0 or self.timestamp_us < 0:
            raise ValueError('negative trial/bin/timestamp')
        if not isinstance(self.feature_q4, np.ndarray) or self.feature_q4.dtype != np.int8 or self.feature_q4.shape != (INPUTS,):
            raise ValueError('expected a contiguous 512-feature INT8 bin')


@dataclass(frozen=True)
class TrialSummary:
    trial_id: int
    bins: int
    emitted_steps: int
    first_timestamp_us: int
    last_timestamp_us: int
    wall_decode_ms: float
    virtual_duration_ms: float
    target_phonemes: int | None
    ctc_edits: int | None
    feedback_action: str
    offset_writes: int
    released: bool
    released_phonemes: str | None
    # An absolute rate cannot be interpreted as measured on-device latency.


@dataclass(frozen=True)
class QueuedCorrection:
    available_at_trial: int
    source_trial: int
    base_version: int
    proposed_delta: int
    channel: int
    improvement_edits: int


def predicted_phonemes(logits: np.ndarray) -> list[int]:
    if logits.dtype != np.int16 or logits.ndim != 2 or logits.shape[1] != CLASSES:
        raise ValueError('expected [steps,41] signed INT16 logits')
    return ctc_collapse(logits.argmax(axis=1).tolist())


def proposal_from_completed_trial(*, initial_state: FixedStreamingDecoder,
                                  raw_bins: np.ndarray, known_target: list[int],
                                  offsets: np.ndarray, channel: int,
                                  step: int = 4) -> tuple[int, int]:
    """Choose ±step only for a strictly smaller PAST-trial CTC edit distance.

    Caller must reveal known_target only AFTER the trial and must not use a
    target that was derived from the *same* prediction being improved.
    """
    if raw_bins.dtype != np.int8 or raw_bins.ndim != 2 or raw_bins.shape[1] != INPUTS:
        raise ValueError('expected completed INT8 trial')
    if offsets.dtype != np.int16 or offsets.shape != (INPUTS,):
        raise ValueError('expected offset snapshot')
    if channel < 0 or channel >= INPUTS or step < 1 or step > 16:
        raise ValueError('bad candidate configuration')
    if any(not isinstance(v, (int, np.integer)) or v < 0 or v >= CLASSES for v in known_target):
        raise ValueError('invalid CTC phoneme labels')
    baseline = deepcopy(initial_state).run(apply_offsets(raw_bins, offsets))
    old = edit_distance(predicted_phonemes(baseline), known_target)
    selected, best = 0, old
    for delta in (-step, step):
        candidate_offsets = offsets.copy()
        new_val = int(candidate_offsets[channel]) + delta
        if not -32768 <= new_val <= 32767:
            continue
        candidate_offsets[channel] = np.int16(new_val)
        candidate = deepcopy(initial_state).run(apply_offsets(raw_bins, candidate_offsets))
        score = edit_distance(predicted_phonemes(candidate), known_target)
        if score < best:
            selected, best = delta, score
    return selected, old - best


class ConnectedStream:
    """Auditable streaming controller with trial-boundary authority.

    Process one trial at a time; `push` checks sequential bins/timestamps, then
    `finish_trial` closes it. Subsequent corrections are considered at next
    `begin_trial`, never during a bin. Only post-trial user/task supervision may
    produce a proposal. The interface itself does not infer communication intent.
    """
    def __init__(self, params: FixedParameters, *, banks: int | None = None,
                 feedback_mode: Literal['off', 'supervised', 'sham'] = 'off',
                 feedback_channel: int = 0, step_q4: int = 4,
                 delay_trials: int = 0, max_offset_q4: int = 32,
                 min_trials_between_writes: int = 1):
        if feedback_mode not in ('off', 'supervised', 'sham') or not 0 <= feedback_channel < INPUTS:
            raise ValueError('bad feedback mode or channel')
        if not 0 <= delay_trials <= 10000 or not 1 <= step_q4 <= 16:
            raise ValueError('bad update settings')
        self.params = params
        self.banks = banks
        self.mode = feedback_mode
        self.channel = feedback_channel
        self.step = step_q4
        self.delay = delay_trials
        self.table = OffsetTable()
        self.gate = LaggedFeedbackGate(max_step_q4=step_q4,
                                       max_abs_offset_q4=max_offset_q4,
                                       min_trials_between_writes=min_trials_between_writes)
        self.version = 0
        self.queue: list[QueuedCorrection] = []
        self.trial_id = -1
        self._active = False
        self._first_us = -1
        self._last_us = -1
        self._bins: list[np.ndarray] = []
        self._logits: list[np.ndarray] = []
        self._first_decoder_state: FixedStreamingDecoder | None = None
        self._offsets_at_start: np.ndarray | None = None
        self._decode_wall = 0.0
        self._released = False
        self._boundary_action = 'no request'
        self._decoder: FixedStreamingDecoder | None = None
        self.completed: list[TrialSummary] = []

    def _make_decoder(self):
        return (FixedStreamingDecoder(self.params) if self.banks is None
                else BankedStreamingDecoder(self.params, banks=self.banks))

    def _offset_snapshot(self) -> np.ndarray:
        return np.where(self.table.valid, self.table.active, 0).astype(np.int16)

    def begin_trial(self, *, trial_id: int, first_timestamp_us: int,
                    allow_output: bool = False) -> str:
        if self._active or trial_id != self.trial_id + 1 or first_timestamp_us < 0:
            raise ValueError('trial sequence violation or unfinished trial')
        if self._last_us >= 0 and first_timestamp_us <= self._last_us:
            raise ValueError('trial time moved backwards')
        if self.table.bin_in_flight:
            raise RuntimeError('unsafe boundary')
        self.trial_id = trial_id
        action = 'no request'
        # Drop expired feedback: proposal expected against exact offset version.
        eligible = [q for q in self.queue if q.available_at_trial <= trial_id]
        self.queue = [q for q in self.queue if q.available_at_trial > trial_id]
        for correction in eligible:
            if correction.base_version != self.version:
                action = 'stale feedback denied'
                continue
            p = FeedbackProposal(source_trial=correction.source_trial,
                                 available_after_trial=correction.source_trial,
                                 channel=correction.channel,
                                 delta_q4=correction.proposed_delta,
                                 provenance=f'post-trial-{correction.source_trial}-{self.mode}')
            decision = self.gate.try_apply(p, next_trial=trial_id,
                                           table=self.table, safe_boundary=True)
            action = decision.reason
            if decision.accepted:
                self.version += 1
        self._boundary_action = action
        self._released = bool(allow_output)
        self._first_us = first_timestamp_us
        self._last_us = -1 if trial_id == 0 else self._last_us
        self._bins = []
        self._logits = []
        self._decode_wall = 0.0
        self._decoder = self._make_decoder()
        self._first_decoder_state = deepcopy(self._decoder)
        self._offsets_at_start = self._offset_snapshot()
        self._active = True
        return action

    def push(self, packet: FeaturePacket) -> np.ndarray | None:
        packet.validate()
        if not self._active or packet.trial_id != self.trial_id or packet.bin_index != len(self._bins):
            raise ValueError('unexpected feature packet')
        expected_us = self._first_us + BIN_PERIOD_US * packet.bin_index
        if packet.timestamp_us != expected_us or (self._last_us >= 0 and packet.timestamp_us <= self._last_us):
            raise ValueError('bad timestamp/20ms cadence')
        if self._decoder is None:
            raise AssertionError('decoder missing')
        self.table.begin_bin()
        start = time.perf_counter()
        try:
            # Offset vector is immutable for the complete 512-feature bin.
            calibrated = self.table.calibrate(packet.feature_q4)
            value = self._decoder.push(calibrated)
        finally:
            self._decode_wall += time.perf_counter() - start
            self.table.end_bin()
        self._bins.append(packet.feature_q4.copy())
        self._last_us = packet.timestamp_us
        if value is not None:
            self._logits.append(value.copy())
        # The privileged internal decoder retains logits for scoring. Unarmed
        # callers receive no logits or phoneme payload from this interface.
        return value if self._released else None

    def abort_trial(self) -> None:
        """Drop a damaged/incomplete trial; retain committed calibration state.

        No partial logits or label-driven correction is released or enqueued.
        The next trial must have the next trial_id and a later timestamp.
        """
        if not self._active:
            raise RuntimeError('cannot abort a non-active trial')
        self._active = False
        self._released = False
        self._bins = []
        self._logits = []
        self._first_decoder_state = None
        self._offsets_at_start = None
        self._decoder = None
        if self.table.bin_in_flight:
            raise AssertionError('attempted to abort while a feature bin is active')

    def attempt_unsafe_commit(self, proposal: FeedbackProposal) -> bool:
        """Test-only explicit unsafe request. Full trial ownership is the gate."""
        if not self._active:
            raise RuntimeError('no active trial')
        decision = self.gate.try_apply(proposal, next_trial=self.trial_id,
                                       table=self.table, safe_boundary=False)
        return decision.accepted

    def finish_trial(self, *, revealed_target: list[int] | None = None) -> TrialSummary:
        if not self._active or not self._bins:
            raise RuntimeError('empty/inactive trial')
        logits = np.stack(self._logits).astype(np.int16)
        expected_steps = (len(self._bins) + 3) // 4
        if logits.shape != (expected_steps, CLASSES):
            raise AssertionError('CTC cadence mismatch')
        predicted = predicted_phonemes(logits)
        if revealed_target is not None:
            known = [int(v) for v in revealed_target]
            if any(v <= 0 or v >= CLASSES for v in known):
                raise ValueError('target phonemes must be 1..40')
            edits = edit_distance(predicted, known)
        else:
            known = None
            edits = None
        if self.mode == 'supervised' and known is not None:
            delta, improvement = proposal_from_completed_trial(
                initial_state=self._first_decoder_state,
                raw_bins=np.stack(self._bins), known_target=known,
                offsets=self._offsets_at_start, channel=self.channel,
                step=self.step)
            if delta:
                self.queue.append(QueuedCorrection(
                    available_at_trial=self.trial_id + 1 + self.delay,
                    source_trial=self.trial_id,
                    base_version=self.version, proposed_delta=delta,
                    channel=self.channel, improvement_edits=improvement))
        elif self.mode == 'sham':
            # Predeclared signed sequence, does NOT access neural labels.
            self.queue.append(QueuedCorrection(
                available_at_trial=self.trial_id + 1 + self.delay,
                source_trial=self.trial_id, base_version=self.version,
                proposed_delta=(self.step if self.trial_id % 2 == 0 else -self.step),
                channel=self.channel, improvement_edits=0))
        released = ' '.join(TOKENS[i] for i in predicted) if self._released else None
        summary = TrialSummary(
            trial_id=self.trial_id, bins=len(self._bins),
            emitted_steps=int(logits.shape[0]), first_timestamp_us=self._first_us,
            last_timestamp_us=self._last_us, wall_decode_ms=round(self._decode_wall*1000, 3),
            virtual_duration_ms=len(self._bins) * BIN_PERIOD_US / 1000,
            target_phonemes=len(known) if known is not None else None,
            ctc_edits=edits, feedback_action=self._boundary_action,
            offset_writes=self.table.accepted_writes,
            released=self._released, released_phonemes=released)
        self.completed.append(summary)
        self._active = False
        self._bins = []
        self._logits = []
        self._first_decoder_state = None
        self._offsets_at_start = None
        return summary


def run_trial_series(params: FixedParameters,
                     trials: list[tuple[np.ndarray, list[int] | None]], *,
                     banks: int | None = None, mode: str = 'off',
                     armed_trials: frozenset[int] = frozenset(),
                     feedback_channel: int = 0, step_q4: int = 4,
                     delay_trials: int = 0) -> dict:
    """Timestamped replay; source trial labels only revealed at end of trial."""
    system = ConnectedStream(params, banks=banks, feedback_mode=mode,
                             feedback_channel=feedback_channel, step_q4=step_q4,
                             delay_trials=delay_trials)
    timestamp = 0
    for t, (bins, label) in enumerate(trials):
        b = np.asarray(bins)
        if b.dtype != np.int8 or b.ndim != 2 or b.shape[1] != INPUTS or len(b) < 1:
            raise ValueError('trials must hold nonempty INT8[T,512]')
        system.begin_trial(trial_id=t, first_timestamp_us=timestamp,
                           allow_output=t in armed_trials)
        for i, x in enumerate(b):
            system.push(FeaturePacket(t, i, timestamp + BIN_PERIOD_US*i, x))
        system.finish_trial(revealed_target=label)
        timestamp += len(b)*BIN_PERIOD_US
    total = sum(x.ctc_edits for x in system.completed if x.ctc_edits is not None)
    count = sum(x.target_phonemes for x in system.completed if x.target_phonemes is not None)
    return {
        'classification_task': 'attempted-speech CTC phonemes, NOT arbitrary thoughts',
        'hardware_backend': 'software numeric reference with banked projection' if banks else 'software numeric reference',
        'physical_measurements': False, 'feedback_mode': mode,
        'supervision': ('offline labels revealed after each completed trial' if mode=='supervised'
                        else 'no label-driven parameter updates'),
        'bin_period_us': BIN_PERIOD_US, 'trials': len(system.completed),
        'bins': sum(x.bins for x in system.completed),
        'outputs': sum(x.emitted_steps for x in system.completed),
        'ctc_edits': total if count else None,
        'target_phonemes': count if count else None,
        'greedy_per': total / count if count else None,
        'committed_writes': system.table.accepted_writes,
        'denied_writes': system.table.rejected_requests,
        'offset_version': system.version,
        'release_control': 'explicit external per-trial opt-in, NOT inferred neural intent',
        'records': [asdict(s) for s in system.completed],
    }
