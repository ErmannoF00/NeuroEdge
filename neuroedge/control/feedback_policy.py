"""Strictly lagged authorization for externally supplied calibration proposals.

Only an interface and safety policy. Proposal generation and clinical efficacy
are outside this module. Every accepted proposal is a change to stored INT16
input offsets, never an update to decoder weights or an in-memory MAC.
"""
from __future__ import annotations
from dataclasses import dataclass
from neuroedge.speech.offset_reference import OffsetTable


@dataclass(frozen=True)
class FeedbackProposal:
    source_trial: int
    available_after_trial: int
    channel: int
    delta_q4: int
    provenance: str


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    new_offset_q4: int | None = None


class LaggedFeedbackGate:
    """Bounds and timestamps prospective single-channel memory updates.

    `next_trial` is the trial about to start, not the one being evaluated.
    `safe_boundary` is supplied by the caller's decoder/stream ownership logic.
    """
    def __init__(self, *, max_step_q4: int = 16, max_abs_offset_q4: int = 128,
                 min_trials_between_writes: int = 2):
        if max_step_q4 < 1 or max_abs_offset_q4 < 1 or min_trials_between_writes < 0:
            raise ValueError('invalid gate bounds')
        if max_abs_offset_q4 > 32767:
            raise ValueError('INT16 bound exceeded')
        self.max_step_q4 = max_step_q4
        self.max_abs_offset_q4 = max_abs_offset_q4
        self.min_trials_between_writes = min_trials_between_writes
        self.last_write_trial: dict[int, int] = {}
        self.used: set[tuple[int, int, str]] = set()

    def try_apply(self, proposal: FeedbackProposal, *, next_trial: int,
                  table: OffsetTable, safe_boundary: bool) -> Decision:
        if not proposal.provenance.strip():
            return Decision(False, 'unattributed feedback')
        if (proposal.source_trial < 0 or proposal.available_after_trial < 0 or
                next_trial < 0 or proposal.available_after_trial < proposal.source_trial or
                proposal.available_after_trial >= next_trial):
            return Decision(False, 'feedback not available before target trial')
        if not (0 <= proposal.channel < 512):
            return Decision(False, 'invalid feature index')
        if proposal.delta_q4 == 0 or abs(proposal.delta_q4) > self.max_step_q4:
            return Decision(False, 'update exceeds bounded step')
        key = (proposal.source_trial, proposal.channel, proposal.provenance)
        if key in self.used:
            return Decision(False, 'feedback already consumed')
        last = self.last_write_trial.get(proposal.channel)
        if last is not None and next_trial - last < self.min_trials_between_writes:
            return Decision(False, 'channel in cooldown')
        if not safe_boundary or table.bin_in_flight:
            return Decision(False, 'decoder not at a safe bin boundary')
        value = table.read_channel(proposal.channel) + proposal.delta_q4
        if abs(value) > self.max_abs_offset_q4:
            return Decision(False, 'offset bound would be exceeded')
        if not table.stage(proposal.channel, value):
            return Decision(False, 'staging slot unavailable')
        if not table.authorize():
            # No deferred retry. Abort a staged proposal on failed authority.
            table.abort()
            return Decision(False, 'memory authorization rejected')
        self.used.add(key)
        self.last_write_trial[proposal.channel] = next_trial
        return Decision(True, 'committed at safe boundary', value)
