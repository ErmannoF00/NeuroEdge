"""C-compatible feature packets -> trial-owned integer decoder interface.

A packet has no release/authorization flag. `externally_armed` can only be set
at the beginning of a trial by trusted local control software. The CRC checks
corruption, not origin or authenticity. This is not a device driver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from neuroedge.system.packet_wire import (
    decode, to_feature_packet, FLAG_BEGIN_TRIAL, FLAG_END_TRIAL,
)
from neuroedge.system.streaming import ConnectedStream, TrialSummary


@dataclass(frozen=True)
class Received:
    released_logits: np.ndarray | None
    completed_trial: TrialSummary | None


class WireStreamReceiver:
    def __init__(self, stream: ConnectedStream):
        self.stream = stream
        self.active = False
        self.packet_count = 0
        self.aborted_trials = 0

    def receive(self, wire: bytes, *, externally_armed: bool = False,
                revealed_target: list[int] | None = None) -> Received:
        p = decode(wire)  # CRC/header rejection BEFORE touching stream state
        begin = bool(p.flags & FLAG_BEGIN_TRIAL)
        end = bool(p.flags & FLAG_END_TRIAL)
        if not self.active:
            if not begin or p.bin_index != 0:
                raise ValueError('new trial must start at bin zero with BEGIN_TRIAL')
        elif begin:
            raise ValueError('duplicate BEGIN_TRIAL without an END/abort')
        if self.active and externally_armed:
            raise ValueError('cannot arm output from the feature data plane mid-trial')
        if revealed_target is not None and not end:
            raise ValueError('label may be revealed only AFTER END_TRIAL')
        if not self.active:
            self.stream.begin_trial(trial_id=p.trial_id,
                                    first_timestamp_us=p.timestamp_us,
                                    allow_output=externally_armed)
            self.active = True
        visible = self.stream.push(to_feature_packet(p))
        self.packet_count += 1
        result = None
        if end:
            result = self.stream.finish_trial(revealed_target=revealed_target)
            self.active = False
        return Received(visible, result)

    def abort(self) -> None:
        if not self.active:
            raise RuntimeError('nothing to abort')
        self.stream.abort_trial()
        self.active = False
        self.aborted_trials += 1
