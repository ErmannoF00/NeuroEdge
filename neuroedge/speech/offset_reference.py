"""existing: signed Q4 offset application, transactional software oracle.

This is an input-calibration mechanism, not an adaptation or clinical model.
One staged candidate, no request queue; authorizations outside a safe boundary
are rejected without consuming the staged candidate. Offsets persist across bins,
not across reset. Updates apply only to future bins.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .fixed_decoder import FixedStreamingDecoder, FixedParameters, INPUTS


def saturating_offset_add(raw: int, offset: int) -> int:
    if not -128 <= int(raw) <= 127:
        raise ValueError("raw must be signed INT8")
    if not -32768 <= int(offset) <= 32767:
        raise ValueError("offset must be signed INT16")
    return max(-128, min(127, int(raw) + int(offset)))


def apply_offsets(raw: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    offsets = np.asarray(offsets)
    if raw.dtype != np.int8 or raw.shape[-1:] != (INPUTS,):
        raise ValueError("input must be signed INT8 with 512 channels")
    if offsets.dtype != np.int16 or offsets.shape != (INPUTS,):
        raise ValueError("offset table must be signed INT16[512]")
    return np.clip(raw.astype(np.int32) + offsets.astype(np.int32), -128, 127).astype(np.int8)


@dataclass
class OffsetTable:
    active: np.ndarray = field(default_factory=lambda: np.zeros(INPUTS, dtype=np.int16))
    valid: np.ndarray = field(default_factory=lambda: np.zeros(INPUTS, dtype=np.bool_))
    pending: tuple[int, int] | None = None
    accepted_writes: int = 0
    rejected_requests: int = 0
    stages_accepted: int = 0
    aborted_stages: int = 0
    bin_in_flight: bool = False
    def stage(self, channel: int, value: int) -> bool:
        if not 0 <= channel < INPUTS or not -32768 <= value <= 32767:
            raise ValueError("invalid channel or offset")
        if self.pending is not None:
            return False
        self.pending = (int(channel), int(value))
        self.stages_accepted += 1
        return True
    def abort(self) -> None:
        if self.pending is not None:
            self.aborted_stages += 1
        self.pending = None
    def authorize(self) -> bool:
        if self.bin_in_flight or self.pending is None:
            self.rejected_requests += 1
            return False
        channel, value = self.pending
        self.active[channel] = value
        self.valid[channel] = True
        self.pending = None
        self.accepted_writes += 1
        return True
    def reset(self) -> None:
        # Match RTL: physical contents remain, but valid bits are cleared.
        self.valid.fill(False)
        self.pending = None
        self.accepted_writes = self.rejected_requests = 0
        self.stages_accepted = self.aborted_stages = 0
        self.bin_in_flight = False
    def begin_bin(self) -> None:
        if self.bin_in_flight:
            raise RuntimeError("in-flight bin")
        self.bin_in_flight = True
    def end_bin(self) -> None:
        if not self.bin_in_flight:
            raise RuntimeError("no in-flight bin")
        self.bin_in_flight = False
    def read_channel(self, channel: int) -> int:
        if not 0 <= channel < INPUTS:
            raise ValueError('channel must be 0..511')
        return int(self.active[channel]) if self.valid[channel] else 0
    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return apply_offsets(raw, np.where(self.valid,self.active,0).astype(np.int16))


class OffsetStreamingDecoder:
    def __init__(self, parameters: FixedParameters):
        self.table = OffsetTable()
        self.decoder = FixedStreamingDecoder(parameters)
    def push(self, raw: np.ndarray) -> np.ndarray | None:
        self.table.begin_bin()
        try:
            value = self.decoder.push(self.table.calibrate(raw))
        finally:
            self.table.end_bin()
        return value
