"""Little-endian binary ABI shared with firmware/stream/neuroedge_feature_packet.c.

Transport framing/checksum is NOT authentication, does NOT imply a live ADC or
allow a feature source to authorize private speech output. Transport control
and user consent must travel over a separately authenticated channel.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
import zlib
import numpy as np

MAGIC = 0x5255454e
VERSION = 1
FEATURES = 512
FLAG_BEGIN_TRIAL = 1
FLAG_END_TRIAL = 2
ALLOWED_FLAGS = FLAG_BEGIN_TRIAL | FLAG_END_TRIAL
HEADER_FORMAT = '<IHHIIQ'
WIRE_SIZE = 540


@dataclass(frozen=True)
class WirePacket:
    flags: int
    trial_id: int
    bin_index: int
    timestamp_us: int
    features: np.ndarray

    def validate(self):
        if self.flags < 0 or self.flags & ~ALLOWED_FLAGS:
            raise ValueError('invalid data plane flags')
        if not 0 <= self.trial_id < 2**32 or not 0 <= self.bin_index < 2**32:
            raise ValueError('trial/bin value overflow')
        if not 0 <= self.timestamp_us < 2**64:
            raise ValueError('timestamp overflow')
        if not isinstance(self.features,np.ndarray) or self.features.shape != (FEATURES,) or self.features.dtype != np.int8:
            raise ValueError('expected 512 signed INT8 features')


def encode(packet: WirePacket) -> bytes:
    packet.validate()
    head = struct.pack(HEADER_FORMAT,MAGIC,VERSION,packet.flags,
                       packet.trial_id,packet.bin_index,packet.timestamp_us)
    body = head + packet.features.tobytes()
    if len(body)!=536:raise AssertionError('bad binary ABI size')
    return body+struct.pack('<I',zlib.crc32(body))


def decode(raw: bytes) -> WirePacket:
    if len(raw) != WIRE_SIZE:
        raise ValueError('incorrect packet length')
    magic, version, flags, trial, index, timestamp = struct.unpack(HEADER_FORMAT,raw[:24])
    if magic!=MAGIC or version!=VERSION or flags & ~ALLOWED_FLAGS:
        raise ValueError('bad header or reserved flags')
    if struct.unpack('<I',raw[-4:])[0] != zlib.crc32(raw[:-4]):
        raise ValueError('CRC mismatch')
    packet=WirePacket(flags,trial,index,timestamp,np.frombuffer(raw[24:536],dtype=np.int8).copy())
    packet.validate()
    return packet


def to_feature_packet(packet: WirePacket):
    """Keep consent out of the data plane; flags only mark trial boundaries."""
    from neuroedge.system.streaming import FeaturePacket
    return FeaturePacket(packet.trial_id, packet.bin_index, packet.timestamp_us, packet.features.copy())
