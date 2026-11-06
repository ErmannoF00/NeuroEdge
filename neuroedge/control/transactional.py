"""Cycle-edge reference for a single-entry staged 512-channel offset RAM.

Not an optimizer. Stage writes are invisible until a separately authorized
request is accepted while the decoder is idle. Invalid (unwritten) entries
read as zero. Count actual committed writes, never inferred energy.
"""
from dataclasses import dataclass,field

@dataclass
class OffsetCommitBank:
    channels:int=512
    active:list[int]=field(default_factory=lambda:[0]*512)
    valid:list[bool]=field(default_factory=lambda:[False]*512)
    pending:tuple[int,int] | None=None
    writes:int=0
    rejected:int=0
    accepted_stages:int=0
    aborted:int=0

    def __post_init__(self):
        if self.channels!=512: raise ValueError('existing supports exactly 512 channels')

    def stage(self,channel:int,value:int)->bool:
        if not 0<=channel<512 or not -32768<=value<=32767: raise ValueError('stage out of range')
        if self.pending is not None: return False
        self.pending=(channel,value)
        self.accepted_stages+=1
        return True

    def tick(self, *, update_request:bool=False, decoder_idle:bool=True, abort:bool=False)->bool:
        if abort:
            if self.pending is not None: self.aborted+=1
            self.pending=None
            return False
        if not update_request: return False
        if self.pending is None or not decoder_idle:
            self.rejected+=1
            return False
        channel,value=self.pending
        self.active[channel]=value
        self.valid[channel]=True
        self.pending=None
        self.writes+=1
        return True

    def read(self,channel:int)->int:
        if not 0<=channel<512:raise ValueError('channel out of range')
        return self.active[channel] if self.valid[channel] else 0

    def reset(self):
        self.valid=[False]*512
        self.pending=None
        self.writes=self.rejected=self.accepted_stages=self.aborted=0
