"""Exact clock-enable policy model of neuroedge_budget_controller.sv."""
from dataclasses import dataclass

@dataclass
class BudgetController:
    window_bins: int = 50
    max_writes: int = 4
    cooldown_bins: int = 5
    high_threshold: int = 1200
    low_threshold: int = 800
    active: bool = False
    window_position: int = 0
    writes: int = 0
    cooldown: int = 0
    observed: int = 0
    suppressed: int = 0

    def sample(self, score: int) -> tuple[int, int, int, int]:
        if not (0 <= score <= 65535):
            raise ValueError('unsigned 16-bit score required')
        if score >= self.high_threshold:
            self.active = True
        elif score <= self.low_threshold:
            self.active = False
        if self.window_position == self.window_bins - 1:
            self.window_position = 0
            self.writes = 0
        else:
            self.window_position += 1
        previous_cooldown = self.cooldown
        self.cooldown = max(0, self.cooldown - 1)
        issued = 0
        if self.active:
            if self.writes < self.max_writes and previous_cooldown == 0:
                self.writes += 1
                self.cooldown = self.cooldown_bins
                issued = 1
            else:
                self.suppressed += 1
        self.observed += 1
        return int(self.active), issued, self.writes, self.suppressed
