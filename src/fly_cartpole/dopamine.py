"""When dopamine is released: punishment on a fall, reward while beating the recent norm."""

from __future__ import annotations

import numpy as np

BASELINE_MODES = ("mean", "best")


class DopamineSchedule:
    def __init__(self, mode: str, window: int, reward_per_step: float) -> None:
        if mode not in BASELINE_MODES:
            raise ValueError(f"baseline mode must be one of {BASELINE_MODES}, got {mode!r}")
        self.mode = mode
        self.window = window
        self.reward_per_step = reward_per_step
        self.lengths: list[int] = []

    @property
    def baseline(self) -> float | None:
        if not self.lengths:
            return None
        if self.mode == "best":
            return float(max(self.lengths))
        return float(np.mean(self.lengths[-self.window :]))

    def signal(self, step: int, terminated: bool) -> tuple[float, float]:
        """(punish, reward) for the step just completed; steps count from 1."""
        if terminated:
            return 1.0, 0.0
        baseline = self.baseline
        if baseline is not None and step > baseline:
            return 0.0, self.reward_per_step
        return 0.0, 0.0

    def end_episode(self, length: int) -> None:
        self.lengths.append(length)
