"""When dopamine is released: punishment on a fall, reward while beating the recent norm, and a posture signal."""

from __future__ import annotations

import numpy as np

BASELINE_MODES = ("mean", "best")
# CartPole-v1 ends an episode beyond these; posture is measured against them.
TRACK_LIMIT = 2.4
ANGLE_LIMIT = 12 * 2 * np.pi / 360


class DopamineSchedule:
    def __init__(self, mode: str, window: int, reward_per_step: float, posture_weight: float = 0.0, gamma: float = 1.0) -> None:
        if mode not in BASELINE_MODES:
            raise ValueError(f"baseline mode must be one of {BASELINE_MODES}, got {mode!r}")
        self.mode = mode
        self.window = window
        self.reward_per_step = reward_per_step
        self.posture_weight = posture_weight
        self.gamma = gamma
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

    def posture(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """Reward for moving toward upright and centred, punishment for moving away (potential-based shaping)."""
        if not self.posture_weight:
            return 0.0
        return self.posture_weight * (self.gamma * posture_potential(next_state) - posture_potential(state))

    def end_episode(self, length: int) -> None:
        self.lengths.append(length)


def posture_potential(state: np.ndarray) -> float:
    return -float((state[2] / ANGLE_LIMIT) ** 2 + (state[0] / TRACK_LIMIT) ** 2)
