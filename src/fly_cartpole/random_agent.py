"""Chance reference: presses a uniformly random button and never learns."""

from __future__ import annotations

import numpy as np

from .encoder import Encoder
from .fly_agent import Decision


class RandomAgent:
    def __init__(self, encoder: Encoder, n_kc: int, rng: np.random.Generator) -> None:
        self.encoder = encoder
        self.n_kc = n_kc
        self.rng = rng

    def reset_episode(self) -> None:
        pass

    def act(self, state: np.ndarray) -> Decision:
        action = int(self.rng.integers(2))
        return Decision(action, (0.0, 0.0), self.encoder.glomeruli(state, action), np.zeros(self.n_kc), np.zeros(0))

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        pass
