"""The fly: a T-maze choice between two imagined odours, and dopamine-gated depression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder import LEFT, RIGHT, Encoder
from .mushroom_body import MushroomBody
from .params import Hyperparameters


@dataclass(frozen=True)
class Decision:
    action: int
    scores: tuple[float, float]
    glomeruli: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray


class FlyAgent:
    def __init__(self, body: MushroomBody, encoder: Encoder, params: Hyperparameters, rng: np.random.Generator) -> None:
        self.body = body
        self.encoder = encoder
        self.params = params
        self.rng = rng
        self.gain = np.ones((body.n_kc, body.n_mbon))
        self.trace = np.zeros(body.n_kc)

    def reset_episode(self) -> None:
        self.trace = np.zeros(self.body.n_kc)

    def option(self, state: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        glomeruli = self.encoder.glomeruli(state, action)
        kc = self.body.kenyon(glomeruli[self.encoder.pn_group])
        mbon = self.body.mbon(kc, self.gain)
        return glomeruli, kc, mbon, float(mbon @ self.body.mbon_valence)

    def act(self, state: np.ndarray) -> Decision:
        options = [self.option(state, action) for action in (LEFT, RIGHT)]
        score_left, score_right = options[LEFT][3], options[RIGHT][3]
        # tanh form of the logistic avoids overflow at large beta.
        p_left = 0.5 * (1.0 + np.tanh(0.5 * self.params.beta * (score_left - score_right)))
        action = LEFT if self.rng.random() < p_left else RIGHT
        glomeruli, kc, mbon, _ = options[action]
        self.trace = self.params.trace_decay * self.trace + kc
        return Decision(action, (score_left, score_right), glomeruli, kc, mbon)

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        dopamine = self.body.dopamine_at_mbon(punish, reward)
        if self.params.learning_rate and dopamine.any():
            self.gain -= self.params.learning_rate * np.outer(self.trace, dopamine)
        self.gain += self.params.gain_decay * (1.0 - self.gain)
        np.clip(self.gain, 0.0, 1.0, out=self.gain)
