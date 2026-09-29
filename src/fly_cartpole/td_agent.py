"""Reference learner: TD actor-critic with eligibility traces on the same Kenyon cell code."""

from __future__ import annotations

import numpy as np

from .encoder import LEFT, RIGHT, Encoder
from .fly_agent import Decision
from .mushroom_body import MushroomBody
from .params import Hyperparameters


class TDAgent:
    def __init__(self, body: MushroomBody, encoder: Encoder, params: Hyperparameters, rng: np.random.Generator) -> None:
        self.body = body
        self.encoder = encoder
        self.params = params
        self.rng = rng
        self.preferences = np.zeros((2, body.n_kc))
        self.value = np.zeros(body.n_kc)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.actor_trace = np.zeros_like(self.preferences)
        self.critic_trace = np.zeros_like(self.value)
        self.last: tuple[np.ndarray, np.ndarray, int] | None = None

    def features(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        glomeruli = self.encoder.glomeruli(state, None)
        return glomeruli, self.body.kenyon(glomeruli[self.encoder.pn_group])

    def policy(self, kc: np.ndarray) -> np.ndarray:
        logits = self.params.beta * (self.preferences @ kc)
        weights = np.exp(logits - logits.max())
        return weights / weights.sum()

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        return self.policy(self.features(state)[1])

    def state_value(self, state: np.ndarray) -> float:
        return float(self.value @ self.features(state)[1])

    def act(self, state: np.ndarray) -> Decision:
        glomeruli, kc = self.features(state)
        probabilities = self.policy(kc)
        action = LEFT if self.rng.random() < probabilities[LEFT] else RIGHT
        self.last = (kc, probabilities, action)
        preferences = self.preferences @ kc
        return Decision(action, (float(preferences[0]), float(preferences[1])), glomeruli, kc, np.zeros(0))

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        kc, probabilities, action = self.last
        params = self.params
        next_value = 0.0 if terminated else self.state_value(next_state)
        td_error = reward - punish + params.gamma * next_value - float(self.value @ kc)
        decay = params.gamma * params.td_trace_decay
        score = -probabilities
        score[action] += 1.0
        self.critic_trace = decay * self.critic_trace + kc
        self.actor_trace = decay * self.actor_trace + params.beta * np.outer(score, kc)
        # Divide by the active-cell count so one update cannot overshoot its target (tile-coding convention).
        self.value += params.td_critic_lr / self.body.k * td_error * self.critic_trace
        self.preferences += params.td_actor_lr / self.body.k * td_error * self.actor_trace
