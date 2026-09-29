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

    def choice_probability(self, score_left: float, score_right: float) -> float:
        # tanh form of the logistic avoids overflow at large beta.
        return 0.5 * (1.0 + np.tanh(0.5 * self.params.beta * (score_left - score_right)))

    def act(self, state: np.ndarray) -> Decision:
        options = [self.option(state, action) for action in (LEFT, RIGHT)]
        score_left, score_right = options[LEFT][3], options[RIGHT][3]
        action = LEFT if self.rng.random() < self.choice_probability(score_left, score_right) else RIGHT
        glomeruli, kc, mbon, _ = options[action]
        self.trace = self.params.trace_decay * self.trace + kc
        self.last_decision = Decision(action, (score_left, score_right), glomeruli, kc, mbon)
        return self.last_decision

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        self.released = (punish, reward)
        self.depress(self.body.dopamine_at_mbon(punish, reward))

    def depress(self, dopamine: np.ndarray) -> None:
        depress(self.gain, self.trace, dopamine, self.params.learning_rate, self.params.gain_decay)


def depress(gain: np.ndarray, trace: np.ndarray, dopamine: np.ndarray, learning_rate: float, decay: float) -> None:
    """Dopamine-gated depression of traced KC->MBON synapses, then recovery toward 1, in place."""
    rows = np.flatnonzero(trace) if learning_rate and dopamine.any() else np.zeros(0, dtype=np.int64)
    if rows.size:
        gain[rows] -= learning_rate * np.outer(trace[rows], dopamine)
    if decay:
        gain += decay * (1.0 - gain)
    # Depression only lowers gains, so only the touched rows can cross the floor.
    if rows.size:
        gain[rows] = np.maximum(gain[rows], 0.0)


class PredictionErrorFly(FlyAgent):
    """Dopamine as a reward prediction error read from the fly's own MBON valence (Bennett et al. 2021)."""

    def expected_score(self, state: np.ndarray) -> float:
        left, right = self.option(state, LEFT)[3], self.option(state, RIGHT)[3]
        p_left = self.choice_probability(left, right)
        return p_left * left + (1.0 - p_left) * right

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        predicted = self.last_decision.scores[self.last_decision.action]
        upcoming = 0.0 if terminated else self.expected_score(next_state)
        error = reward - punish + self.params.gamma * upcoming - predicted
        self.released = (max(-error, 0.0), max(error, 0.0))
        self.depress(self.body.dopamine_at_mbon(*self.released))


class WiredPredictionErrorFly(FlyAgent):
    """Each dopamine neuron predicts from the MBONs that synapse onto it in the connectome."""

    def dan_predictions(self, mbon: np.ndarray) -> np.ndarray:
        # Synapse signs are unknown; MBON valence stands in for them.
        return (self.body.mbon_valence * mbon) @ self.body.mbon_dan

    def expected_predictions(self, state: np.ndarray) -> np.ndarray:
        left, right = self.option(state, LEFT), self.option(state, RIGHT)
        p_left = self.choice_probability(left[3], right[3])
        return p_left * self.dan_predictions(left[2]) + (1.0 - p_left) * self.dan_predictions(right[2])

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        predicted = self.dan_predictions(self.last_decision.mbon)
        upcoming = 0.0 if terminated else self.expected_predictions(next_state)
        errors = reward - punish + self.params.gamma * upcoming - predicted
        punishing = self.body.dan_is_punishment
        self.dan_activity = np.where(punishing, np.maximum(-errors, 0.0), np.maximum(errors, 0.0))
        self.released = (float(self.dan_activity[punishing].mean()), float(self.dan_activity[~punishing].mean()))
        self.depress(self.body.dopamine_from(self.dan_activity))
