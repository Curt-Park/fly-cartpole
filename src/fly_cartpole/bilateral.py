"""Two mushroom bodies, one per push direction, valuing the same state with a prediction-error teacher."""

from __future__ import annotations

import numpy as np

from .circuit import Circuit
from .encoder import LEFT, RIGHT, Encoder, build_encoder
from .fly_agent import Decision
from .mushroom_body import MushroomBody
from .params import Hyperparameters

SIDES = (LEFT, RIGHT)


class BilateralFly:
    """The left mushroom body values pushing left and the right one pushing right."""

    def __init__(self, bodies: tuple[MushroomBody, MushroomBody], encoder: Encoder,
                 pn_groups: tuple[np.ndarray, np.ndarray], params: Hyperparameters, rng: np.random.Generator) -> None:
        self.bodies = bodies
        self.encoder = encoder
        self.pn_groups = pn_groups
        self.params = params
        self.rng = rng
        self.gains = [np.ones((body.n_kc, body.n_mbon)) for body in bodies]
        self.reset_episode()

    def reset_episode(self) -> None:
        self.traces = [np.zeros(body.n_kc) for body in self.bodies]

    def hemisphere(self, glomeruli: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray, float]:
        body = self.bodies[side]
        kc = body.kenyon(glomeruli[self.pn_groups[side]])
        mbon = body.mbon(kc, self.gains[side])
        return kc, mbon, float(mbon @ body.mbon_valence)

    def values(self, state: np.ndarray) -> np.ndarray:
        glomeruli = self.encoder.glomeruli(state, None)
        return np.array([self.hemisphere(glomeruli, side)[2] for side in SIDES])

    def choice_probability(self, value_left: float, value_right: float) -> float:
        # tanh form of the logistic avoids overflow at large beta.
        return 0.5 * (1.0 + np.tanh(0.5 * self.params.bilateral_beta * (value_left - value_right)))

    def act(self, state: np.ndarray) -> Decision:
        glomeruli = self.encoder.glomeruli(state, None)
        hemispheres = [self.hemisphere(glomeruli, side) for side in SIDES]
        value_left, value_right = hemispheres[LEFT][2], hemispheres[RIGHT][2]
        action = LEFT if self.rng.random() < self.choice_probability(value_left, value_right) else RIGHT
        kc, mbon, _ = hemispheres[action]
        for side in SIDES:
            self.traces[side] *= self.params.bilateral_trace_decay
        self.traces[action] += kc
        self.last_decision = Decision(action, (value_left, value_right), glomeruli, kc, mbon)
        return self.last_decision

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        predicted = self.last_decision.scores[self.last_decision.action]
        if terminated:
            upcoming = 0.0
        else:
            next_values = self.values(next_state)
            p_left = self.choice_probability(*next_values)
            upcoming = p_left * next_values[LEFT] + (1.0 - p_left) * next_values[RIGHT]
        error = reward - punish + self.params.gamma * upcoming - predicted
        self.released = (max(-error, 0.0), max(error, 0.0))
        for side, body in zip(SIDES, self.bodies):
            dopamine = body.dopamine_at_mbon(*self.released)
            if self.params.bilateral_learning_rate and dopamine.any():
                self.gains[side] -= self.params.bilateral_learning_rate * np.outer(self.traces[side], dopamine)
            self.gains[side] += self.params.bilateral_gain_decay * (1.0 - self.gains[side])
            np.clip(self.gains[side], 0.0, 1.0, out=self.gains[side])


def build_bilateral(left: Circuit, right: Circuit, params: Hyperparameters, seed: int) -> BilateralFly:
    # One encoder over both sides' glomerulus labels: sister glomeruli left and right carry the same signal.
    labels = np.concatenate([left.pn_glomerulus, right.pn_glomerulus])
    encoder = build_encoder(labels, action_fraction=0.0, tuning_width=params.tuning_width, seed=seed)
    pn_groups = (encoder.pn_group[: left.n_pn], encoder.pn_group[left.n_pn :])
    bodies = (MushroomBody(left, params.kc_sparsity), MushroomBody(right, params.kc_sparsity))
    return BilateralFly(bodies, encoder, pn_groups, params, np.random.default_rng(seed))
