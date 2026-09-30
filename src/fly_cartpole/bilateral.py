"""Two mushroom bodies, one per push direction, valuing the same state with a prediction-error teacher."""

from __future__ import annotations

import numpy as np

from .circuit import Circuit
from .encoder import LEFT, RIGHT, STATE_LIMITS, Encoder, build_encoder
from .fly_agent import Decision, update_gains
from .mushroom_body import MushroomBody
from .params import Hyperparameters

SIDES = (LEFT, RIGHT)
CALIBRATION_STATES = 256


class BilateralFly:
    """The left mushroom body values pushing left and the right one pushing right."""

    def __init__(self, bodies: tuple[MushroomBody, MushroomBody], encoder: Encoder,
                 pn_groups: tuple[np.ndarray, np.ndarray], params: Hyperparameters, rng: np.random.Generator) -> None:
        self.bodies = bodies
        self.encoder = encoder
        self.pn_groups = pn_groups
        self.params = params
        self.rng = rng
        self.gains = [np.full((body.n_kc, body.n_mbon), params.bilateral_initial_gain) for body in bodies]
        self.offsets = np.zeros(len(SIDES))
        # Homeostatic set point: each hemisphere's innate average value over typical states is taken as zero.
        self.calibration_states = np.random.default_rng(0).uniform(-0.5, 0.5, size=(CALIBRATION_STATES, len(STATE_LIMITS))) * STATE_LIMITS
        if params.bilateral_centre:
            self.offsets = np.mean([self.values(state) for state in self.calibration_states], axis=0)
        self.episodes_started = -1
        self.reset_episode()

    def reset_episode(self) -> None:
        self.episodes_started += 1
        self.traces = [np.zeros(body.n_kc) for body in self.bodies]

    @property
    def learning_rate(self) -> float:
        # Plasticity settles with experience, so a fly that balances well stops unlearning it.
        settle = self.params.bilateral_settle_episodes
        rate = self.params.bilateral_learning_rate
        return rate / (1.0 + self.episodes_started / settle) if settle else rate

    def hemisphere(self, glomeruli: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray, float]:
        body = self.bodies[side]
        kc = body.kenyon(glomeruli[self.pn_groups[side]])
        mbon = body.mbon(kc, self.gains[side])
        return kc, mbon, float(mbon @ body.mbon_valence) - self.offsets[side]

    def values(self, state: np.ndarray) -> np.ndarray:
        glomeruli = self.encoder.glomeruli(state, None)
        return np.array([self.hemisphere(glomeruli, side)[2] for side in SIDES])

    def choice_probability(self, value_left: float, value_right: float) -> float:
        # tanh form of the logistic avoids overflow at large beta.
        committed = 0.5 * (1.0 + np.tanh(0.5 * self.params.bilateral_beta * (value_left - value_right)))
        # Spontaneous choices ignore the values, so a side the fly has written off still gets retried.
        spontaneous = self.params.bilateral_spontaneous
        return spontaneous / 2 + (1.0 - spontaneous) * committed

    def act(self, state: np.ndarray) -> Decision:
        glomeruli = self.encoder.glomeruli(state, None)
        hemispheres = [self.hemisphere(glomeruli, side) for side in SIDES]
        value_left, value_right = hemispheres[LEFT][2], hemispheres[RIGHT][2]
        action = LEFT if self.rng.random() < self.choice_probability(value_left, value_right) else RIGHT
        kc, mbon, _ = hemispheres[action]
        for side in SIDES:
            self.traces[side] *= self.params.bilateral_trace_decay
            # Traces this faint change no gain measurably; dropping them keeps learning sparse and fast.
            self.traces[side][self.traces[side] < 1e-10] = 0.0
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
        # Outcomes are scaled into the value range that bounded gains can express.
        error = self.params.bilateral_outcome_scale * (reward - punish) + self.params.gamma * upcoming - predicted
        self.released = (max(-error, 0.0), max(error, 0.0))
        worse, better = self.released
        push_pull = self.params.bilateral_plasticity == "push-pull"
        for side, body in zip(SIDES, self.bodies):
            # Push-pull: where the opposing dopamine population lands, the same traced synapses recover instead.
            restore_at = body.dopamine_at_mbon(better, worse) if push_pull else None
            update_gains(self.gains[side], self.traces[side], body.dopamine_at_mbon(worse, better), restore_at,
                         self.learning_rate, self.params.bilateral_gain_decay)


def build_bilateral(left: Circuit, right: Circuit, params: Hyperparameters, seed: int) -> BilateralFly:
    # One encoder over both sides' glomerulus labels: sister glomeruli left and right carry the same signal.
    labels = np.concatenate([left.pn_glomerulus, right.pn_glomerulus])
    encoder = build_encoder(labels, action_fraction=0.0, tuning_width=params.tuning_width, seed=seed)
    pn_groups = (encoder.pn_group[: left.n_pn], encoder.pn_group[left.n_pn :])
    bodies = (MushroomBody(left, params.kc_sparsity), MushroomBody(right, params.kc_sparsity))
    return BilateralFly(bodies, encoder, pn_groups, params, np.random.default_rng(seed))
