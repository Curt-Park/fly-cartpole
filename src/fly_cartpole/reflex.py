"""The reflex fly: the flight-stabilisation circuit, wired as measured, steering the cart without learning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .circuit import shuffle_edges
from .encoder import STATE_LIMITS
from .flight import ROLES
from .paths import FLIGHT_PATH

# Gains are ordered by sense: ocelli read the angle, halteres the angular velocity, HS cells the cart's drift,
# and a landmark at the track centre the cart's position.
SENSES = ("angle", "rate", "drift", "position")
X, X_DOT, ANGLE, ANGLE_DOT = 0, 1, 2, 3
WIRING_ONLY = (1.0, 1.0, 1.0, 0.0)
WITH_LANDMARK = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class FlightCircuit:
    body_id: np.ndarray
    cell_type: np.ndarray
    role: np.ndarray
    side: np.ndarray
    amplitude: np.ndarray
    soma: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    coupling: np.ndarray

    @property
    def size(self) -> int:
        return len(self.role)

    def members(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.role == ROLES.index(role))


def load_flight(path: Path = FLIGHT_PATH) -> FlightCircuit:
    with np.load(path) as arrays:
        return FlightCircuit(
            body_id=arrays["body_id"], cell_type=arrays["type"].astype(str), role=arrays["role"].astype(np.int8),
            side=arrays["side"].astype(np.int8), amplitude=arrays["amplitude"].astype(bool), soma=arrays["soma"],
            pre=arrays["pre"].astype(np.int64), post=arrays["post"].astype(np.int64),
            coupling=arrays["coupling"].astype(np.float64),
        )


def shuffle_flight(circuit: FlightCircuit, rng: np.random.Generator) -> FlightCircuit:
    # Sensors and motors keep their identities; each connection keeps its coupling but may change target.
    return replace(circuit, post=shuffle_edges(circuit.pre, circuit.post, rng))


def sensor_drive(circuit: FlightCircuit, state: np.ndarray, gains: tuple[float, float, float, float]) -> np.ndarray:
    """A pole tilting or falling right is read as the body rolling right; a cart moving right as leftward optic flow."""
    normalised = np.clip(state / STATE_LIMITS, -1.0, 1.0)
    drive = np.zeros(circuit.size)
    ocellar, haltere, hs = circuit.members("ocellar"), circuit.members("haltere"), circuit.members("hs")
    # The upper ocellus sees more sky, and ocellar L-neurons are hyperpolarised by light.
    # To slide back toward the landmark a fly first banks toward it, so position shifts the attitude it holds.
    drive[ocellar] = (gains[0] * normalised[ANGLE] + gains[3] * normalised[X]) * circuit.side[ocellar]
    # The haltere on the side moving down is excited (the corrective haltere-to-b1 reflex).
    drive[haltere] = gains[1] * normalised[ANGLE_DOT] * circuit.side[haltere]
    # Leftward frontal flow is front-to-back on the left eye, the HS cells' preferred direction.
    drive[hs] = -gains[2] * normalised[X_DOT] * circuit.side[hs]
    return drive


def relative_gains(log_gains: np.ndarray) -> list[float]:
    """Gains with a geometric mean of 1: scaling every sense alike leaves the linear reflex's choices unchanged."""
    return np.exp(log_gains - log_gains.mean()).tolist() + [0.0] * (len(SENSES) - len(log_gains))


class ReflexDecision(NamedTuple):
    action: int
    steer: float


class ReflexFly:
    """Leaky signed rate units; activity is the deviation from tonic firing, so inhibition is negative."""

    def __init__(self, circuit: FlightCircuit, gains: tuple[float, float, float, float] = WIRING_ONLY,
                 substeps: int = 4, leak: float = 0.5) -> None:
        self.circuit = circuit
        self.gains = tuple(gains)
        self.substeps = substeps
        self.leak = leak
        self.right_amplitude = np.flatnonzero(circuit.amplitude & (circuit.side > 0))
        self.left_amplitude = np.flatnonzero(circuit.amplitude & (circuit.side < 0))
        self.released = (0.0, 0.0)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.activity = np.zeros(self.circuit.size)

    def step_circuit(self, drive: np.ndarray) -> None:
        circuit = self.circuit
        for _ in range(self.substeps):
            synaptic = np.bincount(circuit.post, weights=self.activity[circuit.pre] * circuit.coupling, minlength=circuit.size)
            self.activity = (1.0 - self.leak) * self.activity + self.leak * (synaptic + drive)

    def act(self, state: np.ndarray) -> ReflexDecision:
        self.step_circuit(sensor_drive(self.circuit, state, self.gains))
        # A body rolling right is righted by a stronger right wingbeat, matched to pushing the cart right.
        steer = float(self.activity[self.right_amplitude].sum() - self.activity[self.left_amplitude].sum())
        return ReflexDecision(1 if steer > 0 else 0, steer)

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        """Reflexes do not learn from outcomes."""


class AdaptiveReflexFly(ReflexFly):
    """Tries slightly different sensor gains each episode and keeps changes that beat its recent record."""

    def __init__(self, circuit: FlightCircuit, rng: np.random.Generator, eta: float, sigma: float,
                 baseline_window: int = 20, substeps: int = 4, leak: float = 0.5, station_keeping: bool = False) -> None:
        self.rng = rng
        self.eta = eta
        self.sigma = sigma
        self.baseline_window = baseline_window
        # Without a landmark the position gain stays at zero and is never tried.
        self.adapted = len(SENSES) if station_keeping else len(SENSES) - 1
        self.log_gains = np.zeros(self.adapted)
        self.trial = np.zeros(self.adapted)
        self.lengths: list[int] = []
        self.steps = 0
        super().__init__(circuit, WIRING_ONLY, substeps, leak)

    def gains_from(self, log_gains: np.ndarray) -> tuple[float, ...]:
        return tuple(np.exp(log_gains)) + (0.0,) * (len(SENSES) - self.adapted)

    def tuned_gains(self) -> tuple[float, ...]:
        return self.gains_from(self.log_gains)

    def finish_episode(self, length: int) -> None:
        if self.lengths:
            baseline = float(np.mean(self.lengths[-self.baseline_window:]))
            # Weight perturbation: only magnitudes adapt, the wiring keeps each pathway's sign.
            dopamine = (length - baseline) / baseline
            self.log_gains += self.eta * dopamine * self.trial / self.sigma
        self.lengths.append(length)

    def reset_episode(self) -> None:
        if self.steps:
            self.finish_episode(self.steps)
        self.steps = 0
        self.trial = self.rng.normal(0.0, self.sigma, self.adapted)
        self.gains = self.gains_from(self.log_gains + self.trial)
        super().reset_episode()

    def act(self, state: np.ndarray) -> ReflexDecision:
        self.steps += 1
        return super().act(state)
