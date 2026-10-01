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
# and a landmark at the track centre the cart's position and, as it slides across the eye, its motion.
SENSES = ("angle", "rate", "drift", "position", "motion")
X, X_DOT, ANGLE, ANGLE_DOT = 0, 1, 2, 3
WIRING_ONLY = (1.0, 1.0, 1.0, 0.0, 0.0)
WITH_LANDMARK = (1.0, 1.0, 1.0, 1.0, 0.0)
# How many senses a fly tunes: none of the landmark, its position, or its position and motion.
LANDMARKS = {"none": 3, "position": 4, "motion": 5}
BALANCE_SENSES = 3  # angle, rate and drift: the senses that hold the pole up
# What an episode is judged by: how long the pole stayed up, how long the cart stayed near the landmark, or both halves.
RECORDS = ("length", "centred", "balanced")


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


def sensor_drive(circuit: FlightCircuit, state: np.ndarray, gains: tuple[float, ...]) -> np.ndarray:
    """A pole tilting or falling right is read as the body rolling right; a cart moving right as leftward optic flow."""
    normalised = np.clip(state / STATE_LIMITS, -1.0, 1.0)
    drive = np.zeros(circuit.size)
    ocellar, haltere, hs = circuit.members("ocellar"), circuit.members("haltere"), circuit.members("hs")
    # The upper ocellus sees more sky, and ocellar L-neurons are hyperpolarised by light.
    # To slide back toward the landmark a fly first banks toward it, so position shifts the attitude it holds;
    # the landmark's slide across the eye shifts it too, damping the return.
    drive[ocellar] = (gains[0] * normalised[ANGLE] + gains[3] * normalised[X] + gains[4] * normalised[X_DOT]) * circuit.side[ocellar]
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

    def __init__(self, circuit: FlightCircuit, gains: tuple[float, ...] = WIRING_ONLY,
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
                 baseline_window: int = 20, substeps: int = 4, leak: float = 0.5, landmark: str = "none",
                 record: str = "length", curriculum: int = 0) -> None:
        if landmark not in LANDMARKS:
            raise ValueError(f"landmark must be one of {tuple(LANDMARKS)}, got {landmark!r}")
        if record not in RECORDS:
            raise ValueError(f"record must be one of {RECORDS}, got {record!r}")
        if curriculum and landmark == "none":
            raise ValueError("a curriculum needs a landmark to learn in its second stage")
        self.record = record
        # Episodes spent learning to balance, judged by length, before learning to hold station; 0 means one stage.
        self.curriculum = curriculum
        self.finished = 0
        self.rng = rng
        self.eta = eta
        self.sigma = sigma
        self.baseline_window = baseline_window
        # Landmark senses the fly does not have stay at zero and are never tried.
        self.adapted = LANDMARKS[landmark]
        self.log_gains = np.zeros(self.adapted)
        self.trial = np.zeros(self.adapted)
        self.records: list[float] = []
        self.steps = 0
        self.time_near_landmark = 0.0
        super().__init__(circuit, WIRING_ONLY, substeps, leak)

    def gains_from(self, log_gains: np.ndarray) -> tuple[float, ...]:
        return tuple(np.exp(log_gains)) + (0.0,) * (len(SENSES) - self.adapted)

    def tuned_gains(self) -> tuple[float, ...]:
        return self.gains_from(self.log_gains)

    def finish_episode(self, record: float) -> None:
        if self.records:
            baseline = float(np.mean(self.records[-self.baseline_window:]))
            # Weight perturbation: only magnitudes adapt, the wiring keeps each pathway's sign.
            dopamine = (record - baseline) / baseline
            self.log_gains += self.eta * dopamine * self.trial / self.sigma
        self.records.append(record)

    def reset_episode(self) -> None:
        if self.steps:
            self.finish_episode(self.episode_record())
            self.finished += 1
            if self.finished == self.curriculum:
                # Holding station is judged on its own scale, so its record starts afresh.
                self.records = []
        self.steps = 0
        self.time_near_landmark = 0.0
        self.trial = self.rng.normal(0.0, self.sigma, self.adapted)
        if self.holding_station():
            # Once it balances, the fly keeps its balance gains and explores only the landmark's.
            self.trial[:BALANCE_SENSES] = 0.0
        self.gains = self.gains_from(self.log_gains + self.trial)
        super().reset_episode()

    def holding_station(self) -> bool:
        return bool(self.curriculum) and self.finished >= self.curriculum

    def episode_record(self) -> float:
        if self.curriculum:
            return self.time_near_landmark if self.holding_station() else self.steps
        if self.record == "length":
            return self.steps
        if self.record == "centred":
            return self.time_near_landmark
        return (self.steps + self.time_near_landmark) / 2

    def act(self, state: np.ndarray) -> ReflexDecision:
        self.steps += 1
        # Time near the landmark keeps rewarding station keeping once every episode reaches the cap.
        self.time_near_landmark += 1.0 - min(abs(state[X]) / STATE_LIMITS[X], 1.0)
        return super().act(state)
