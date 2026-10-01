"""Export the reflex fly's flight circuit, its fixed and self-tuned gains, and parity checks for the web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .dopamine import DopamineSchedule
from .encoder import STATE_LIMITS
from .flight import ROLES
from .paths import FLIGHT_PATH, WEB_DATA_DIR
from .reflex import FlightCircuit, ReflexFly, load_flight
from .reflex_report import ReflexParameters, make_reflex_agent
from .run import run_episodes


def scene_positions(soma: np.ndarray) -> list:
    """Centre and scale soma coordinates into [-1, 1]; missing somas become None; dorsal is up."""
    known = soma[np.isfinite(soma).all(axis=1)]
    # Centre on the extent: cells crowd the nerve cord, and a mean centre would push the brain out of frame.
    centre = (known.min(axis=0) + known.max(axis=0)) / 2
    scale = np.abs(known - centre).max()
    flip = np.array([1.0, -1.0, 1.0])
    return [np.round((row - centre) / scale * flip, 4).tolist() if np.isfinite(row).all() else None for row in soma]


def display_positions(circuit: FlightCircuit) -> np.ndarray:
    """Cell bodies where known; afferents whose somata lie outside the nervous system are drawn where they synapse."""
    positions = circuit.soma.copy()
    known = np.isfinite(circuit.soma).all(axis=1)
    for neuron in np.flatnonzero(~known):
        targets = circuit.post[circuit.pre == neuron]
        targets = targets[known[targets]]
        if len(targets):
            positions[neuron] = circuit.soma[targets].mean(axis=0)
    return positions


def physics_check(seed: int, steps: int = 60) -> dict:
    """A Gymnasium trajectory the browser's CartPole must reproduce."""
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    start = [float(value) for value in env.unwrapped.state]
    actions = np.random.default_rng(seed).integers(2, size=steps).tolist()
    states = []
    for action in actions:
        _, _, terminated, _, _ = env.step(action)
        states.append([float(value) for value in env.unwrapped.state])
        if terminated:
            break
    env.close()
    return {"start": start, "actions": actions[: len(states)], "states": states}


def reflex_trajectory(circuit: FlightCircuit, gains: list[float], params: ReflexParameters, seed: int, steps: int = 150) -> dict:
    """States the Python reflex saw and what it did; the browser must act the same on the same states."""
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=seed)
    fly = ReflexFly(circuit, tuple(gains), params.substeps, params.leak)
    states, actions, steers = [], [], []
    for _ in range(steps):
        decision = fly.act(state)
        states.append([float(value) for value in state])
        actions.append(decision.action)
        steers.append(decision.steer)
        state, _, terminated, truncated, _ = env.step(decision.action)
        if terminated or truncated:
            break
    env.close()
    return {"gains": list(gains), "states": states, "actions": actions, "steers": steers}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: the browser's JSON.parse rejects NaN.
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))


def export_flight(seed: int, params: ReflexParameters, flight_path: Path = FLIGHT_PATH, web_data_dir: Path = WEB_DATA_DIR) -> dict:
    circuit = load_flight(flight_path)
    adaptive = make_reflex_agent("fly-reflex-adaptive", circuit, params, seed)
    lengths = run_episodes(adaptive, DopamineSchedule("mean", params.baseline_window, 0.0), params.adapt_episodes, seed)
    gains_adapted = np.exp(adaptive.log_gains).tolist()
    write_json(web_data_dir / "flight.json", {
        "roles": list(ROLES),
        "role": circuit.role.tolist(),
        "side": circuit.side.tolist(),
        "amplitude": circuit.amplitude.tolist(),
        "pre": circuit.pre.tolist(),
        "post": circuit.post.tolist(),
        "coupling": circuit.coupling.tolist(),
        "positions": scene_positions(display_positions(circuit)),
        "state_limits": STATE_LIMITS.tolist(),
        "leak": params.leak,
        "substeps": params.substeps,
        "gains_fixed": [1.0, 1.0, 1.0],
        "gains_adapted": gains_adapted,
        "trajectory": reflex_trajectory(circuit, gains_adapted, params, seed),
        "physics": physics_check(seed),
    })
    return {
        "seed": seed,
        "adaptation_episodes": params.adapt_episodes,
        "final_100_mean": float(np.mean(lengths[-100:])),
        "longest": max(lengths),
        "gains_adapted": gains_adapted,
    }
