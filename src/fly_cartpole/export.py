"""Train the bilateral fly and export it, with checks, for the live web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .bilateral import SIDES, BilateralFly
from .circuit import load_circuit
from .encoder import STATE_LIMITS
from .params import Hyperparameters
from .paths import CIRCUIT_PATH, LEFT_CIRCUIT_PATH, WEB_DATA_DIR
from .run import make_agent, run_episodes

CONDITION = "fly-bilateral"
POPULATIONS = ("pn", "kc", "mbon")


def sparse(matrix: np.ndarray) -> dict[str, list]:
    rows, cols = np.nonzero(matrix)
    return {"rows": rows.tolist(), "cols": cols.tolist(), "values": matrix[rows, cols].tolist()}


def naive_values(fly: BilateralFly, state: np.ndarray) -> list[float]:
    glomeruli = fly.encoder.glomeruli(state, None)
    values = []
    for side, body in zip(SIDES, fly.bodies):
        kc = body.kenyon(glomeruli[fly.pn_groups[side]])
        naive = body.kc_mbon * fly.params.bilateral_initial_gain
        values.append(float((kc @ naive) @ body.mbon_valence) - fly.offsets[side])
    return values


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


def export_model(fly: BilateralFly, baseline: float | None, seed: int) -> dict:
    sides = []
    for side, body in zip(SIDES, fly.bodies):
        rows, cols = np.nonzero(body.kc_mbon)
        sides.append({
            "pn_group": fly.pn_groups[side].tolist(),
            "n_kc": body.n_kc,
            "n_mbon": body.n_mbon,
            "k": body.k,
            "pn_kc": sparse(body.pn_kc),
            "kc_mbon": {"rows": rows.tolist(), "cols": cols.tolist()},
            "kc_mbon_naive": (body.kc_mbon * fly.params.bilateral_initial_gain)[rows, cols].tolist(),
            "kc_mbon_trained": (body.kc_mbon * fly.gains[side])[rows, cols].tolist(),
            "mbon_valence": body.mbon_valence.astype(int).tolist(),
            "offset": float(fly.offsets[side]),
            "dan_mbon": sparse(body.dan_mbon),
        })
    states = np.random.default_rng(seed).uniform(-STATE_LIMITS, STATE_LIMITS, size=(12, len(STATE_LIMITS))) * 0.5
    return {
        "beta": fly.params.bilateral_beta,
        "gamma": fly.params.gamma,
        "reward_per_step": fly.params.reward_per_step,
        "outcome_scale": fly.params.bilateral_outcome_scale,
        "baseline_window": fly.params.baseline_window,
        "initial_baseline": baseline,
        "state_limits": STATE_LIMITS.tolist(),
        "encoder": {
            "group_role": fly.encoder.group_role.tolist(),
            "preferred": fly.encoder.preferred.tolist(),
            "sigma": fly.encoder.sigma.tolist(),
        },
        "sides": sides,
        "samples": [
            {"state": state.tolist(), "values_trained": fly.values(state).tolist(), "values_naive": naive_values(fly, state)}
            for state in states
        ],
        "physics": physics_check(seed),
    }


def scene_positions(somas: dict[str, np.ndarray]) -> dict[str, list]:
    """Centre and scale soma coordinates into [-1, 1]; missing somas become None; dorsal is up."""
    known = np.concatenate([soma[np.isfinite(soma).all(axis=1)] for soma in somas.values()])
    centre = known.mean(axis=0)
    scale = np.abs(known - centre).max()
    flip = np.array([1.0, -1.0, 1.0])
    return {
        name: [np.round((row - centre) / scale * flip, 4).tolist() if np.isfinite(row).all() else None for row in soma]
        for name, soma in somas.items()
    }


def export_cells(left_circuit_path: Path, right_circuit_path: Path) -> dict:
    somas = {}
    with np.load(left_circuit_path) as left, np.load(right_circuit_path) as right:
        for side, arrays in (("left", left), ("right", right)):
            for name in POPULATIONS:
                somas[f"{side}_{name}"] = arrays[f"{name}_soma"]
        somas["dan"] = right["dan_soma"]
        punishment = right["dan_is_punishment"].astype(bool).tolist()
    positions = scene_positions(somas)
    return {
        "left": {name: positions[f"left_{name}"] for name in POPULATIONS},
        "right": {name: positions[f"right_{name}"] for name in POPULATIONS},
        "dan": positions["dan"],
        "dan_is_punishment": punishment,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: the browser's JSON.parse rejects NaN.
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))


def export(seed: int, episodes: int, params: Hyperparameters, circuit_path: Path = CIRCUIT_PATH,
           left_circuit_path: Path = LEFT_CIRCUIT_PATH, web_data_dir: Path = WEB_DATA_DIR) -> dict:
    fly, schedule = make_agent(CONDITION, load_circuit(circuit_path), params, seed, load_circuit(left_circuit_path))
    lengths = run_episodes(fly, schedule, episodes, seed)
    write_json(web_data_dir / "model.json", export_model(fly, schedule.baseline, seed))
    write_json(web_data_dir / "cells.json", export_cells(left_circuit_path, circuit_path))
    return {
        "seed": seed,
        "training_episodes": episodes,
        "final_100_mean": float(np.mean(lengths[-100:])),
        "longest": max(lengths),
    }
