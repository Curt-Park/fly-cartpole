"""Record a naive and a trained episode and export them for the web viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .circuit import load_circuit
from .encoder import Encoder
from .fly_agent import Decision
from .params import Hyperparameters
from .paths import CIRCUIT_PATH, WEB_DATA_DIR
from .run import make_agent, run_episodes

FRAME_FIELDS = ("step", "x", "theta", "action", "score_left", "score_right", "glomeruli", "kc", "mbon", "punish", "reward", "baseline")
POPULATIONS = ("pn", "kc", "mbon", "dan")


class EpisodeRecorder:
    """on_step callback that keeps full frames for the selected episodes only."""

    def __init__(self, keep: set[int]) -> None:
        self.keep = keep
        self.episodes: dict[int, dict[str, list]] = {}

    def __call__(self, episode: int, step: int, state: np.ndarray, decision: Decision,
                 punish: float, reward: float, baseline: float | None) -> None:
        if episode not in self.keep:
            return
        frames = self.episodes.setdefault(episode, {name: [] for name in FRAME_FIELDS})
        values = {
            "step": step,
            "x": round(float(state[0]), 4),
            "theta": round(float(state[2]), 4),
            "action": decision.action,
            "score_left": round(decision.scores[0], 6),
            "score_right": round(decision.scores[1], 6),
            "glomeruli": np.round(decision.glomeruli, 3).tolist(),
            "kc": np.flatnonzero(decision.kc).tolist(),
            "mbon": np.round(decision.mbon, 6).tolist(),
            "punish": punish,
            "reward": reward,
            "baseline": None if baseline is None else round(baseline, 2),
        }
        for name, value in values.items():
            frames[name].append(value)


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


def export_cells(circuit_path: Path, encoder: Encoder) -> dict:
    with np.load(circuit_path) as arrays:
        somas = {name: arrays[f"{name}_soma"] for name in POPULATIONS}
        valence = arrays["mbon_valence"].tolist()
        punishment = arrays["dan_is_punishment"].astype(bool).tolist()
    return {
        **scene_positions(somas),
        "pn_group": encoder.pn_group.tolist(),
        "group_role": encoder.group_role.tolist(),
        "mbon_valence": valence,
        "dan_is_punishment": punishment,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: the browser's JSON.parse rejects NaN.
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))


def record(seed: int, episodes: int, params: Hyperparameters, circuit_path: Path = CIRCUIT_PATH,
           web_data_dir: Path = WEB_DATA_DIR) -> dict:
    agent, schedule = make_agent("fly", load_circuit(circuit_path), params, seed)
    recorder = EpisodeRecorder(keep={0, episodes})
    lengths = run_episodes(agent, schedule, episodes + 1, seed, on_step=recorder)
    payload = {
        "seed": seed,
        "training_episodes": episodes,
        "naive": recorder.episodes[0],
        "trained": recorder.episodes[episodes],
    }
    write_json(web_data_dir / "episodes.json", payload)
    write_json(web_data_dir / "cells.json", export_cells(circuit_path, agent.encoder))
    return {
        "seed": seed,
        "naive_length": lengths[0],
        "trained_length": lengths[-1],
        "final_100_mean": float(np.mean(lengths[-101:-1])),
    }
