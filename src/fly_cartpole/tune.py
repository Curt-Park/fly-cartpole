"""Hyperparameter search on tuning seeds, kept apart from evaluation seeds."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .circuit import Circuit, load_circuit
from .encoder import LEFT, RIGHT, STATE_LIMITS
from .params import HYPERPARAMETERS_FILE, Hyperparameters, save_hyperparameters
from .paths import CIRCUIT_PATH, LEFT_CIRCUIT_PATH, RESULTS_DIR
from .run import make_agent, run_many

TUNING_SEEDS = (100, 101, 102, 103, 104)
FLY_SPACE = {
    "trace_decay": (0.3, 0.6, 0.8, 0.95),
    "learning_rate": (0.02, 0.1, 0.5),
    "beta": (3.0, 10.0, 30.0),
    "gain_decay": (1e-4, 1e-3, 1e-2),
    "reward_per_step": (0.02, 0.1, 0.5),
    "action_fraction": (0.1, 0.25),
}
RPE_SPACE = {
    "rpe_learning_rate": (0.02, 0.1, 0.5),
    "rpe_trace_decay": (0.5, 0.8, 0.95),
}
BILATERAL_SPACE = {
    "bilateral_learning_rate": (0.005, 0.01, 0.02),
    "bilateral_trace_decay": (0.3, 0.5),
    "bilateral_beta": (100.0, 300.0),
    "bilateral_gain_decay": (1e-5, 1e-4),
}
TD_SPACE = {
    "td_actor_lr": (0.01, 0.05, 0.2),
    "td_critic_lr": (0.05, 0.2),
}


def final_score(lengths: list[int], window: int = 100) -> float:
    return float(np.mean(lengths[-window:]))


def sample_configs(space: dict[str, tuple], count: int, rng: np.random.Generator) -> list[dict]:
    grid = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    chosen = rng.choice(len(grid), size=min(count, len(grid)), replace=False)
    return [grid[index] for index in sorted(chosen)]


def search(condition: str, base: Hyperparameters, configs: list[dict], episodes: int, workers: int,
           circuit_path: Path, seeds: tuple[int, ...], left_circuit_path: Path = LEFT_CIRCUIT_PATH) -> list[dict]:
    jobs = [(condition, seed, replace(base, **config)) for config in configs for seed in seeds]
    lengths = run_many(jobs, episodes, workers, circuit_path, left_circuit_path)
    rows = []
    for index, config in enumerate(configs):
        per_seed = [final_score(run) for run in lengths[index * len(seeds) : (index + 1) * len(seeds)]]
        rows.append({"config": config, "score": float(np.mean(per_seed)), "per_seed": per_seed})
        print(f"{condition} {config}: {rows[-1]['score']:.1f}")
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def option_overlap(circuit: Circuit, params: Hyperparameters, seed: int, samples: int = 200) -> float:
    """Mean Jaccard overlap of the Kenyon cells recruited by pushing left versus right."""
    agent, _ = make_agent("fly", circuit, params, seed)
    states = np.random.default_rng(seed).uniform(-STATE_LIMITS, STATE_LIMITS, size=(samples, len(STATE_LIMITS))) * 0.5
    overlaps = []
    for state in states:
        left = agent.option(state, LEFT)[1] > 0
        right = agent.option(state, RIGHT)[1] > 0
        overlaps.append((left & right).sum() / max((left | right).sum(), 1))
    return float(np.mean(overlaps))


def full_grid(space: dict[str, tuple]) -> list[dict]:
    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def tune(configs: int, episodes: int, workers: int, circuit_path: Path = CIRCUIT_PATH, results_dir: Path = RESULTS_DIR,
         base: Hyperparameters = Hyperparameters(), seeds: tuple[int, ...] = TUNING_SEEDS,
         left_circuit_path: Path = LEFT_CIRCUIT_PATH, bilateral_episodes: int = 500) -> Hyperparameters:
    rng = np.random.default_rng(0)
    table: dict = {"episodes": episodes, "bilateral_episodes": bilateral_episodes, "seeds": list(seeds)}
    best = base
    # Each search keeps the winners of the searches before it; the bilateral fly needs longer runs to show learning.
    for condition, configs_to_try, length in (
        ("fly", sample_configs(FLY_SPACE, configs, rng), episodes),
        ("fly-rpe", full_grid(RPE_SPACE), episodes),
        ("fly-bilateral", full_grid(BILATERAL_SPACE), bilateral_episodes),
        ("td", full_grid(TD_SPACE), episodes),
    ):
        table[condition] = search(condition, best, configs_to_try, length, workers, circuit_path, seeds, left_circuit_path)
        best = replace(best, **table[condition][0]["config"])
    save_hyperparameters(best, results_dir / HYPERPARAMETERS_FILE)
    table["left_right_kc_overlap"] = option_overlap(load_circuit(circuit_path), best, seeds[0])
    (results_dir / "tuning.json").write_text(json.dumps(table, indent=2) + "\n")
    return best
