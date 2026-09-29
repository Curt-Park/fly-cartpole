"""Episode loop, experimental conditions and per-seed result files."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from .circuit import Circuit, load_circuit, shuffle_circuit
from .dopamine import DopamineSchedule
from .encoder import build_encoder
from .fly_agent import Decision, FlyAgent
from .mushroom_body import MushroomBody
from .params import Hyperparameters
from .paths import CIRCUIT_PATH, RESULTS_DIR
from .td_agent import TDAgent

CONDITIONS = ("fly", "fly-best", "fly-shuffled", "fly-frozen", "td")
StepCallback = Callable[[int, int, np.ndarray, Decision, float, float, "float | None"], None]


def make_agent(condition: str, circuit: Circuit, params: Hyperparameters, seed: int) -> tuple[FlyAgent | TDAgent, DopamineSchedule]:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    rng = np.random.default_rng(seed)
    if condition == "fly-shuffled":
        circuit = shuffle_circuit(circuit, rng)
    if condition == "fly-frozen":
        params = replace(params, learning_rate=0.0)
    body = MushroomBody(circuit, params.kc_sparsity)
    # Same seed, same glomerulus assignment: conditions are paired per seed.
    encoder = build_encoder(circuit.pn_glomerulus, params.action_fraction, params.tuning_width, seed)
    schedule = DopamineSchedule("best" if condition == "fly-best" else "mean", params.baseline_window, params.reward_per_step)
    agent_class = TDAgent if condition == "td" else FlyAgent
    return agent_class(body, encoder, params, rng), schedule


def run_episodes(agent, schedule: DopamineSchedule, episodes: int, seed: int, on_step: StepCallback | None = None) -> list[int]:
    env = gym.make("CartPole-v1")
    lengths = []
    state, _ = env.reset(seed=seed)
    for episode in range(episodes):
        if episode:
            state, _ = env.reset()
        agent.reset_episode()
        step = 0
        while True:
            decision = agent.act(state)
            next_state, _, terminated, truncated, _ = env.step(decision.action)
            step += 1
            punish, reward = schedule.signal(step, terminated)
            agent.learn(punish, reward, next_state, terminated)
            if on_step is not None:
                on_step(episode, step, state, decision, punish, reward, schedule.baseline)
            state = next_state
            if terminated or truncated:
                break
        schedule.end_episode(step)
        lengths.append(step)
    env.close()
    return lengths


def run_condition(condition: str, seed: int, episodes: int, params: Hyperparameters, circuit_path: Path = CIRCUIT_PATH) -> list[int]:
    agent, schedule = make_agent(condition, load_circuit(circuit_path), params, seed)
    return run_episodes(agent, schedule, episodes, seed)


def run_many(jobs: list[tuple[str, int, Hyperparameters]], episodes: int, workers: int, circuit_path: Path = CIRCUIT_PATH) -> list[list[int]]:
    if workers <= 1:
        return [run_condition(condition, seed, episodes, params, circuit_path) for condition, seed, params in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_condition, condition, seed, episodes, params, circuit_path) for condition, seed, params in jobs]
        return [future.result() for future in futures]


def save_lengths(condition: str, seed: int, lengths: list[int], params: Hyperparameters, results_dir: Path = RESULTS_DIR) -> Path:
    path = results_dir / condition / f"seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"condition": condition, "seed": seed, "hyperparameters": asdict(params), "lengths": lengths}
    path.write_text(json.dumps(payload) + "\n")
    return path


def load_lengths(condition: str, results_dir: Path = RESULTS_DIR) -> dict[int, list[int]]:
    by_seed = {}
    for path in sorted((results_dir / condition).glob("seed_*.json")):
        stored = json.loads(path.read_text())
        by_seed[stored["seed"]] = stored["lengths"]
    return by_seed
