"""Episode loop, experimental conditions and per-seed result files."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from .bilateral import BilateralFly, build_bilateral
from .circuit import Circuit, load_circuit, shuffle_circuit
from .dopamine import DopamineSchedule
from .encoder import build_encoder
from .fly_agent import Decision, FlyAgent, PredictionErrorFly, WiredPredictionErrorFly
from .mushroom_body import MushroomBody
from .params import Hyperparameters
from .paths import CIRCUIT_PATH, LEFT_CIRCUIT_PATH, RESULTS_DIR
from .random_agent import RandomAgent
from .td_agent import TDAgent

CONDITIONS = ("fly", "fly-best", "fly-shuffled", "fly-frozen", "fly-rpe", "fly-rpe-wired",
              "fly-bilateral", "fly-bilateral-shuffled", "fly-bilateral-frozen", "td", "random")
BILATERAL = ("fly-bilateral", "fly-bilateral-shuffled", "fly-bilateral-frozen")
PREDICTION_ERROR_AGENTS = {"fly-rpe": PredictionErrorFly, "fly-rpe-wired": WiredPredictionErrorFly}
StepCallback = Callable[[int, int, np.ndarray, Decision, float, float, "float | None"], None]


def make_agent(condition: str, circuit: Circuit, params: Hyperparameters, seed: int,
               left_circuit: Circuit | None = None) -> tuple[FlyAgent | TDAgent | RandomAgent | BilateralFly, DopamineSchedule]:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    rng = np.random.default_rng(seed)
    if condition in BILATERAL:
        left_circuit = left_circuit if left_circuit is not None else load_circuit(LEFT_CIRCUIT_PATH)
        if condition == "fly-bilateral-shuffled":
            circuit, left_circuit = shuffle_circuit(circuit, rng), shuffle_circuit(left_circuit, rng)
        if condition == "fly-bilateral-frozen":
            params = replace(params, bilateral_learning_rate=0.0)
        schedule = DopamineSchedule("mean", params.baseline_window, params.reward_per_step, params.posture_weight, params.gamma)
        return build_bilateral(left_circuit, circuit, params, seed), schedule
    if condition == "fly-shuffled":
        circuit = shuffle_circuit(circuit, rng)
    if condition == "fly-frozen":
        params = replace(params, learning_rate=0.0)
    if condition in PREDICTION_ERROR_AGENTS:
        params = replace(params, learning_rate=params.rpe_learning_rate, trace_decay=params.rpe_trace_decay)
    body = MushroomBody(circuit, params.kc_sparsity)
    # Same seed, same glomerulus assignment: conditions are paired per seed.
    encoder = build_encoder(circuit.pn_glomerulus, params.action_fraction, params.tuning_width, seed)
    schedule = DopamineSchedule("best" if condition == "fly-best" else "mean", params.baseline_window, params.reward_per_step,
                                params.posture_weight, params.gamma)
    if condition == "random":
        return RandomAgent(encoder, body.n_kc, rng), schedule
    agent_class = PREDICTION_ERROR_AGENTS.get(condition, TDAgent if condition == "td" else FlyAgent)
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
            reward += schedule.posture(state, next_state)
            agent.learn(punish, reward, next_state, terminated)
            if on_step is not None:
                on_step(episode, step, state, decision, *agent.released, schedule.baseline)
            state = next_state
            if terminated or truncated:
                break
        schedule.end_episode(step)
        lengths.append(step)
    env.close()
    return lengths


def run_condition(condition: str, seed: int, episodes: int, params: Hyperparameters, circuit_path: Path = CIRCUIT_PATH,
                  left_circuit_path: Path = LEFT_CIRCUIT_PATH) -> list[int]:
    left_circuit = load_circuit(left_circuit_path) if condition in BILATERAL else None
    agent, schedule = make_agent(condition, load_circuit(circuit_path), params, seed, left_circuit)
    return run_episodes(agent, schedule, episodes, seed)


def run_many(jobs: list[tuple[str, int, Hyperparameters]], episodes: int, workers: int, circuit_path: Path = CIRCUIT_PATH,
             left_circuit_path: Path = LEFT_CIRCUIT_PATH) -> list[list[int]]:
    if workers <= 1:
        return [run_condition(condition, seed, episodes, params, circuit_path, left_circuit_path) for condition, seed, params in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_condition, condition, seed, episodes, params, circuit_path, left_circuit_path)
                   for condition, seed, params in jobs]
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
