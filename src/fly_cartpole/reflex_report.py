"""Run, self-tune and compare the reflex fly on its own seeds, written to results/reflex/."""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from .dopamine import DopamineSchedule
from .paths import FLIGHT_PATH, REFLEX_RESULTS_DIR, STATION_RESULTS_DIR
from .reflex import (SENSES, WIRING_ONLY, AdaptiveReflexFly, FlightCircuit, ReflexDecision, ReflexFly, load_flight,
                     relative_gains, shuffle_flight)
from .report import FINAL_WINDOW, REFERENCE_COLOUR, final_means, moving_average, permutation_test, summarise
from .run import run_episodes, save_lengths

REFLEX_CONDITIONS = ("fly-reflex", "fly-reflex-adaptive", "fly-reflex-adaptive-shuffled", "random")
STATION = "fly-reflex-station"
ADAPTIVE = ("fly-reflex-adaptive", "fly-reflex-adaptive-shuffled", STATION)
STATION_CONDITIONS = ("fly-reflex-adaptive", STATION)
REFLEX_CLAIMS = (
    ("reflex beats chance", "fly-reflex", "random", "greater"),
    ("self-tuning helps", "fly-reflex-adaptive", "fly-reflex", "greater"),
    ("wiring contributes", "fly-reflex-adaptive", "fly-reflex-adaptive-shuffled", "greater"),
)
STATION_CLAIMS = (("position helps within the 500-step cap", STATION, "fly-reflex-adaptive", "greater"),)
COLOURS = {"fly-reflex": "#2a78d6", "fly-reflex-adaptive": "#e34948", "fly-reflex-adaptive-shuffled": "#e34948",
           STATION: "#1baf7a"}
PARAMETERS_FILE = "hyperparameters.json"


@dataclass(frozen=True)
class ReflexParameters:
    eta: float = 0.1
    sigma: float = 0.2
    baseline_window: int = 20
    adapt_episodes: int = 600
    fixed_episodes: int = 100
    leak: float = 0.5
    substeps: int = 4


def save_reflex_parameters(params: ReflexParameters, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(params), indent=2) + "\n")


def load_reflex_parameters(path: Path = REFLEX_RESULTS_DIR / PARAMETERS_FILE) -> ReflexParameters:
    if not path.exists():
        # A silent default would run every comparison with untuned rates.
        raise FileNotFoundError(f"{path} not found: run `fly-cartpole reflex-tune` first")
    known = {field.name for field in fields(ReflexParameters)}
    return ReflexParameters(**{name: value for name, value in json.loads(path.read_text()).items() if name in known})


class RandomPusher:
    """Chance reference: pushes at random and never learns."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.released = (0.0, 0.0)

    def reset_episode(self) -> None:
        pass

    def act(self, state: np.ndarray) -> ReflexDecision:
        return ReflexDecision(int(self.rng.integers(2)), 0.0)

    def learn(self, punish: float, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        pass


def episodes_for(condition: str, params: ReflexParameters) -> int:
    return params.adapt_episodes if condition in ADAPTIVE else params.fixed_episodes


def make_reflex_agent(condition: str, circuit: FlightCircuit, params: ReflexParameters, seed: int):
    if condition not in REFLEX_CONDITIONS + (STATION,):
        raise ValueError(f"condition must be one of {REFLEX_CONDITIONS + (STATION,)}, got {condition!r}")
    rng = np.random.default_rng(seed)
    if condition == "random":
        return RandomPusher(rng)
    if condition == "fly-reflex":
        return ReflexFly(circuit, WIRING_ONLY, params.substeps, params.leak)
    if condition == "fly-reflex-adaptive-shuffled":
        circuit = shuffle_flight(circuit, rng)
    return AdaptiveReflexFly(circuit, rng, params.eta, params.sigma, params.baseline_window, params.substeps, params.leak,
                             station_keeping=condition == STATION)


def run_reflex_condition(condition: str, seed: int, params: ReflexParameters, flight_path: Path = FLIGHT_PATH) -> list[int]:
    agent = make_reflex_agent(condition, load_flight(flight_path), params, seed)
    # Reflex agents ignore dopamine; the schedule only satisfies the shared episode loop.
    return run_episodes(agent, DopamineSchedule("mean", params.baseline_window, 0.0), episodes_for(condition, params), seed)


def run_reflex_many(jobs: list[tuple[str, int, ReflexParameters]], workers: int, flight_path: Path = FLIGHT_PATH) -> list[list[int]]:
    if workers <= 1:
        return [run_reflex_condition(condition, seed, params, flight_path) for condition, seed, params in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_reflex_condition, condition, seed, params, flight_path) for condition, seed, params in jobs]
        return [future.result() for future in futures]


def reflex_tune(seeds: list[int], workers: int, flight_path: Path = FLIGHT_PATH, results_dir: Path = REFLEX_RESULTS_DIR,
                etas: tuple[float, ...] = (0.03, 0.1, 0.3), sigmas: tuple[float, ...] = (0.1, 0.3),
                base: ReflexParameters = ReflexParameters()) -> ReflexParameters:
    configs = [replace(base, eta=eta, sigma=sigma) for eta, sigma in itertools.product(etas, sigmas)]
    jobs = [("fly-reflex-adaptive", seed, config) for config in configs for seed in seeds]
    lengths = run_reflex_many(jobs, workers, flight_path)
    table = []
    for index, config in enumerate(configs):
        per_seed = [float(np.mean(run[-FINAL_WINDOW:])) for run in lengths[index * len(seeds):(index + 1) * len(seeds)]]
        table.append({"eta": config.eta, "sigma": config.sigma, "score": float(np.mean(per_seed)), "per_seed": per_seed})
    table.sort(key=lambda row: row["score"], reverse=True)
    best = replace(base, eta=table[0]["eta"], sigma=table[0]["sigma"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tuning.json").write_text(json.dumps({"seeds": list(seeds), "configs": table}, indent=2) + "\n")
    save_reflex_parameters(best, results_dir / PARAMETERS_FILE)
    return best


def plot_reflex(results: dict[str, dict[int, list[int]]], params: ReflexParameters, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.2), facecolor="#fcfcfb")
    axis.set_facecolor("#fcfcfb")
    episodes = np.arange(1, params.adapt_episodes + 1)
    for condition, by_seed in results.items():
        if condition in ADAPTIVE:
            curves = np.array([moving_average(np.array(lengths)) for lengths in by_seed.values()])
            mean, spread = curves.mean(axis=0), curves.std(axis=0)
            dashed = condition.endswith("shuffled")
            if not dashed:
                axis.fill_between(episodes, mean - spread, mean + spread, color=COLOURS[condition], alpha=0.12, linewidth=0)
            axis.plot(episodes, mean, color=COLOURS[condition], linewidth=2, linestyle="--" if dashed else "-", label=condition)
        else:
            # Without learning, every episode is drawn from the same reflex: a flat reference.
            level = final_means(by_seed).mean()
            colour, label = (REFERENCE_COLOUR, "random (chance)") if condition == "random" else (COLOURS[condition], "fly-reflex (fixed gains)")
            axis.axhline(level, color=colour, linewidth=2 if condition != "random" else 1.5,
                         linestyle="--" if condition == "random" else ":", label=label)
    axis.set_ylim(0, 510)
    axis.axhline(500, color="#c3c2b7", linewidth=1, linestyle=":")
    axis.set_title("Reflex fly balancing CartPole (mean ± std over seeds)", color="#0b0b0b", loc="left", fontsize=11)
    axis.set_xlabel("episode", color="#52514e")
    axis.set_ylabel(f"episode length, {FINAL_WINDOW}-episode moving average", color="#52514e")
    axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axis.tick_params(colors="#898781")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c3c2b7")
    axis.legend(frameon=False, labelcolor="#0b0b0b", fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def reflex_compare(seeds: list[int], workers: int, params: ReflexParameters, flight_path: Path = FLIGHT_PATH,
                   results_dir: Path = REFLEX_RESULTS_DIR) -> str:
    jobs = [(condition, seed, params) for condition in REFLEX_CONDITIONS for seed in seeds]
    results: dict[str, dict[int, list[int]]] = {condition: {} for condition in REFLEX_CONDITIONS}
    for (condition, seed, _), lengths in zip(jobs, run_reflex_many(jobs, workers, flight_path)):
        save_lengths(condition, seed, lengths, params, results_dir)
        results[condition][seed] = lengths
    plot_reflex(results, params, results_dir / "learning_curves.png")
    summary = summarise(results, np.random.default_rng(0), REFLEX_CLAIMS)
    (results_dir / "summary.md").write_text(summary)
    return summary


LONG_EPISODES = 10
LONG_STEPS = 2000


def hold_station(circuit: FlightCircuit, gains: tuple[float, ...], params: ReflexParameters, seed: int,
                 episodes: int = LONG_EPISODES, steps: int = LONG_STEPS) -> dict:
    """Episodes past CartPole-v1's 500-step cap, exploration off: they tell a slowed drift from a held station."""
    env = gym.make("CartPole-v1", max_episode_steps=steps)
    fly = ReflexFly(circuit, tuple(gains), params.substeps, params.leak)
    lengths, exits, offsets = [], 0, []
    # Offset seeds keep these starts apart from the adaptation episodes'.
    state, _ = env.reset(seed=seed + 10_000)
    for episode in range(episodes):
        if episode:
            state, _ = env.reset()
        fly.reset_episode()
        step = 0
        while True:
            state, _, terminated, truncated, _ = env.step(fly.act(state).action)
            step += 1
            offsets.append(abs(float(state[0])))
            if terminated or truncated:
                break
        lengths.append(step)
        exits += bool(terminated and abs(state[0]) > env.unwrapped.x_threshold)
    env.close()
    return {"lengths": lengths, "exits": exits, "mean_offset": float(np.mean(offsets))}


def run_station_condition(condition: str, seed: int, params: ReflexParameters, flight_path: Path = FLIGHT_PATH,
                          long_episodes: int = LONG_EPISODES, long_steps: int = LONG_STEPS) -> dict:
    circuit = load_flight(flight_path)
    fly = make_reflex_agent(condition, circuit, params, seed)
    lengths = run_episodes(fly, DopamineSchedule("mean", params.baseline_window, 0.0), params.adapt_episodes, seed)
    held = hold_station(circuit, fly.tuned_gains(), params, seed, long_episodes, long_steps)
    return {"adaptation": lengths, "gains": relative_gains(fly.log_gains), **held}


def station_compare(seeds: list[int], workers: int, params: ReflexParameters, flight_path: Path = FLIGHT_PATH,
                    results_dir: Path = STATION_RESULTS_DIR, long_episodes: int = LONG_EPISODES,
                    long_steps: int = LONG_STEPS) -> str:
    jobs = [(condition, seed) for condition in STATION_CONDITIONS for seed in seeds]
    if workers <= 1:
        outcomes = [run_station_condition(condition, seed, params, flight_path, long_episodes, long_steps) for condition, seed in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_station_condition, condition, seed, params, flight_path, long_episodes, long_steps)
                       for condition, seed in jobs]
            outcomes = [future.result() for future in futures]
    results: dict[str, dict[int, list[int]]] = {condition: {} for condition in STATION_CONDITIONS}
    held: dict[str, dict[str, dict]] = {condition: {} for condition in STATION_CONDITIONS}
    for (condition, seed), outcome in zip(jobs, outcomes):
        results[condition][seed] = outcome.pop("adaptation")
        save_lengths(condition, seed, results[condition][seed], params, results_dir)
        held[condition][str(seed)] = outcome
    (results_dir / "station.json").write_text(json.dumps(held, indent=1) + "\n")
    plot_reflex(results, params, results_dir / "learning_curves.png")
    summary = summarise(results, np.random.default_rng(0), STATION_CLAIMS) + held_summary(held, long_episodes, long_steps)
    (results_dir / "summary.md").write_text(summary)
    return summary


def held_summary(held: dict[str, dict[str, dict]], long_episodes: int, long_steps: int) -> str:
    lines = [
        "",
        f"## Past the 500-step cap: {long_episodes} episodes of up to {long_steps} steps per seed",
        "",
        "Tuned gains, exploration off. Gains are shown relative to the ocelli; only their ratios steer the fly.",
        "",
        f"| condition | mean length ± std | track exits | mean distance from centre (m) | median gains relative to the ocelli ({', '.join(SENSES)}) |",
        "|---|---|---|---|---|",
    ]
    per_seed = {}
    for condition, by_seed in held.items():
        lengths = np.array([np.mean(outcome["lengths"]) for outcome in by_seed.values()])
        offsets = np.array([outcome["mean_offset"] for outcome in by_seed.values()])
        exits = sum(outcome["exits"] for outcome in by_seed.values()) / (long_episodes * len(by_seed))
        gains = np.median([np.array(outcome["gains"]) / outcome["gains"][0] for outcome in by_seed.values()], axis=0)
        per_seed[condition] = (lengths, offsets)
        lines.append(f"| {condition} | {lengths.mean():.0f} ± {lengths.std():.0f} | {exits:.0%} | "
                     f"{offsets.mean():.2f} ± {offsets.std():.2f} | {', '.join(f'{gain:.2f}' for gain in gains)} |")
    rng = np.random.default_rng(1)
    station_lengths, station_offsets = per_seed[STATION]
    adaptive_lengths, adaptive_offsets = per_seed["fly-reflex-adaptive"]
    lines += [
        "",
        "| claim | comparison | permutation p |",
        "|---|---|---|",
        f"| position holds the pole longer | {STATION} vs fly-reflex-adaptive, long-episode length (greater) | "
        f"{permutation_test(station_lengths, adaptive_lengths, rng):.4f} |",
        f"| position keeps the cart near the centre | fly-reflex-adaptive vs {STATION}, distance from centre (greater) | "
        f"{permutation_test(adaptive_offsets, station_offsets, rng):.4f} |",
    ]
    return "\n".join(lines) + "\n"
