"""Run, self-tune and compare the reflex fly on its own seeds, written to results/reflex/."""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from .dopamine import DopamineSchedule
from .paths import FLIGHT_PATH, REFLEX_RESULTS_DIR
from .reflex import AdaptiveReflexFly, FlightCircuit, ReflexDecision, ReflexFly, load_flight, shuffle_flight
from .report import FINAL_WINDOW, REFERENCE_COLOUR, final_means, moving_average, summarise
from .run import run_episodes, save_lengths

REFLEX_CONDITIONS = ("fly-reflex", "fly-reflex-adaptive", "fly-reflex-adaptive-shuffled", "random")
ADAPTIVE = ("fly-reflex-adaptive", "fly-reflex-adaptive-shuffled")
REFLEX_CLAIMS = (
    ("reflex beats chance", "fly-reflex", "random", "greater"),
    ("self-tuning helps", "fly-reflex-adaptive", "fly-reflex", "greater"),
    ("wiring contributes", "fly-reflex-adaptive", "fly-reflex-adaptive-shuffled", "greater"),
)
COLOURS = {"fly-reflex": "#2a78d6", "fly-reflex-adaptive": "#e34948", "fly-reflex-adaptive-shuffled": "#e34948"}
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
        return ReflexParameters()
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
    if condition not in REFLEX_CONDITIONS:
        raise ValueError(f"condition must be one of {REFLEX_CONDITIONS}, got {condition!r}")
    rng = np.random.default_rng(seed)
    if condition == "random":
        return RandomPusher(rng)
    if condition == "fly-reflex":
        return ReflexFly(circuit, (1.0, 1.0, 1.0), params.substeps, params.leak)
    if condition == "fly-reflex-adaptive-shuffled":
        circuit = shuffle_flight(circuit, rng)
    return AdaptiveReflexFly(circuit, rng, params.eta, params.sigma, params.baseline_window, params.substeps, params.leak)


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
