"""Compare conditions: learning curves, a summary table and permutation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .params import Hyperparameters
from .paths import CIRCUIT_PATH, LEFT_CIRCUIT_PATH, RESULTS_DIR
from .run import CONDITIONS, run_many, save_lengths

FINAL_WINDOW = 100
# Validated categorical slots in fixed order; bilateral controls reuse their parent's hue with a dash pattern.
COLOURS = {
    "fly": "#2a78d6", "fly-best": "#eb6834", "fly-shuffled": "#1baf7a", "fly-frozen": "#eda100", "td": "#e87ba4",
    "fly-rpe": "#008300", "fly-rpe-wired": "#4a3aa7", "fly-bilateral": "#e34948",
    "fly-bilateral-shuffled": "#e34948", "fly-bilateral-frozen": "#e34948",
}
DASHES = {"fly-bilateral-shuffled": "--", "fly-bilateral-frozen": ":"}
# The random policy is a chance reference, drawn in neutral ink rather than as another series.
REFERENCE_COLOUR = "#898781"
PANELS = (
    ("Which hypothesis learns?", ("fly", "fly-rpe", "fly-rpe-wired", "fly-bilateral", "td", "random"), 510),
    ("Bilateral fly and its controls", ("fly-bilateral", "fly-bilateral-shuffled", "fly-bilateral-frozen", "random"), 510),
    ("Original fly and its controls (zoomed)", ("fly", "fly-best", "fly-shuffled", "fly-frozen", "random"), 40),
)
CLAIMS = (
    ("learning", "fly", "fly-frozen", "greater"),
    ("wiring contributes", "fly", "fly-shuffled", "greater"),
    ("baseline mode", "fly", "fly-best", "two-sided"),
    ("prediction error helps", "fly-rpe", "fly", "greater"),
    ("measured feedback wiring works", "fly-rpe-wired", "fly", "greater"),
    ("bilateral fly learns", "fly-bilateral", "fly-bilateral-frozen", "greater"),
    ("bilateral fly beats chance", "fly-bilateral", "random", "greater"),
    ("bilateral wiring contributes", "fly-bilateral", "fly-bilateral-shuffled", "greater"),
)


def moving_average(values: np.ndarray, window: int = FINAL_WINDOW) -> np.ndarray:
    """Trailing mean; early points average whatever is available."""
    totals = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    return (totals[ends] - totals[starts]) / (ends - starts)


def permutation_test(first: np.ndarray, second: np.ndarray, rng: np.random.Generator,
                     permutations: int = 10_000, alternative: str = "greater") -> float:
    pooled = np.concatenate([first, second])
    observed = first.mean() - second.mean()
    hits = 0
    for _ in range(permutations):
        shuffled = rng.permutation(pooled)
        difference = shuffled[: len(first)].mean() - shuffled[len(first) :].mean()
        hits += difference >= observed if alternative == "greater" else abs(difference) >= abs(observed)
    return (hits + 1) / (permutations + 1)


def final_means(by_seed: dict[int, list[int]]) -> np.ndarray:
    return np.array([np.mean(lengths[-FINAL_WINDOW:]) for lengths in by_seed.values()])


def plot_curves(results: dict[str, dict[int, list[int]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(PANELS), figsize=(17, 5.6), facecolor="#fcfcfb")
    for axis, (title, conditions, ceiling) in zip(axes, PANELS):
        axis.set_facecolor("#fcfcfb")
        for condition in (name for name in conditions if name in results):
            curves = np.array([moving_average(np.array(lengths)) for lengths in results[condition].values()])
            mean, spread = curves.mean(axis=0), curves.std(axis=0)
            episodes = np.arange(1, curves.shape[1] + 1)
            if condition == "random":
                axis.plot(episodes, mean, color=REFERENCE_COLOUR, linewidth=1.5, linestyle="--", label="random (chance)")
                continue
            if condition not in DASHES:
                axis.fill_between(episodes, mean - spread, mean + spread, color=COLOURS[condition], alpha=0.12, linewidth=0)
            axis.plot(episodes, mean, color=COLOURS[condition], linewidth=2, linestyle=DASHES.get(condition, "-"), label=condition)
        axis.set_ylim(0, ceiling)
        if ceiling > 500:
            axis.axhline(500, color="#c3c2b7", linewidth=1, linestyle=":")
        axis.set_title(title, color="#0b0b0b", loc="left", fontsize=11)
        axis.set_xlabel("episode", color="#52514e")
        axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        axis.tick_params(colors="#898781")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#c3c2b7")
        axis.legend(frameon=False, labelcolor="#0b0b0b", fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    axes[0].set_ylabel(f"episode length, {FINAL_WINDOW}-episode moving average", color="#52514e")
    figure.suptitle("Mushroom body learning CartPole (mean ± std over seeds)", color="#0b0b0b", x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def summarise(results: dict[str, dict[int, list[int]]], rng: np.random.Generator) -> str:
    lines = [
        "# Results",
        "",
        f"Mean length of the final {FINAL_WINDOW} episodes, per seed. 500 is the CartPole-v1 ceiling.",
        "",
        "| condition | mean ± std | episodes reaching 500 | per seed |",
        "|---|---|---|---|",
    ]
    for condition, by_seed in results.items():
        means = final_means(by_seed)
        reached = np.mean([np.mean(np.array(lengths[-FINAL_WINDOW:]) >= 500) for lengths in by_seed.values()])
        per_seed = ", ".join(f"{value:.0f}" for value in means)
        lines.append(f"| {condition} | {means.mean():.1f} ± {means.std():.1f} | {reached:.0%} | {per_seed} |")
    lines += ["", "| claim | comparison | permutation p |", "|---|---|---|"]
    for claim, first, second, alternative in (claim for claim in CLAIMS if claim[1] in results and claim[2] in results):
        p_value = permutation_test(final_means(results[first]), final_means(results[second]), rng, alternative=alternative)
        lines.append(f"| {claim} | {first} vs {second} ({alternative}) | {p_value:.4f} |")
    return "\n".join(lines) + "\n"


def compare(seeds: list[int], episodes: int, params: Hyperparameters, workers: int,
            circuit_path: Path = CIRCUIT_PATH, results_dir: Path = RESULTS_DIR,
            left_circuit_path: Path = LEFT_CIRCUIT_PATH) -> str:
    jobs = [(condition, seed, params) for condition in CONDITIONS for seed in seeds]
    results: dict[str, dict[int, list[int]]] = {condition: {} for condition in CONDITIONS}
    for (condition, seed, _), lengths in zip(jobs, run_many(jobs, episodes, workers, circuit_path, left_circuit_path)):
        save_lengths(condition, seed, lengths, params, results_dir)
        results[condition][seed] = lengths
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(results, results_dir / "learning_curves.png")
    summary = summarise(results, np.random.default_rng(0))
    (results_dir / "summary.md").write_text(summary)
    return summary
