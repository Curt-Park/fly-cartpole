"""Hyperparameters shared by the fly, the TD reference and the dopamine schedule."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .paths import RESULTS_DIR

HYPERPARAMETERS_FILE = "hyperparameters.json"


@dataclass(frozen=True)
class Hyperparameters:
    kc_sparsity: float = 0.05
    beta: float = 2.5
    trace_decay: float = 0.95
    learning_rate: float = 0.5
    gain_decay: float = 0.001
    reward_per_step: float = 0.05
    action_fraction: float = 0.25
    tuning_width: float = 1.0
    baseline_window: int = 20
    rpe_learning_rate: float = 0.1
    rpe_trace_decay: float = 0.9
    td_actor_lr: float = 0.01
    td_critic_lr: float = 0.05
    td_trace_decay: float = 0.9
    gamma: float = 0.99


def save_hyperparameters(params: Hyperparameters, path: Path = RESULTS_DIR / HYPERPARAMETERS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(params), indent=2) + "\n")


def load_hyperparameters(path: Path = RESULTS_DIR / HYPERPARAMETERS_FILE) -> Hyperparameters:
    if not path.exists():
        print(f"{path} not found; using defaults (run `fly-cartpole tune` first)", file=sys.stderr)
        return Hyperparameters()
    known = {field.name for field in fields(Hyperparameters)}
    stored = json.loads(path.read_text())
    return Hyperparameters(**{name: value for name, value in stored.items() if name in known})
