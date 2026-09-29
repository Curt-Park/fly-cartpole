import json

import numpy as np

from fly_cartpole.circuit import load_circuit
from fly_cartpole.params import Hyperparameters, load_hyperparameters
from fly_cartpole.tune import RPE_SPACE, final_score, option_overlap, sample_configs, tune

PARAMS = Hyperparameters(kc_sparsity=0.1)


def test_sample_configs_are_distinct_and_capped_by_the_grid():
    configs = sample_configs({"alpha": (1, 2), "beta": (3,)}, count=10, rng=np.random.default_rng(0))
    assert sorted(config["alpha"] for config in configs) == [1, 2]


def test_final_score_uses_the_last_window():
    assert final_score([1] * 50 + [100] * 100) == 100.0


def test_option_overlap_is_a_fraction(circuit_path):
    overlap = option_overlap(load_circuit(circuit_path), PARAMS, seed=0, samples=20)
    assert 0.0 <= overlap <= 1.0


def test_tune_writes_hyperparameters_and_the_search_table(circuit_path, tmp_path):
    best = tune(configs=2, episodes=3, workers=1, circuit_path=circuit_path, results_dir=tmp_path, base=PARAMS,
                left_circuit_path=circuit_path, bilateral_episodes=3)
    assert load_hyperparameters(tmp_path / "hyperparameters.json") == best
    table = json.loads((tmp_path / "tuning.json").read_text())
    assert len(table["fly"]) == 2
    assert len(table["fly-rpe"]) == len(RPE_SPACE["rpe_learning_rate"]) * len(RPE_SPACE["rpe_trace_decay"])
    assert len(table["fly-bilateral"]) >= 1
    assert len(table["td"]) == 6
    assert best.bilateral_learning_rate == table["fly-bilateral"][0]["config"]["bilateral_learning_rate"]
    assert 0.0 <= table["left_right_kc_overlap"] <= 1.0


def test_tuning_never_changes_the_task_reward(circuit_path, tmp_path):
    best = tune(configs=2, episodes=3, workers=1, circuit_path=circuit_path, results_dir=tmp_path, base=PARAMS,
                left_circuit_path=circuit_path, bilateral_episodes=3)
    assert best.reward_per_step == PARAMS.reward_per_step
