import numpy as np

from fly_cartpole.params import Hyperparameters
from fly_cartpole.report import compare, moving_average, permutation_test


def test_moving_average_is_trailing():
    assert moving_average(np.array([2.0, 4.0, 6.0]), window=2).tolist() == [2.0, 3.0, 5.0]


def test_permutation_test_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    high = 200.0 + rng.normal(size=10)
    low = 20.0 + rng.normal(size=10)
    assert permutation_test(high, low, rng) < 0.01


def test_permutation_test_is_calm_for_identical_groups():
    rng = np.random.default_rng(0)
    values = rng.normal(size=10)
    assert permutation_test(values, values.copy(), rng) > 0.3


def test_compare_writes_curves_summary_and_per_seed_results(circuit_path, tmp_path):
    summary = compare(seeds=[0, 1], episodes=3, params=Hyperparameters(kc_sparsity=0.1), workers=1,
                      circuit_path=circuit_path, results_dir=tmp_path)
    assert (tmp_path / "learning_curves.png").stat().st_size > 0
    assert (tmp_path / "summary.md").read_text() == summary
    assert (tmp_path / "td" / "seed_1.json").exists()
    assert "fly-frozen" in summary and "wiring contributes" in summary
