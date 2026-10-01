import json

import pytest

from fly_cartpole.reflex_report import (REFLEX_CONDITIONS, ReflexParameters, episodes_for, load_reflex_parameters,
                                        reflex_compare, reflex_tune, run_reflex_condition)

QUICK = ReflexParameters(adapt_episodes=3, fixed_episodes=2)


@pytest.mark.parametrize("condition", REFLEX_CONDITIONS)
def test_every_reflex_condition_runs(flight_path, condition):
    lengths = run_reflex_condition(condition, seed=0, params=QUICK, flight_path=flight_path)
    assert len(lengths) == episodes_for(condition, QUICK)
    assert all(1 <= length <= 500 for length in lengths)


def test_the_fixed_reflex_repeats_exactly(flight_path):
    assert run_reflex_condition("fly-reflex", 3, QUICK, flight_path) == run_reflex_condition("fly-reflex", 3, QUICK, flight_path)


def test_reflex_tune_writes_the_table_and_the_chosen_rates(flight_path, tmp_path):
    best = reflex_tune(seeds=[0], workers=1, flight_path=flight_path, results_dir=tmp_path, etas=(0.1,), sigmas=(0.1, 0.2), base=QUICK)
    table = json.loads((tmp_path / "tuning.json").read_text())
    assert len(table["configs"]) == 2 and best.sigma in (0.1, 0.2)
    assert load_reflex_parameters(tmp_path / "hyperparameters.json") == best


def test_reflex_compare_writes_lengths_a_plot_and_the_claims(flight_path, tmp_path):
    summary = reflex_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path)
    for claim in ("reflex beats chance", "self-tuning helps", "wiring contributes"):
        assert claim in summary
    assert (tmp_path / "learning_curves.png").exists() and (tmp_path / "fly-reflex" / "seed_1.json").exists()
