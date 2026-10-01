import json

import pytest

from fly_cartpole.reflex import load_flight
from fly_cartpole.reflex_report import (REFLEX_CONDITIONS, ReflexParameters, episodes_for, hold_station,
                                        load_reflex_parameters, reflex_compare, reflex_tune, run_reflex_condition,
                                        station_compare)

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


def test_the_station_keeping_fly_runs(flight_path):
    lengths = run_reflex_condition("fly-reflex-station", seed=0, params=QUICK, flight_path=flight_path)
    assert len(lengths) == QUICK.adapt_episodes


def test_holding_station_reports_lengths_exits_and_how_far_the_cart_strays(flight_path):
    held = hold_station(load_flight(flight_path), (1.0, 1.0, 1.0, 1.0), QUICK, seed=0, episodes=2, steps=50)
    assert len(held["lengths"]) == 2 and all(1 <= length <= 50 for length in held["lengths"])
    assert 0 <= held["exits"] <= 2 and held["mean_offset"] >= 0


def test_station_compare_writes_lengths_long_episodes_and_the_claims(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=2, long_steps=50)
    for text in ("position helps", "fly-reflex-station", "track exits"):
        assert text in summary
    stored = json.loads((tmp_path / "station.json").read_text())
    assert set(stored) == {"fly-reflex-adaptive", "fly-reflex-station"} and len(stored["fly-reflex-station"]["1"]["gains"]) == 4
    assert (tmp_path / "fly-reflex-station" / "seed_1.json").exists()
