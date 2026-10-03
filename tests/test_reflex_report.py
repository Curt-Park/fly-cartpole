import json

import numpy as np
import pytest

from fly_cartpole.reflex import load_flight
from fly_cartpole.reflex_report import (REFLEX_CONDITIONS, ReflexParameters, episodes_for, hold_station,
                                        load_reflex_parameters, make_reflex_agent, reflex_compare, reflex_tune,
                                        run_reflex_condition,
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
    held = hold_station(load_flight(flight_path), (1.0, 1.0, 1.0, 1.0, 0.0), QUICK, seed=0, episodes=2, steps=50)
    assert len(held["lengths"]) == 2 and all(1 <= length <= 50 for length in held["lengths"])
    assert 0 <= held["exits"] <= 2 and held["mean_offset"] >= 0


def test_station_compare_writes_lengths_long_episodes_and_the_claims(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=2, long_steps=50)
    for text in ("position helps within the 500-step cap", "fly-reflex-station", "track exits"):
        assert text in summary
    stored = json.loads((tmp_path / "station.json").read_text())
    assert set(stored) == {"fly-reflex-station-fixed", "fly-reflex-adaptive", "fly-reflex-station"}
    assert len(stored["fly-reflex-station"]["1"]["gains"]) == 5
    assert stored["fly-reflex-station-fixed"]["1"]["gains"] == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert (tmp_path / "fly-reflex-station" / "seed_1.json").exists()


def test_a_missing_parameter_file_is_an_error_not_a_silent_default(tmp_path):
    with pytest.raises(FileNotFoundError, match="reflex-tune"):
        load_reflex_parameters(tmp_path / "hyperparameters.json")


def test_a_landmark_pointing_the_wrong_way_drives_the_cart_off_the_track(flight_path):
    circuit = load_flight(flight_path)
    corrective = hold_station(circuit, (1.0, 1.0, 1.0, 0.3, 0.0), QUICK, seed=0, episodes=5, steps=2000)
    reversed_sign = hold_station(circuit, (1.0, 1.0, 1.0, -0.3, 0.0), QUICK, seed=0, episodes=5, steps=2000)
    assert reversed_sign["exits"] == 5 and corrective["exits"] < reversed_sign["exits"]


def test_the_long_episode_claims_favour_the_fly_that_holds_station():
    from fly_cartpole.reflex_report import held_summary

    def outcome(length, offset, gains):
        return {"lengths": [length], "exits": int(length < 2000), "mean_offset": offset, "gains": gains}

    held = {
        "fly-reflex-adaptive": {str(seed): outcome(800 + seed, 0.9, [2.0, 0.5, 1.0, 0.0]) for seed in range(6)},
        "fly-reflex-station": {str(seed): outcome(2000, 0.1 + seed / 100, [4.0, 1.0, 1.0, 0.25]) for seed in range(6)},
    }
    lines = held_summary(held, long_episodes=1, long_steps=2000).splitlines()
    claims = {line.split(" | ")[0].lstrip("| "): float(line.split(" | ")[-1].rstrip(" |")) for line in lines if line.startswith("| position")}
    assert claims["position holds the pole longer"] < 0.05 and claims["position keeps the cart near the centre"] < 0.05
    # Gains are shown relative to the ocelli, so rows that tune different senses stay comparable.
    assert any("1.00, 0.25, 0.50, 0.00" in line for line in lines) and any("1.00, 0.25, 0.25, 0.06" in line for line in lines)


def test_the_fixed_landmark_fly_runs_without_adapting(flight_path):
    lengths = run_reflex_condition("fly-reflex-station-fixed", seed=0, params=QUICK, flight_path=flight_path)
    assert len(lengths) == QUICK.fixed_episodes


def test_station_compare_adds_conditions_to_earlier_results(flight_path, tmp_path):
    options = dict(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path, long_episodes=2, long_steps=50)
    station_compare(conditions=("fly-reflex-adaptive", "fly-reflex-station"), **options)
    summary = station_compare(conditions=("fly-reflex-station-fixed",), **options)
    assert set(json.loads((tmp_path / "station.json").read_text())) == {"fly-reflex-adaptive", "fly-reflex-station", "fly-reflex-station-fixed"}
    for text in ("fly-reflex-adaptive", "fly-reflex-station-fixed", "self-tuning helps with a landmark"):
        assert text in summary


def test_the_centred_landmark_fly_runs(flight_path):
    lengths = run_reflex_condition("fly-reflex-station-centred", seed=0, params=QUICK, flight_path=flight_path)
    assert len(lengths) == QUICK.adapt_episodes


def test_station_compare_tests_the_centred_record_against_the_length_record(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=2, long_steps=50, conditions=("fly-reflex-station", "fly-reflex-station-centred"))
    for text in ("a centred record keeps the cart nearer the centre", "a centred record changes the 500-step score"):
        assert text in summary


def test_the_balanced_landmark_fly_runs(flight_path):
    lengths = run_reflex_condition("fly-reflex-station-balanced", seed=0, params=QUICK, flight_path=flight_path)
    assert len(lengths) == QUICK.adapt_episodes


def test_station_compare_tests_the_balanced_record_against_the_length_record(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=2, long_steps=50, conditions=("fly-reflex-station", "fly-reflex-station-balanced"))
    for text in ("a balanced record keeps the cart nearer the centre", "a balanced record changes the 500-step score"):
        assert text in summary


def test_each_finished_seed_is_saved_before_the_comparison_ends(flight_path, tmp_path, monkeypatch):
    from fly_cartpole import reflex_report

    real_run = reflex_report.run_station_condition
    calls = []

    def run_then_fail(*args, **kwargs):
        calls.append(args[:2])
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(reflex_report, "run_station_condition", run_then_fail)
    with pytest.raises(RuntimeError):
        station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                        long_episodes=2, long_steps=50, conditions=("fly-reflex-station",))
    assert (tmp_path / "fly-reflex-station" / "seed_0.json").exists()
    assert set(json.loads((tmp_path / "station.json").read_text())["fly-reflex-station"]) == {"0"}


def test_the_landmark_motion_fly_runs(flight_path):
    lengths = run_reflex_condition("fly-reflex-station-motion", seed=0, params=QUICK, flight_path=flight_path)
    assert len(lengths) == QUICK.adapt_episodes


def test_station_compare_tests_the_motion_term_against_position_alone(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=2, long_steps=50, conditions=("fly-reflex-station-balanced", "fly-reflex-station-motion"))
    for text in ("a motion term keeps the cart nearer the centre", "a motion term changes the 500-step score"):
        assert text in summary


@pytest.mark.parametrize("condition", ["fly-reflex-station-staged", "fly-reflex-station-staged-motion"])
def test_the_staged_landmark_flies_run(flight_path, condition):
    lengths = run_reflex_condition(condition, seed=0, params=QUICK, flight_path=flight_path)
    assert len(lengths) == QUICK.adapt_episodes


def test_station_compare_tests_the_staged_motion_fly_against_the_reference(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=2, long_steps=50, conditions=("fly-reflex-station-staged-motion", "fly-reflex-station"))
    for text in ("curriculum learning keeps the cart nearer the centre", "curriculum learning holds the pole longer",
                 "curriculum learning changes the 500-step score"):
        assert text in summary


def test_the_linear_conditions_run_the_circuit_s_memoryless_map(flight_path):
    circuit = load_flight(flight_path)
    for condition in ("linear", "linear-adaptive", "linear-station-staged-motion"):
        assert make_reflex_agent(condition, circuit, QUICK, seed=0).memoryless
    assert not make_reflex_agent("fly-reflex", circuit, QUICK, seed=0).memoryless
    staged = make_reflex_agent("linear-station-staged-motion", circuit, QUICK, seed=0)
    assert staged.curriculum == QUICK.adapt_episodes // 2 and staged.adapted == 5


def test_long_episodes_keep_the_linear_map(flight_path, monkeypatch):
    import fly_cartpole.reflex_report as report

    built = []
    original = report.ReflexFly

    def spy(*args, **kwargs):
        built.append(kwargs.get("memoryless", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(report, "ReflexFly", spy)
    report.run_station_condition("linear-station-staged-motion", 0, QUICK, flight_path, long_episodes=1, long_steps=20)
    report.run_station_condition("fly-reflex-station-staged-motion", 0, QUICK, flight_path, long_episodes=1, long_steps=20)
    assert built == [True, False]


def test_reflex_compare_tests_the_circuit_against_its_linear_map(flight_path, tmp_path):
    summary = reflex_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                             conditions=("fly-reflex", "fly-reflex-adaptive", "linear", "linear-adaptive"))
    for claim in ("the untuned circuit differs from its linear map", "the self-tuned circuit differs from its linear map"):
        assert claim in summary
    assert "wiring contributes" not in summary


def test_station_compare_tests_the_final_fly_against_its_linear_map(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=2, long_steps=50,
                              conditions=("fly-reflex-station-staged-motion", "linear-station-staged-motion"))
    for text in ("curriculum learning: the circuit differs from its linear map",
                 "the circuit's distance from the centre differs from its linear map's",
                 "the circuit's long-episode length differs from its linear map's"):
        assert text in summary


def test_reflex_compare_can_compare_with_conditions_saved_elsewhere(flight_path, tmp_path):
    reflex_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path / "fly",
                   conditions=("fly-reflex",))
    summary = reflex_compare(seeds=[2, 3], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path / "linear",
                             conditions=("linear",), references=(tmp_path / "fly",))
    assert "the untuned circuit differs from its linear map" in summary
    # References are read where they were saved, not copied.
    assert not (tmp_path / "linear" / "fly-reflex").exists()


def test_station_compare_can_compare_with_conditions_saved_elsewhere(flight_path, tmp_path):
    options = dict(workers=1, params=QUICK, flight_path=flight_path, long_episodes=2, long_steps=50)
    station_compare(seeds=[0, 1], results_dir=tmp_path / "fly", conditions=("fly-reflex-station-staged-motion",), **options)
    summary = station_compare(seeds=[2, 3], results_dir=tmp_path / "linear", conditions=("linear-station-staged-motion",),
                              references=(tmp_path / "fly",), **options)
    assert "the circuit's distance from the centre differs from its linear map's" in summary
    saved = json.loads((tmp_path / "linear" / "station.json").read_text())
    assert set(saved) == {"linear-station-staged-motion"}


def test_lqr_gains_stabilise_cartpole_linearised_about_the_upright_pole():
    from fly_cartpole.reflex_report import cartpole_linearisation, lqr_gains

    transition, push = cartpole_linearisation()
    gains = lqr_gains(transition, push)
    assert np.abs(np.linalg.eigvals(transition - push @ gains[None, :])).max() < 1


def test_lqr_pushes_toward_a_lean_and_banks_toward_the_centre(flight_path):
    lqr = make_reflex_agent("lqr", load_flight(flight_path), QUICK, seed=0)
    assert lqr.act(np.array([0.0, 0.0, 0.05, 0.0])).action == 1
    # Like the landmark fly, it first pushes away from the centre so the pole leans back toward it.
    assert lqr.act(np.array([1.0, 0.0, 0.0, 0.0])).action == 1
    assert episodes_for("lqr", QUICK) == QUICK.fixed_episodes


def test_long_episodes_run_the_lqr_itself(flight_path, monkeypatch):
    import fly_cartpole.reflex_report as report

    built = []
    original = report.ReflexFly
    monkeypatch.setattr(report, "ReflexFly", lambda *args, **kwargs: built.append(kwargs) or original(*args, **kwargs))
    outcome = report.run_station_condition("lqr", 0, QUICK, flight_path, long_episodes=1, long_steps=200)
    assert built == [] and outcome["lengths"] == [200] and outcome["gains"] == []


def test_station_compare_reports_the_lqr_without_gains(flight_path, tmp_path):
    summary = station_compare(seeds=[0, 1], workers=1, params=QUICK, flight_path=flight_path, results_dir=tmp_path,
                              long_episodes=1, long_steps=50, conditions=("lqr",))
    assert "| lqr |" in summary and "| – |" in summary


def test_reflex_tune_can_tune_another_condition(flight_path, tmp_path):
    reflex_tune(seeds=[0], workers=1, flight_path=flight_path, results_dir=tmp_path, etas=(0.1,), sigmas=(0.1,), base=QUICK,
                condition="linear-adaptive")
    assert json.loads((tmp_path / "tuning.json").read_text())["condition"] == "linear-adaptive"
