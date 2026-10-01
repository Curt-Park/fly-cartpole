import numpy as np
import pytest

from fly_cartpole.reflex import ReflexFly, load_flight, sensor_drive

HALTERE_LEFT, HALTERE_RIGHT, OCELLUS_LEFT, OCELLUS_RIGHT, HS_LEFT, HS_RIGHT = range(6)
UNIT_GAINS = (1.0, 1.0, 1.0)


def drive_for(flight_path, state, gains=UNIT_GAINS):
    return sensor_drive(load_flight(flight_path), np.array(state, dtype=np.float64), gains)


def test_tilting_right_inhibits_the_left_ocellus_and_excites_the_right(flight_path):
    drive = drive_for(flight_path, [0.0, 0.0, 0.105, 0.0])
    assert drive[OCELLUS_LEFT] == pytest.approx(-0.5) and drive[OCELLUS_RIGHT] == pytest.approx(0.5)


def test_falling_right_excites_the_right_haltere(flight_path):
    drive = drive_for(flight_path, [0.0, 0.0, 0.0, 1.75])
    assert drive[HALTERE_RIGHT] == pytest.approx(0.5) and drive[HALTERE_LEFT] == pytest.approx(-0.5)


def test_drifting_right_excites_the_left_hs_cell(flight_path):
    drive = drive_for(flight_path, [0.0, 1.5, 0.0, 0.0])
    assert drive[HS_LEFT] == pytest.approx(0.5) and drive[HS_RIGHT] == pytest.approx(-0.5)


def test_sensor_drive_is_clipped_to_the_working_range_and_scaled_by_each_gain(flight_path):
    drive = drive_for(flight_path, [0.0, 0.0, 1.0, 0.0], gains=(2.0, 1.0, 1.0))
    assert drive[OCELLUS_RIGHT] == pytest.approx(2.0)
    assert not drive[[HALTERE_LEFT, HALTERE_RIGHT, HS_LEFT, HS_RIGHT]].any()


def test_a_pole_falling_right_makes_the_fly_push_right(flight_path):
    fly = ReflexFly(load_flight(flight_path))
    decision = fly.act(np.array([0.0, 0.0, 0.0, 1.0]))
    assert decision.action == 1 and decision.steer > 0


def test_a_tie_pushes_left(flight_path):
    fly = ReflexFly(load_flight(flight_path))
    assert fly.act(np.zeros(4)).action == 0


def test_activity_resets_each_episode(flight_path):
    fly = ReflexFly(load_flight(flight_path))
    fly.act(np.array([0.0, 0.0, 0.1, 0.5]))
    assert fly.activity.any()
    fly.reset_episode()
    assert not fly.activity.any()


def test_activity_stays_bounded_under_constant_drive(flight_path):
    fly = ReflexFly(load_flight(flight_path))
    for _ in range(1000):
        fly.act(np.array([0.0, 3.0, 0.2, 3.5]))
    assert np.isfinite(fly.activity).all() and np.abs(fly.activity).max() < 10


def test_learning_changes_nothing(flight_path):
    fly = ReflexFly(load_flight(flight_path))
    fly.act(np.array([0.0, 0.0, 0.1, 0.5]))
    before = fly.activity.copy()
    fly.learn(punish=1.0, reward=0.0, next_state=np.zeros(4), terminated=True)
    assert np.array_equal(fly.activity, before) and fly.released == (0.0, 0.0)


def adaptive(flight_path, eta=0.5, sigma=0.2):
    from fly_cartpole.reflex import AdaptiveReflexFly

    return AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=eta, sigma=sigma)


def finish_episode_of(fly, steps, trial):
    fly.trial = np.array(trial, dtype=np.float64)
    for _ in range(steps):
        fly.act(np.zeros(4))
    fly.reset_episode()


def test_the_first_episode_sets_the_baseline_without_changing_the_gains(flight_path):
    fly = adaptive(flight_path)
    finish_episode_of(fly, 12, [0.1, 0.0, 0.0])
    assert fly.lengths == [12] and not fly.log_gains.any()


def test_a_trial_that_beats_the_baseline_pulls_the_gains_toward_it(flight_path):
    fly = adaptive(flight_path, eta=0.5, sigma=0.2)
    fly.lengths = [10, 10]
    finish_episode_of(fly, 30, [0.1, 0.0, 0.0])
    # dopamine (30 - 10) / 10 = 2; step 0.5 * 2 * 0.1 / 0.2 = 0.5
    assert fly.log_gains == pytest.approx([0.5, 0.0, 0.0])


def test_a_trial_that_falls_short_pushes_the_gains_away(flight_path):
    fly = adaptive(flight_path, eta=0.5, sigma=0.2)
    fly.lengths = [10, 10]
    finish_episode_of(fly, 5, [0.0, -0.2, 0.0])
    # dopamine (5 - 10) / 10 = -0.5; step 0.5 * -0.5 * -0.2 / 0.2 = 0.25
    assert fly.log_gains == pytest.approx([0.0, 0.25, 0.0])


def test_the_baseline_is_the_recent_mean_only(flight_path):
    fly = adaptive(flight_path, eta=0.5, sigma=0.2)
    fly.lengths = [1000] + [10] * 20
    finish_episode_of(fly, 10, [0.1, 0.1, 0.1])
    assert not fly.log_gains.any()


def test_every_episode_tries_new_positive_gains_around_the_learned_ones(flight_path):
    fly = adaptive(flight_path)
    fly.log_gains = np.array([1.0, -1.0, 0.0])
    fly.reset_episode()
    assert fly.trial.any()
    assert np.allclose(fly.gains, np.exp(fly.log_gains + fly.trial)) and min(fly.gains) > 0
