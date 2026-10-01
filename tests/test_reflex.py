import numpy as np
import pytest

from fly_cartpole.reflex import ReflexFly, load_flight, sensor_drive

HALTERE_LEFT, HALTERE_RIGHT, OCELLUS_LEFT, OCELLUS_RIGHT, HS_LEFT, HS_RIGHT = range(6)
UNIT_GAINS = (1.0, 1.0, 1.0, 0.0, 0.0)


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
    drive = drive_for(flight_path, [0.0, 0.0, 1.0, 0.0], gains=(2.0, 1.0, 1.0, 0.0, 0.0))
    assert drive[OCELLUS_RIGHT] == pytest.approx(2.0)
    assert not drive[[HALTERE_LEFT, HALTERE_RIGHT, HS_LEFT, HS_RIGHT]].any()


def test_a_cart_right_of_centre_reads_to_the_ocelli_like_a_right_tilt(flight_path):
    # To slide back left the fly must first bank left, so it holds the pole as if it leaned right.
    drive = drive_for(flight_path, [1.2, 0.0, 0.0, 0.0], gains=(1.0, 1.0, 1.0, 1.0, 0.0))
    assert drive[OCELLUS_LEFT] == pytest.approx(-0.5) and drive[OCELLUS_RIGHT] == pytest.approx(0.5)
    assert not drive[[HALTERE_LEFT, HALTERE_RIGHT, HS_LEFT, HS_RIGHT]].any()


def test_a_landmark_sliding_left_reads_to_the_ocelli_like_a_right_tilt(flight_path):
    # A cart moving right sees the landmark slide left; the fly banks against the slide to damp it.
    drive = drive_for(flight_path, [0.0, 1.5, 0.0, 0.0], gains=(1.0, 1.0, 0.0, 0.0, 1.0))
    assert drive[OCELLUS_LEFT] == pytest.approx(-0.5) and drive[OCELLUS_RIGHT] == pytest.approx(0.5)
    assert not drive[[HALTERE_LEFT, HALTERE_RIGHT, HS_LEFT, HS_RIGHT]].any()


def test_without_a_position_gain_the_ocelli_ignore_where_the_cart_is(flight_path):
    assert not drive_for(flight_path, [1.2, 0.0, 0.0, 0.0]).any()


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
    assert fly.records == [12] and not fly.log_gains.any()


def test_a_trial_that_beats_the_baseline_pulls_the_gains_toward_it(flight_path):
    fly = adaptive(flight_path, eta=0.5, sigma=0.2)
    fly.records = [10, 10]
    finish_episode_of(fly, 30, [0.1, 0.0, 0.0])
    # dopamine (30 - 10) / 10 = 2; step 0.5 * 2 * 0.1 / 0.2 = 0.5
    assert fly.log_gains == pytest.approx([0.5, 0.0, 0.0])


def test_a_trial_that_falls_short_pushes_the_gains_away(flight_path):
    fly = adaptive(flight_path, eta=0.5, sigma=0.2)
    fly.records = [10, 10]
    finish_episode_of(fly, 5, [0.0, -0.2, 0.0])
    # dopamine (5 - 10) / 10 = -0.5; step 0.5 * -0.5 * -0.2 / 0.2 = 0.25
    assert fly.log_gains == pytest.approx([0.0, 0.25, 0.0])


def test_the_baseline_is_the_recent_mean_only(flight_path):
    fly = adaptive(flight_path, eta=0.5, sigma=0.2)
    fly.records = [1000] + [10] * 20
    finish_episode_of(fly, 10, [0.1, 0.1, 0.1])
    assert not fly.log_gains.any()


def test_every_episode_tries_new_positive_gains_around_the_learned_ones(flight_path):
    fly = adaptive(flight_path)
    fly.log_gains = np.array([1.0, -1.0, 0.0])
    fly.reset_episode()
    assert fly.trial.any()
    assert np.allclose(fly.gains[:3], np.exp(fly.log_gains + fly.trial)) and min(fly.gains[:3]) > 0
    assert fly.gains[3:] == (0.0, 0.0)


def test_without_station_keeping_the_trials_draw_as_before(flight_path):
    fly = adaptive(flight_path, sigma=0.2)
    assert np.array_equal(fly.trial, np.random.default_rng(0).normal(0.0, 0.2, 3))


def test_a_station_keeping_fly_tunes_its_position_gain_too(flight_path):
    from fly_cartpole.reflex import AdaptiveReflexFly

    fly = AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=0.5, sigma=0.2, landmark="position")
    assert len(fly.gains) == 5 and fly.gains[3] > 0 and fly.gains[4] == 0.0
    fly.records = [10, 10]
    finish_episode_of(fly, 30, [0.0, 0.0, 0.0, 0.1])
    assert fly.log_gains == pytest.approx([0.0, 0.0, 0.0, 0.5])


def test_tuned_gains_leave_out_the_trial(flight_path):
    fly = adaptive(flight_path)
    fly.log_gains = np.array([1.0, -1.0, 0.5])
    fly.reset_episode()
    assert fly.tuned_gains() == pytest.approx((np.e, np.exp(-1.0), np.exp(0.5), 0.0, 0.0))


def test_a_shuffled_flight_circuit_keeps_roles_degrees_and_couplings(flight_path):
    from fly_cartpole.reflex import shuffle_flight

    circuit = load_flight(flight_path)
    shuffled = shuffle_flight(circuit, np.random.default_rng(0))
    assert np.array_equal(shuffled.role, circuit.role) and np.array_equal(shuffled.pre, circuit.pre)
    assert np.array_equal(shuffled.coupling, circuit.coupling)
    assert np.array_equal(np.bincount(shuffled.post, minlength=9), np.bincount(circuit.post, minlength=9))


def test_a_centred_record_judges_an_episode_by_its_time_near_the_landmark(flight_path):
    from fly_cartpole.reflex import AdaptiveReflexFly

    fly = AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=0.5, sigma=0.2,
                            landmark="position", record="centred")
    fly.records = [10.0, 10.0]
    fly.trial = np.array([0.0, 0.0, 0.0, 0.1])
    for _ in range(30):
        fly.act(np.array([1.2, 0.0, 0.0, 0.0]))  # halfway to the track's edge: half a step's credit
    fly.reset_episode()
    # score 30 * 0.5 = 15; dopamine (15 - 10) / 10 = 0.5; step 0.5 * 0.5 * 0.1 / 0.2 = 0.125
    assert fly.records[-1] == pytest.approx(15.0) and fly.log_gains == pytest.approx([0.0, 0.0, 0.0, 0.125])


def test_episodes_are_judged_by_length_unless_asked_otherwise(flight_path):
    fly = adaptive(flight_path)
    for _ in range(7):
        fly.act(np.array([1.2, 0.0, 0.0, 0.0]))
    fly.reset_episode()
    assert fly.records == [7]


def test_a_balanced_record_gives_half_credit_for_staying_up_and_half_for_staying_near_the_landmark(flight_path):
    from fly_cartpole.reflex import AdaptiveReflexFly

    fly = AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=0.5, sigma=0.2,
                            landmark="position", record="balanced")
    fly.records = [10.0, 10.0]
    fly.trial = np.array([0.0, 0.0, 0.0, 0.1])
    for _ in range(30):
        fly.act(np.array([1.2, 0.0, 0.0, 0.0]))
    fly.reset_episode()
    # score 30 * (0.5 + 0.5 * 0.5) = 22.5; dopamine (22.5 - 10) / 10 = 1.25; step 0.5 * 1.25 * 0.1 / 0.2 = 0.3125
    assert fly.records[-1] == pytest.approx(22.5) and fly.log_gains == pytest.approx([0.0, 0.0, 0.0, 0.3125])


def test_a_fly_that_sees_the_landmark_move_tunes_that_gain_too(flight_path):
    from fly_cartpole.reflex import AdaptiveReflexFly

    fly = AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=0.5, sigma=0.2, landmark="motion")
    assert len(fly.log_gains) == 5 and fly.gains[4] > 0
    fly.records = [10, 10]
    finish_episode_of(fly, 30, [0.0, 0.0, 0.0, 0.0, 0.1])
    assert fly.log_gains == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.5])


def test_an_unknown_landmark_is_an_error(flight_path):
    from fly_cartpole.reflex import AdaptiveReflexFly

    with pytest.raises(ValueError, match="landmark"):
        AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=0.5, sigma=0.2, landmark="stripe")


def test_a_staged_fly_learns_to_balance_first_then_to_hold_station_with_balance_frozen(flight_path):
    from fly_cartpole.reflex import AdaptiveReflexFly

    fly = AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=0.5, sigma=0.2,
                            landmark="position", curriculum=2)
    off_centre = np.array([1.2, 0.0, 0.0, 0.0])
    for steps in (5, 7):
        assert fly.trial[:3].any()  # the first stage tries every gain
        for _ in range(steps):
            fly.act(off_centre)
        fly.reset_episode()
    # The second stage starts its own record, judged on time near the landmark, with the balance gains frozen.
    assert fly.records == [] and not fly.trial[:3].any() and fly.trial[3] != 0
    for _ in range(10):
        fly.act(off_centre)
    fly.reset_episode()
    assert fly.records == [pytest.approx(5.0)]


def test_a_curriculum_without_a_landmark_is_an_error(flight_path):
    from fly_cartpole.reflex import AdaptiveReflexFly

    with pytest.raises(ValueError, match="landmark"):
        AdaptiveReflexFly(load_flight(flight_path), np.random.default_rng(0), eta=0.5, sigma=0.2, curriculum=10)
