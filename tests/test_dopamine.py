import pytest

from fly_cartpole.dopamine import DopamineSchedule


def schedule_after(lengths, mode="mean", window=2):
    schedule = DopamineSchedule(mode, window=window, reward_per_step=0.1)
    for length in lengths:
        schedule.end_episode(length)
    return schedule


def test_no_reward_before_any_episode_completes():
    schedule = schedule_after([])
    assert schedule.baseline is None
    assert schedule.signal(step=400, terminated=False) == (0.0, 0.0)


def test_reward_starts_after_passing_the_mean_of_recent_episodes():
    schedule = schedule_after([100, 10, 20])
    assert schedule.baseline == 15.0
    assert schedule.signal(15, terminated=False) == (0.0, 0.0)
    assert schedule.signal(16, terminated=False) == (0.0, 0.1)


def test_best_mode_uses_the_longest_episode():
    assert schedule_after([100, 10, 20], mode="best").baseline == 100.0


def test_a_fall_is_punished_and_never_rewarded():
    assert schedule_after([5]).signal(50, terminated=True) == (1.0, 0.0)


def test_reaching_the_time_limit_is_not_punished():
    assert schedule_after([300]).signal(500, terminated=False) == (0.0, 0.1)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        DopamineSchedule("median", window=20, reward_per_step=0.1)
