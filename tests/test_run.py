from dataclasses import replace

import numpy as np
import pytest

from fly_cartpole.circuit import load_circuit
from fly_cartpole.params import Hyperparameters
from fly_cartpole.run import CONDITIONS, load_lengths, make_agent, run_condition, run_episodes, run_many, save_lengths

PARAMS = Hyperparameters(kc_sparsity=0.1)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_each_condition_runs_end_to_end(circuit_path, condition):
    lengths = run_condition(condition, seed=0, episodes=5, params=PARAMS, circuit_path=circuit_path,
                            left_circuit_path=circuit_path)
    assert len(lengths) == 5
    assert all(1 <= length <= 500 for length in lengths)


def test_the_same_seed_repeats_exactly(circuit_path):
    assert run_condition("fly", 3, 5, PARAMS, circuit_path) == run_condition("fly", 3, 5, PARAMS, circuit_path)


def test_on_step_sees_every_step(circuit_path):
    agent, schedule = make_agent("fly", load_circuit(circuit_path), PARAMS, seed=0)
    seen = []
    lengths = run_episodes(agent, schedule, 2, seed=0, on_step=lambda episode, step, *rest: seen.append((episode, step)))
    assert len(seen) == sum(lengths)
    assert seen[-1] == (1, lengths[1])


def test_unknown_condition_is_rejected(circuit_path):
    with pytest.raises(ValueError):
        make_agent("bee", load_circuit(circuit_path), PARAMS, seed=0)


def test_run_many_keeps_job_order(circuit_path):
    jobs = [("fly", 0, PARAMS), ("td", 0, PARAMS)]
    lengths = run_many(jobs, episodes=3, workers=1, circuit_path=circuit_path)
    assert lengths[1] == run_condition("td", 0, 3, PARAMS, circuit_path)


def test_results_round_trip(tmp_path):
    save_lengths("fly", 2, [10, 20], PARAMS, tmp_path)
    assert load_lengths("fly", tmp_path) == {2: [10, 20]}


def test_random_reference_presses_both_buttons_and_never_learns(circuit_path):
    agent, _ = make_agent("random", load_circuit(circuit_path), PARAMS, seed=0)
    actions = {agent.act(np.zeros(4)).action for _ in range(50)}
    agent.learn(punish=1.0, reward=0.0, next_state=np.zeros(4), terminated=True)
    assert actions == {0, 1}
    assert "random" in CONDITIONS


def test_the_posture_signal_reaches_the_learner(circuit_path):
    agent, schedule = make_agent("random", load_circuit(circuit_path), replace(PARAMS, posture_weight=3.0), seed=0)
    rewards = []
    agent.learn = lambda punish, reward, next_state, terminated: rewards.append(reward)
    run_episodes(agent, schedule, 1, seed=0)
    assert rewards and all(reward != 0.0 for reward in rewards)
