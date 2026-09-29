from dataclasses import replace

import numpy as np
import pytest

from fly_cartpole.bilateral import BilateralFly, build_bilateral
from fly_cartpole.circuit import load_circuit
from fly_cartpole.encoder import LEFT, RIGHT, ROLE_PUSH_LEFT
from fly_cartpole.params import Hyperparameters

STATE = np.array([0.1, 0.0, 0.02, -0.3])


def make(circuit_path, **overrides) -> BilateralFly:
    params = replace(Hyperparameters(kc_sparsity=0.1, bilateral_gain_decay=0.0), **overrides)
    circuit = load_circuit(circuit_path)
    return build_bilateral(circuit, circuit, params, seed=0)


def test_both_hemispheres_hear_the_same_state_and_no_action(circuit_path):
    fly = make(circuit_path)
    decision = fly.act(STATE)
    assert not decision.glomeruli[fly.encoder.group_role >= ROLE_PUSH_LEFT].any()
    assert decision.action in (LEFT, RIGHT)
    assert decision.scores == pytest.approx(tuple(fly.values(STATE)))


def test_only_the_chosen_hemisphere_collects_a_trace(circuit_path):
    fly = make(circuit_path)
    decision = fly.act(STATE)
    other = RIGHT if decision.action == LEFT else LEFT
    assert np.array_equal(fly.traces[decision.action], decision.kc)
    assert not fly.traces[other].any()


def test_a_fall_lowers_the_value_of_the_chosen_side_only(circuit_path):
    fly = make(circuit_path, bilateral_learning_rate=1.0)
    decision = fly.act(STATE)
    other = RIGHT if decision.action == LEFT else LEFT
    before = fly.values(STATE)
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    after = fly.values(STATE)
    assert after[decision.action] < before[decision.action]
    assert after[other] == pytest.approx(before[other])
    assert fly.released == pytest.approx((1.0 + before[decision.action], 0.0))


def test_doing_better_than_expected_raises_the_chosen_value(circuit_path):
    fly = make(circuit_path, bilateral_learning_rate=0.5)
    decision = fly.act(STATE)
    before = fly.values(STATE)[decision.action]
    fly.learn(punish=0.0, reward=1.0, next_state=STATE, terminated=True)
    assert fly.values(STATE)[decision.action] > before


def test_centred_hemispheres_start_with_no_average_preference(circuit_path):
    fly = make(circuit_path, bilateral_centre=True)
    average = np.mean([fly.values(state) for state in fly.calibration_states], axis=0)
    assert np.allclose(average, 0.0, atol=1e-12)
    assert fly.offsets.any()


def test_uncentred_hemispheres_keep_their_innate_offset(circuit_path):
    fly = make(circuit_path, bilateral_centre=False)
    assert not fly.offsets.any()
