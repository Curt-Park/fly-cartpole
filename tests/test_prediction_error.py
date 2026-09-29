from dataclasses import replace

import numpy as np
import pytest

from fly_cartpole.circuit import load_circuit
from fly_cartpole.encoder import build_encoder
from fly_cartpole.fly_agent import PredictionErrorFly, WiredPredictionErrorFly
from fly_cartpole.mushroom_body import MushroomBody
from fly_cartpole.params import Hyperparameters

STATE = np.array([0.1, 0.0, 0.02, -0.3])


def make(agent_class, circuit_path, **overrides):
    circuit = load_circuit(circuit_path)
    params = replace(Hyperparameters(kc_sparsity=0.1, gain_decay=0.0), **overrides)
    body = MushroomBody(circuit, params.kc_sparsity)
    encoder = build_encoder(circuit.pn_glomerulus, params.action_fraction, params.tuning_width, seed=0)
    return agent_class(body, encoder, params, np.random.default_rng(0))


def split(error):
    return (max(-error, 0.0), max(error, 0.0))


def test_a_fall_releases_punishment_for_the_missed_prediction(circuit_path):
    fly = make(PredictionErrorFly, circuit_path)
    decision = fly.act(STATE)
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    assert fly.released == pytest.approx(split(-1.0 - decision.scores[decision.action]))


def test_an_unexpected_reward_releases_only_reward_dopamine(circuit_path):
    fly = make(PredictionErrorFly, circuit_path)
    decision = fly.act(STATE)
    fly.learn(punish=0.0, reward=5.0, next_state=STATE, terminated=True)
    assert fly.released[0] == 0.0
    assert fly.released[1] == pytest.approx(5.0 - decision.scores[decision.action])


def test_the_expected_future_counts_toward_the_error(circuit_path):
    fly = make(PredictionErrorFly, circuit_path, gamma=0.5)
    decision = fly.act(STATE)
    upcoming = fly.expected_score(STATE)
    fly.learn(punish=0.0, reward=0.0, next_state=STATE, terminated=False)
    assert fly.released == pytest.approx(split(0.5 * upcoming - decision.scores[decision.action]))


def test_a_dopamine_neuron_without_mbon_feedback_hears_only_reinforcement(circuit_path):
    fly = make(WiredPredictionErrorFly, circuit_path)
    fly.act(STATE)
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    assert fly.dan_activity[3] == pytest.approx(1.0)


def test_wired_errors_follow_each_neurons_own_mbon_inputs(circuit_path):
    fly = make(WiredPredictionErrorFly, circuit_path)
    decision = fly.act(STATE)
    predictions = fly.dan_predictions(decision.mbon)
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    errors = -1.0 - predictions
    expected = np.where(fly.body.dan_is_punishment, np.maximum(-errors, 0.0), np.maximum(errors, 0.0))
    assert np.allclose(fly.dan_activity, expected)
