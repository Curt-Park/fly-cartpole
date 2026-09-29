from dataclasses import replace

import numpy as np

from fly_cartpole.circuit import load_circuit
from fly_cartpole.encoder import LEFT, RIGHT, build_encoder
from fly_cartpole.fly_agent import FlyAgent
from fly_cartpole.mushroom_body import MushroomBody
from fly_cartpole.params import Hyperparameters

STATE = np.array([0.1, 0.0, 0.02, -0.3])


def make_fly(circuit_path, **overrides) -> FlyAgent:
    circuit = load_circuit(circuit_path)
    params = replace(Hyperparameters(kc_sparsity=0.1), **overrides)
    body = MushroomBody(circuit, params.kc_sparsity)
    encoder = build_encoder(circuit.pn_glomerulus, params.action_fraction, params.tuning_width, seed=0)
    return FlyAgent(body, encoder, params, np.random.default_rng(0))


def test_act_returns_the_chosen_option(circuit_path):
    fly = make_fly(circuit_path)
    decision = fly.act(STATE)
    assert decision.action in (LEFT, RIGHT)
    glomeruli, kc, mbon, _ = fly.option(STATE, decision.action)
    assert np.array_equal(decision.kc, kc)
    assert np.array_equal(decision.glomeruli, glomeruli)
    assert decision.mbon.shape == (6,)


def test_trace_accumulates_chosen_codes_with_decay(circuit_path):
    fly = make_fly(circuit_path, trace_decay=0.5)
    first = fly.act(STATE).kc
    second = fly.act(STATE).kc
    assert np.allclose(fly.trace, 0.5 * first + second)
    fly.reset_episode()
    assert not fly.trace.any()


def test_punishment_depresses_only_compartments_with_ppl1_input(circuit_path):
    fly = make_fly(circuit_path, gain_decay=0.0)
    fly.act(STATE)
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    depressed = fly.gain < 1.0
    assert np.flatnonzero(depressed.any(axis=0)).tolist() == [2, 3]
    assert set(np.flatnonzero(depressed.any(axis=1)).tolist()) <= set(np.flatnonzero(fly.trace).tolist())


def test_reward_depresses_only_compartments_with_pam_input(circuit_path):
    fly = make_fly(circuit_path, gain_decay=0.0)
    fly.act(STATE)
    fly.learn(punish=0.0, reward=1.0, next_state=STATE, terminated=False)
    assert np.flatnonzero((fly.gain < 1.0).any(axis=0)).tolist() == [0, 1, 2]


def test_decay_pulls_depressed_synapses_back_toward_one(circuit_path):
    fly = make_fly(circuit_path, gain_decay=0.1)
    fly.gain[:] = 0.5
    fly.learn(punish=0.0, reward=0.0, next_state=STATE, terminated=False)
    assert np.allclose(fly.gain, 0.55)


def test_gains_never_go_negative(circuit_path):
    fly = make_fly(circuit_path, learning_rate=1000.0)
    fly.act(STATE)
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    assert fly.gain.min() == 0.0


def test_frozen_fly_never_changes(circuit_path):
    fly = make_fly(circuit_path, learning_rate=0.0)
    fly.act(STATE)
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    assert np.all(fly.gain == 1.0)


def test_punishing_an_option_lowers_its_score_against_the_other(circuit_path):
    fly = make_fly(circuit_path, gain_decay=0.0)
    before = fly.option(STATE, LEFT)[3] - fly.option(STATE, RIGHT)[3]
    fly.trace = fly.option(STATE, LEFT)[1].copy()
    fly.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    after = fly.option(STATE, LEFT)[3] - fly.option(STATE, RIGHT)[3]
    assert after < before
