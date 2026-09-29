from dataclasses import replace

import numpy as np

from fly_cartpole.circuit import load_circuit
from fly_cartpole.encoder import ROLE_PUSH_LEFT, build_encoder
from fly_cartpole.mushroom_body import MushroomBody
from fly_cartpole.params import Hyperparameters
from fly_cartpole.td_agent import TDAgent

STATE = np.array([0.1, 0.0, 0.02, -0.3])


def make_td(circuit_path, **overrides) -> TDAgent:
    circuit = load_circuit(circuit_path)
    params = replace(Hyperparameters(kc_sparsity=0.1), **overrides)
    body = MushroomBody(circuit, params.kc_sparsity)
    encoder = build_encoder(circuit.pn_glomerulus, params.action_fraction, params.tuning_width, seed=0)
    return TDAgent(body, encoder, params, np.random.default_rng(0))


def test_td_sees_the_state_only(circuit_path):
    td = make_td(circuit_path)
    decision = td.act(STATE)
    assert not decision.glomeruli[td.encoder.group_role >= ROLE_PUSH_LEFT].any()
    assert decision.mbon.size == 0


def test_a_punished_action_becomes_less_likely(circuit_path):
    td = make_td(circuit_path, td_actor_lr=0.5, td_critic_lr=0.5)
    decision = td.act(STATE)
    before = td.probabilities(STATE)[decision.action]
    td.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    assert td.probabilities(STATE)[decision.action] < before


def test_the_value_of_a_punished_state_drops(circuit_path):
    td = make_td(circuit_path, td_critic_lr=0.5)
    td.act(STATE)
    td.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    assert td.state_value(STATE) < 0.0


def test_one_critic_update_never_overshoots_its_target(circuit_path):
    td = make_td(circuit_path, td_critic_lr=1.0)
    td.act(STATE)
    td.learn(punish=1.0, reward=0.0, next_state=STATE, terminated=True)
    assert -1.0 <= td.state_value(STATE) < 0.0
