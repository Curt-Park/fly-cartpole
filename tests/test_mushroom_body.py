import numpy as np

from fly_cartpole.circuit import load_circuit
from fly_cartpole.mushroom_body import MushroomBody, kenyon_code


def test_kenyon_code_keeps_exactly_k_cells_even_with_ties():
    code = kenyon_code(np.array([0.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.5]), k=3)
    assert (code > 0).sum() == 3
    assert code.max() == 1.0
    assert set(np.flatnonzero(code).tolist()) <= {1, 2, 3, 4}


def test_kenyon_code_of_no_input_is_silent():
    assert not kenyon_code(np.zeros(5), k=2).any()


def test_k_follows_sparsity(circuit_path):
    assert MushroomBody(load_circuit(circuit_path), kc_sparsity=0.1).k == 6


def test_dopamine_lands_only_where_each_population_innervates(circuit_path):
    body = MushroomBody(load_circuit(circuit_path), kc_sparsity=0.1)
    assert np.flatnonzero(body.dopamine_at_mbon(punish=1.0, reward=0.0)).tolist() == [2, 3]
    assert np.flatnonzero(body.dopamine_at_mbon(punish=0.0, reward=1.0)).tolist() == [0, 1, 2]


def test_kenyon_ties_go_to_the_lowest_index_so_the_browser_can_match():
    code = kenyon_code(np.ones(20), k=3)
    assert np.flatnonzero(code).tolist() == [0, 1, 2]
