import numpy as np

from fly_cartpole.circuit import load_circuit, normalise_columns, shuffle_bipartite, shuffle_circuit


def test_load_circuit_reads_every_block(circuit_path):
    circuit = load_circuit(circuit_path)
    assert (circuit.n_pn, circuit.n_kc, circuit.n_mbon, circuit.n_dan) == (24, 60, 6, 4)
    assert circuit.pn_glomerulus[0] == "G0"
    assert circuit.dan_is_punishment.tolist() == [False, False, True, True]
    assert circuit.mbon_valence.tolist() == [-1, -1, 1, 1, 0, 0]
    assert circuit.mbon_dan.shape == (6, 4)


def test_normalise_columns_sums_to_one_and_leaves_empty_columns_empty():
    normalised = normalise_columns(np.array([[1.0, 0.0], [3.0, 0.0]]))
    assert np.allclose(normalised[:, 0], [0.25, 0.75])
    assert not normalised[:, 1].any()


def test_shuffle_keeps_degrees_and_weights_but_moves_edges():
    rng = np.random.default_rng(0)
    matrix = (rng.random((30, 80)) < 0.1) * rng.integers(1, 50, size=(30, 80))
    shuffled = shuffle_bipartite(matrix, np.random.default_rng(1))
    assert np.array_equal((shuffled > 0).sum(axis=0), (matrix > 0).sum(axis=0))
    assert np.array_equal((shuffled > 0).sum(axis=1), (matrix > 0).sum(axis=1))
    assert sorted(shuffled[shuffled > 0].tolist()) == sorted(matrix[matrix > 0].tolist())
    moved_fraction = ((shuffled > 0) != (matrix > 0)).sum() / (2 * (matrix > 0).sum())
    assert moved_fraction > 0.5


def test_shuffle_circuit_leaves_dopamine_wiring_alone(circuit_path):
    circuit = load_circuit(circuit_path)
    shuffled = shuffle_circuit(circuit, np.random.default_rng(0))
    assert np.array_equal(shuffled.dan_mbon, circuit.dan_mbon)
    assert not np.array_equal(shuffled.pn_kc, circuit.pn_kc)
    assert not np.array_equal(shuffled.kc_mbon, circuit.kc_mbon)
