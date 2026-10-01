"""The right mushroom body circuit extracted from MaleCNS v1.0."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .paths import CIRCUIT_PATH


@dataclass(frozen=True)
class Circuit:
    pn_kc: np.ndarray
    kc_mbon: np.ndarray
    dan_mbon: np.ndarray
    mbon_dan: np.ndarray
    pn_glomerulus: np.ndarray
    mbon_valence: np.ndarray
    dan_is_punishment: np.ndarray

    @property
    def n_pn(self) -> int:
        return self.pn_kc.shape[0]

    @property
    def n_kc(self) -> int:
        return self.pn_kc.shape[1]

    @property
    def n_mbon(self) -> int:
        return self.kc_mbon.shape[1]

    @property
    def n_dan(self) -> int:
        return self.dan_mbon.shape[0]


def load_circuit(path: Path = CIRCUIT_PATH) -> Circuit:
    with np.load(path) as arrays:
        return Circuit(
            pn_kc=arrays["pn_kc"].astype(np.float64),
            kc_mbon=arrays["kc_mbon"].astype(np.float64),
            dan_mbon=arrays["dan_mbon"].astype(np.float64),
            mbon_dan=arrays["mbon_dan"].astype(np.float64),
            pn_glomerulus=arrays["pn_glomerulus"].astype(str),
            mbon_valence=arrays["mbon_valence"].astype(np.int8),
            dan_is_punishment=arrays["dan_is_punishment"].astype(bool),
        )


def normalise_columns(matrix: np.ndarray) -> np.ndarray:
    """Scale each postsynaptic cell's inputs to sum to one."""
    totals = matrix.sum(axis=0, keepdims=True)
    totals[totals == 0] = 1.0
    return matrix / totals


def shuffle_bipartite(matrix: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Degree-preserving randomisation by double-edge swaps; synapse counts travel with their edges."""
    rows, cols = np.nonzero(matrix)
    weights = matrix[rows, cols]
    cols = cols.copy()
    n_edges = len(rows)
    if n_edges < 2:
        return matrix.copy()
    occupied = set(zip(rows.tolist(), cols.tolist()))
    for first, second in rng.integers(n_edges, size=(swaps_per_edge * n_edges, 2)).tolist():
        row_a, col_a, row_b, col_b = int(rows[first]), int(cols[first]), int(rows[second]), int(cols[second])
        if row_a == row_b or col_a == col_b or (row_a, col_b) in occupied or (row_b, col_a) in occupied:
            continue
        occupied.difference_update({(row_a, col_a), (row_b, col_b)})
        occupied.update({(row_a, col_b), (row_b, col_a)})
        cols[first], cols[second] = col_b, col_a
    shuffled = np.zeros_like(matrix)
    shuffled[rows, cols] = weights
    return shuffled


def shuffle_edges(pre: np.ndarray, post: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Degree-preserving double-edge swaps on a directed graph; returns new targets, never self or duplicate connections."""
    targets = post.copy()
    occupied = set(zip(pre.tolist(), targets.tolist()))
    for first, second in rng.integers(len(pre), size=(swaps_per_edge * len(pre), 2)).tolist():
        source_a, target_a, source_b, target_b = int(pre[first]), int(targets[first]), int(pre[second]), int(targets[second])
        if source_a == source_b or target_a == target_b or source_a == target_b or source_b == target_a:
            continue
        if (source_a, target_b) in occupied or (source_b, target_a) in occupied:
            continue
        occupied.difference_update({(source_a, target_a), (source_b, target_b)})
        occupied.update({(source_a, target_b), (source_b, target_a)})
        targets[first], targets[second] = target_b, target_a
    return targets


def shuffle_circuit(circuit: Circuit, rng: np.random.Generator) -> Circuit:
    # DAN->MBON stays intact: it defines what dopamine is able to change.
    return replace(
        circuit,
        pn_kc=shuffle_bipartite(circuit.pn_kc, rng),
        kc_mbon=shuffle_bipartite(circuit.kc_mbon, rng),
    )
