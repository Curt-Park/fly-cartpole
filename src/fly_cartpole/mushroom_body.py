"""Fixed mushroom body wiring: PN -> KC -> MBON, and where dopamine lands."""

from __future__ import annotations

import numpy as np

from .circuit import Circuit, normalise_columns


def kenyon_code(drive: np.ndarray, k: int) -> np.ndarray:
    """Keep exactly the k strongest cells; k-winners-take-all stands in for APL inhibition."""
    code = np.zeros_like(drive)
    if drive.max() <= 0:
        return code
    # Stable order breaks ties toward the lowest index, which the browser viewer reproduces exactly.
    winners = np.argsort(-drive, kind="stable")[:k]
    code[winners] = drive[winners]
    return code / code.max()


class MushroomBody:
    def __init__(self, circuit: Circuit, kc_sparsity: float) -> None:
        self.pn_kc = normalise_columns(circuit.pn_kc)
        self.kc_mbon = normalise_columns(circuit.kc_mbon)
        self.dan_mbon = normalise_columns(circuit.dan_mbon)
        self.mbon_dan = normalise_columns(circuit.mbon_dan)
        self.mbon_valence = circuit.mbon_valence.astype(np.float64)
        self.dan_is_punishment = circuit.dan_is_punishment
        self.n_kc, self.n_mbon = circuit.n_kc, circuit.n_mbon
        self.k = max(1, round(kc_sparsity * circuit.n_kc))
        # PN->KC is ~3% dense; summing only real edges, PN by PN, is faster and matches the browser's order.
        self.pn_rows, self.pn_cols = np.nonzero(self.pn_kc)
        self.pn_values = self.pn_kc[self.pn_rows, self.pn_cols]

    def kenyon(self, pn: np.ndarray) -> np.ndarray:
        drive = np.bincount(self.pn_cols, weights=pn[self.pn_rows] * self.pn_values, minlength=self.n_kc)
        return kenyon_code(drive, self.k)

    def mbon(self, kc: np.ndarray, gain: np.ndarray) -> np.ndarray:
        active = np.flatnonzero(kc)
        return kc[active] @ (self.kc_mbon[active] * gain[active])

    def dopamine_at_mbon(self, punish: float, reward: float) -> np.ndarray:
        return self.dopamine_from(np.where(self.dan_is_punishment, punish, reward))

    def dopamine_from(self, dan_activity: np.ndarray) -> np.ndarray:
        # Anatomy, not a parameter, decides which compartments a signal can change.
        return dan_activity @ self.dan_mbon
