"""Fixed mushroom body wiring: PN -> KC -> MBON, and where dopamine lands."""

from __future__ import annotations

import numpy as np

from .circuit import Circuit, normalise_columns


def kenyon_code(drive: np.ndarray, k: int) -> np.ndarray:
    """Keep exactly the k strongest cells; k-winners-take-all stands in for APL inhibition."""
    code = np.zeros_like(drive)
    if drive.max() <= 0:
        return code
    winners = np.argpartition(drive, -k)[-k:]
    code[winners] = drive[winners]
    return code / code.max()


class MushroomBody:
    def __init__(self, circuit: Circuit, kc_sparsity: float) -> None:
        self.pn_kc = normalise_columns(circuit.pn_kc)
        self.kc_mbon = normalise_columns(circuit.kc_mbon)
        self.dan_mbon = normalise_columns(circuit.dan_mbon)
        self.mbon_valence = circuit.mbon_valence.astype(np.float64)
        self.dan_is_punishment = circuit.dan_is_punishment
        self.n_kc, self.n_mbon = circuit.n_kc, circuit.n_mbon
        self.k = max(1, round(kc_sparsity * circuit.n_kc))

    def kenyon(self, pn: np.ndarray) -> np.ndarray:
        return kenyon_code(pn @ self.pn_kc, self.k)

    def mbon(self, kc: np.ndarray, gain: np.ndarray) -> np.ndarray:
        return kc @ (self.kc_mbon * gain)

    def dopamine_at_mbon(self, punish: float, reward: float) -> np.ndarray:
        # Anatomy, not a parameter, decides which compartments a signal can change.
        return np.where(self.dan_is_punishment, punish, reward) @ self.dan_mbon
