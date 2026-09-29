import numpy as np
import pytest

N_PN, N_KC, N_MBON, N_DAN = 24, 60, 6, 4


def synthetic_arrays(seed: int = 0) -> dict[str, np.ndarray]:
    """A tiny circuit with every field of data/mb_right.npz."""
    rng = np.random.default_rng(seed)
    pn_kc = np.zeros((N_PN, N_KC), dtype=np.int32)
    for kc in range(N_KC):
        pn_kc[rng.choice(N_PN, size=4, replace=False), kc] = rng.integers(1, 20, size=4)
    kc_mbon = ((rng.random((N_KC, N_MBON)) < 0.5) * rng.integers(1, 10, size=(N_KC, N_MBON))).astype(np.int32)
    kc_mbon[:, 0] = np.maximum(kc_mbon[:, 0], 1)
    # PAM rows 0-1 reach MBON 0, 1, 2; PPL1 rows 2-3 reach MBON 2, 3; MBON 4, 5 get no dopamine.
    dan_mbon = np.array(
        [[5, 0, 1, 0, 0, 0], [0, 4, 0, 0, 0, 0], [0, 0, 6, 3, 0, 0], [0, 0, 0, 3, 0, 0]],
        dtype=np.int32,
    )
    # MBON 4 feeds nothing back; DAN 3 (PPL1) hears no MBON.
    mbon_dan = np.array(
        [[2, 0, 1, 0], [0, 3, 0, 0], [1, 0, 2, 0], [0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]],
        dtype=np.int32,
    )
    arrays = {
        "mbon_dan": mbon_dan,
        "pn_kc": pn_kc,
        "kc_mbon": kc_mbon,
        "dan_mbon": dan_mbon,
        "pn_glomerulus": np.repeat(np.array([f"G{index}" for index in range(12)]), 2),
        "mbon_valence": np.array([-1, -1, 1, 1, 0, 0], dtype=np.int8),
        "dan_is_punishment": np.array([0, 0, 1, 1], dtype=np.int8),
    }
    for name, size in (("pn", N_PN), ("kc", N_KC), ("mbon", N_MBON), ("dan", N_DAN)):
        arrays[f"{name}_body_id"] = np.arange(size, dtype=np.int64) + 10_000
        arrays[f"{name}_soma"] = rng.normal(30_000.0, 2_000.0, size=(size, 3))
    arrays["kc_soma"][0] = np.nan
    return arrays


@pytest.fixture
def circuit_path(tmp_path):
    path = tmp_path / "mb_right.npz"
    np.savez_compressed(path, **synthetic_arrays())
    return path
