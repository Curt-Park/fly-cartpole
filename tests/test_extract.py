import numpy as np
import pandas as pd
import pytest

from fly_cartpole.circuit import load_circuit
from fly_cartpole.extract import build_circuit_arrays, download_file
from fly_cartpole.paths import CIRCUIT_PATH


def annotations_table() -> pd.DataFrame:
    rows = [
        (1, "ALPN", "R", "DA1_lPN", np.array([10, 20, 30])),
        (2, "ALPN", "R", "DA1_lPN", np.array([11, 21, 31])),
        (3, "ALPN", "R", "VA1v_adPN", None),
        (4, "ALPN", "L", "DA1_lPN", np.array([1, 2, 3])),
        (10, "Kenyon_Cell", "R", "KCg-m", np.array([5, 5, 5])),
        (11, "Kenyon_Cell", "R", "KCg-m", np.array([6, 6, 6])),
        (12, "Kenyon_Cell", "R", "KCab", np.array([7, 7, 7])),
        (20, "MBON", "R", "MBON01", np.array([8, 8, 8])),
        (21, "MBON", "R", "MBON02", np.array([9, 9, 9])),
        (22, "MBON", "R", "MBON03", np.array([9, 9, 9])),
        (30, "DAN", "R", "PAM01", np.array([1, 1, 1])),
        (31, "DAN", "L", "PPL101", np.array([2, 2, 2])),
        (32, "DAN", "R", "PAM02", np.array([3, 3, 3])),
    ]
    return pd.DataFrame(rows, columns=["bodyId", "class", "somaSide", "type", "somaLocation"])


def edges_table() -> pd.DataFrame:
    rows = [
        (1, 10, 5), (2, 10, 3), (3, 11, 4), (1, 12, 2),  # PN -> KC
        (10, 20, 7), (11, 21, 6),  # KC -> MBON; KC 12 never reaches an MBON
        (30, 20, 9), (31, 21, 8), (31, 20, 2),  # DAN -> MBON; PAM 32 reaches none
        (4, 10, 50),  # left-side PN, not selected
        (21, 31, 3), (20, 30, 1),  # MBON -> DAN feedback
    ]
    return pd.DataFrame(rows, columns=["body_pre", "body_post", "weight"])


def test_build_keeps_right_side_cells_on_the_pathway():
    arrays, manifest = build_circuit_arrays(annotations_table(), edges_table())
    assert arrays["pn_body_id"].tolist() == [1, 2, 3]
    assert arrays["kc_body_id"].tolist() == [10, 11]
    assert arrays["mbon_body_id"].tolist() == [20, 21, 22]
    assert arrays["dan_body_id"].tolist() == [30, 31]
    assert arrays["pn_kc"].tolist() == [[5, 0], [3, 0], [0, 4]]
    assert manifest["counts"]["kc_dropped_off_pathway"] == 1


def test_build_derives_valence_from_the_dominant_dopamine_input():
    arrays, manifest = build_circuit_arrays(annotations_table(), edges_table())
    assert arrays["mbon_valence"].tolist() == [-1, 1, 0]
    assert arrays["dan_is_punishment"].tolist() == [0, 1]
    assert manifest["mbon_valence"] == {"approach": 1, "avoidance": 1, "neutral": 1}


def test_build_labels_glomeruli_and_keeps_missing_somas_as_nan():
    arrays, manifest = build_circuit_arrays(annotations_table(), edges_table())
    assert arrays["pn_glomerulus"].tolist() == ["DA1", "DA1", "VA1v"]
    assert np.isnan(arrays["pn_soma"][2]).all()
    assert arrays["kc_soma"][1].tolist() == [6.0, 6.0, 6.0]
    assert manifest["cells_without_soma"]["pn"] == 1


def test_download_replaces_a_stale_partial_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"complete payload")
    target = tmp_path / "cache" / "file.bin"
    target.parent.mkdir()
    target.with_name("file.bin.part").write_bytes(b"trunc")
    download_file(source.as_uri(), target)
    assert target.read_bytes() == b"complete payload"
    assert not target.with_name("file.bin.part").exists()


def test_download_skips_a_complete_file(tmp_path):
    target = tmp_path / "file.bin"
    target.write_bytes(b"already here")
    download_file((tmp_path / "missing.bin").as_uri(), target)
    assert target.read_bytes() == b"already here"


@pytest.mark.skipif(not CIRCUIT_PATH.exists(), reason="run `fly-cartpole extract` first")
def test_committed_circuit_matches_malecns_counts():
    circuit = load_circuit()
    assert circuit.n_pn == 343
    assert circuit.n_mbon == 49
    assert set(circuit.mbon_valence.tolist()) <= {-1, 0, 1}
    assert (circuit.pn_kc.sum(axis=0) > 0).all()
    assert (circuit.kc_mbon.sum(axis=1) > 0).all()


def test_build_keeps_mbon_to_dopamine_feedback():
    arrays, manifest = build_circuit_arrays(annotations_table(), edges_table())
    assert arrays["mbon_dan"].tolist() == [[1, 0], [0, 3], [0, 0]]
    assert manifest["edges"]["mbon_dan"] == 2
