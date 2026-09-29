import json

from fly_cartpole.export import export
from fly_cartpole.params import Hyperparameters

PARAMS = Hyperparameters(kc_sparsity=0.1)


def run_export(circuit_path, tmp_path) -> dict:
    return export(seed=0, episodes=3, params=PARAMS, circuit_path=circuit_path, left_circuit_path=circuit_path,
                  web_data_dir=tmp_path)


def test_export_writes_a_model_the_browser_can_run(circuit_path, tmp_path):
    summary = run_export(circuit_path, tmp_path)
    model = json.loads((tmp_path / "model.json").read_text())
    assert len(model["sides"]) == 2
    side = model["sides"][0]
    assert len(side["kc_mbon_naive"]) == len(side["kc_mbon_trained"]) == len(side["kc_mbon"]["rows"])
    assert len(side["pn_group"]) == 24 and side["n_kc"] == 60 and side["k"] == 6
    assert len(model["samples"][0]["values_trained"]) == 2
    assert len(model["physics"]["states"]) == len(model["physics"]["actions"])
    assert summary["training_episodes"] == 3
    assert model["outcome_scale"] == PARAMS.bilateral_outcome_scale


def test_cells_without_soma_become_null_in_valid_json(circuit_path, tmp_path):
    run_export(circuit_path, tmp_path)
    text = (tmp_path / "cells.json").read_text()
    cells = json.loads(text)
    assert "NaN" not in text
    assert cells["right"]["kc"][0] is None and cells["left"]["kc"][0] is None
    assert len(cells["dan"]) == 4 and len(cells["dan_is_punishment"]) == 4
    assert all(abs(value) <= 1.0 for point in cells["right"]["pn"] for value in point)
