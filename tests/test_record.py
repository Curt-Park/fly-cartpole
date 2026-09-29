import json

from fly_cartpole.params import Hyperparameters
from fly_cartpole.record import FRAME_FIELDS, record

PARAMS = Hyperparameters(kc_sparsity=0.1)


def test_record_exports_a_naive_and_a_trained_episode(circuit_path, tmp_path):
    summary = record(seed=0, episodes=3, params=PARAMS, circuit_path=circuit_path, web_data_dir=tmp_path)
    episodes = json.loads((tmp_path / "episodes.json").read_text())
    for name in ("naive", "trained"):
        frames = episodes[name]
        assert set(FRAME_FIELDS) <= set(frames)
        assert len(frames["step"]) == len(frames["kc"]) == len(frames["mbon"]) >= 1
        assert len(frames["mbon"][0]) == 6
        assert len(frames["glomeruli"][0]) == 12
    assert episodes["naive"]["baseline"][0] is None
    assert summary["trained_length"] == len(episodes["trained"]["step"])


def test_cells_without_soma_become_null_in_valid_json(circuit_path, tmp_path):
    record(seed=0, episodes=1, params=PARAMS, circuit_path=circuit_path, web_data_dir=tmp_path)
    text = (tmp_path / "cells.json").read_text()
    cells = json.loads(text)
    assert "NaN" not in text
    assert cells["kc"][0] is None
    assert len(cells["kc"]) == 60 and len(cells["pn_group"]) == 24
    assert all(abs(value) <= 1.0 for point in cells["pn"] for value in point)
