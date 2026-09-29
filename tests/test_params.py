import json

from fly_cartpole.params import Hyperparameters, load_hyperparameters, save_hyperparameters


def test_missing_file_falls_back_to_defaults(tmp_path, capsys):
    assert load_hyperparameters(tmp_path / "missing.json") == Hyperparameters()
    assert "fly-cartpole tune" in capsys.readouterr().err


def test_round_trip_ignores_unknown_keys(tmp_path):
    path = tmp_path / "hyperparameters.json"
    save_hyperparameters(Hyperparameters(beta=7.0), path)
    stored = json.loads(path.read_text())
    stored["retired_setting"] = 1
    path.write_text(json.dumps(stored))
    assert load_hyperparameters(path) == Hyperparameters(beta=7.0)
