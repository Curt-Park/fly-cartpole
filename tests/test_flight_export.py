import json
import pytest

from fly_cartpole.flight_export import export_flight
from fly_cartpole.reflex_report import ReflexParameters

QUICK = ReflexParameters(adapt_episodes=3, fixed_episodes=2)


def test_export_writes_a_circuit_the_browser_can_run(flight_path, tmp_path):
    summary = export_flight(seed=0, params=QUICK, flight_path=flight_path, web_data_dir=tmp_path)
    text = (tmp_path / "flight.json").read_text()
    model = json.loads(text)
    assert "NaN" not in text
    assert len(model["pre"]) == len(model["post"]) == len(model["coupling"]) == 8
    assert len(model["positions"]) == 9 and model["positions"][8] is None
    assert min(model["gains_tuned"][:4]) > 0
    trajectory = model["trajectory"]
    assert len(trajectory["states"]) == len(trajectory["actions"]) == len(trajectory["steers"]) > 0
    assert summary["adaptation_episodes"] == 3


def test_afferents_without_a_position_are_drawn_where_they_synapse(tmp_path):
    import numpy as np

    from conftest import synthetic_flight_arrays

    arrays = synthetic_flight_arrays()
    arrays["soma"][0] = np.nan  # the left haltere afferent synapses only onto b1 L (neuron 6)
    path = tmp_path / "flight.npz"
    np.savez_compressed(path, **arrays)
    export_flight(seed=0, params=QUICK, flight_path=path, web_data_dir=tmp_path)
    positions = json.loads((tmp_path / "flight.json").read_text())["positions"]
    assert positions[0] == positions[6]


def test_scene_is_centred_on_its_extent_not_on_where_cells_crowd():
    import numpy as np
    from fly_cartpole.flight_export import scene_positions

    crowded = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    depth = [point[2] for point in scene_positions(crowded)]
    assert min(depth) == -1.0 and max(depth) == 1.0


def test_adapted_gains_are_reported_relative_to_their_geometric_mean():
    import numpy as np
    from fly_cartpole.reflex import relative_gains

    gains = relative_gains(np.array([0.9, 0.5, -0.2]))
    assert np.prod(gains[:3]) == pytest.approx(1.0) and gains[3:] == [0.0, 0.0]
    assert gains[0] / gains[1] == pytest.approx(np.exp(0.4))
    assert np.prod(relative_gains(np.array([0.9, 0.5, -0.2, 0.4, 0.1]))) == pytest.approx(1.0)


def test_the_viewer_runs_the_gains_it_reports(flight_path, tmp_path):
    summary = export_flight(seed=0, params=QUICK, flight_path=flight_path, web_data_dir=tmp_path)
    model = json.loads((tmp_path / "flight.json").read_text())
    # The landmark is part of the fly in both settings, and the parity trajectory exercises every sense.
    assert model["gains_fixed"] == [1.0] * 5 and "gains_adapted" not in model
    assert summary["gains_tuned"] == model["gains_tuned"] == model["trajectory"]["gains"]
    # The viewer runs the reference fly: it learned to balance, then to hold station by the landmark's place and slide.
    assert len(model["gains_tuned"]) == 5 and min(model["gains_tuned"]) > 0
