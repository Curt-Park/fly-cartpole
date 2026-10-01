import json

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
    assert model["gains_fixed"] == [1.0, 1.0, 1.0] and min(model["gains_adapted"]) > 0
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
