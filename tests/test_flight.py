import numpy as np
import pandas as pd
import pytest

from fly_cartpole.flight import ROLES, build_flight_arrays, on_paths, transmitter_signs

# body: (type, superclass, subclass, somaSide, rootSide, somaLocation, tosomaLocation, transmitter)
BODIES = {
    1: ("SApp", "vnc_sensory", "haltere", None, "L", None, [1.0, 2.0, 3.0], "acetylcholine"),
    2: ("SApp", "vnc_sensory", "haltere", None, "R", None, [4.0, 5.0, 6.0], "acetylcholine"),
    3: ("OCG01a", "cb_sensory", "", "L", "L", [7.0, 8.0, 9.0], None, "acetylcholine"),
    4: ("OCG01a", "cb_sensory", "", "R", "R", [7.0, 8.0, 9.0], None, "acetylcholine"),
    5: ("HSN", "visual_projection", "", "L", "L", [0.0, 0.0, 0.0], None, "acetylcholine"),
    6: ("HSN", "visual_projection", "", "R", "R", [0.0, 0.0, 0.0], None, "acetylcholine"),
    7: ("IN1", "vnc_intrinsic", "", "L", "L", [1.0, 1.0, 1.0], None, "gaba"),
    8: ("IN1", "vnc_intrinsic", "", "R", "R", [1.0, 1.0, 1.0], None, "gaba"),
    9: ("b1 MN", "vnc_motor", "wm", "L", "L", [2.0, 2.0, 2.0], None, "glutamate"),
    10: ("b1 MN", "vnc_motor", "wm", "R", "R", [2.0, 2.0, 2.0], None, "glutamate"),
    11: ("b2 MN", "vnc_motor", "wm", "L", "L", [3.0, 3.0, 3.0], None, "glutamate"),
    12: ("b2 MN", "vnc_motor", "wm", "R", "R", [3.0, 3.0, 3.0], None, "glutamate"),
    13: ("b3 MN", "vnc_motor", "wm", "L", "L", None, None, "glutamate"),
    14: ("b3 MN", "vnc_motor", "wm", "R", "R", None, None, "glutamate"),
    15: ("FAR1", "cb_intrinsic", "", "L", "L", None, None, "acetylcholine"),
    16: ("FAR2", "cb_intrinsic", "", "L", "L", None, None, "acetylcholine"),
    17: ("FAR3", "cb_intrinsic", "", "L", "L", None, None, "acetylcholine"),
    18: ("", "", "", None, None, None, None, "acetylcholine"),
    19: ("OUT1", "cb_intrinsic", "", "L", "L", None, None, "acetylcholine"),
}
EDGES = [
    (1, 9, 20), (2, 10, 20),            # halteres onto ipsilateral b1
    (3, 7, 10), (4, 8, 10),             # ocelli onto interneurons
    (5, 7, 6), (6, 8, 6),               # HS onto interneurons
    (7, 10, 10), (8, 9, 10),            # inhibitory interneurons onto contralateral b1
    (7, 11, 8), (8, 12, 8), (7, 13, 6), (8, 14, 6),
    (1, 13, 3),                         # too weak to keep
    (1, 15, 10), (15, 16, 10), (16, 17, 10), (17, 9, 10),  # four connections: too long
    (18, 9, 100),                       # unannotated: ignored entirely
    (19, 9, 10),                        # annotated but off every path: counts toward b1 L's input
]


@pytest.fixture
def synthetic_flight():
    rows = [
        {"bodyId": body, "type": cell_type, "superclass": superclass, "subclass": subclass, "somaSide": soma_side,
         "rootSide": root_side, "somaLocation": soma, "tosomaLocation": tosoma}
        for body, (cell_type, superclass, subclass, soma_side, root_side, soma, tosoma, _) in BODIES.items()
    ]
    annotations = pd.DataFrame(rows)
    edges = pd.DataFrame(EDGES, columns=["body_pre", "body_post", "weight"])
    transmitters = pd.DataFrame({"body": list(BODIES), "consensus_nt": [values[-1] for values in BODIES.values()]})
    return annotations, edges, transmitters


def index_of(arrays, body):
    return int(np.flatnonzero(arrays["body_id"] == body)[0])


def coupling_between(arrays, pre_body, post_body):
    pre, post = index_of(arrays, pre_body), index_of(arrays, post_body)
    match = (arrays["pre"] == pre) & (arrays["post"] == post)
    return float(arrays["coupling"][match][0]) if match.any() else None


def test_transmitter_signs():
    signs = transmitter_signs(pd.Series(["acetylcholine", "gaba", "glutamate", "histamine", "dopamine", None]))
    assert signs.tolist() == [1.0, -1.0, -1.0, -1.0, 0.0, 0.0]


def test_on_paths_keeps_only_short_strong_paths():
    # 0 -> 1 -> 2 is short; 0 -> 3 -> 2 has a weak link; 0 -> 4 -> 5 -> 6 -> 2 is four connections long.
    pre = np.array([0, 1, 0, 3, 0, 4, 5, 6])
    post = np.array([1, 2, 3, 2, 4, 5, 6, 2])
    weight = np.array([9, 9, 9, 1, 9, 9, 9, 9])
    kept = on_paths(pre, post, weight, sources=np.array([0]), targets=np.array([2]), size=7, max_hops=3, min_synapses=5)
    assert kept.tolist() == [True, True, True, False, False, False, False]


def test_circuit_keeps_short_paths_from_sensors_to_wing_motors(synthetic_flight):
    arrays, manifest = build_flight_arrays(*synthetic_flight)
    assert sorted(arrays["body_id"].tolist()) == list(range(1, 15))
    assert manifest["counts"] == {"neurons": 14, "haltere": 2, "ocellar": 2, "hs": 2, "wing_motor": 6, "interneuron": 2}
    assert coupling_between(arrays, 1, 13) is None


def test_couplings_are_signed_shares_of_all_annotated_input(synthetic_flight):
    arrays, _ = build_flight_arrays(*synthetic_flight)
    # b1 L hears haltere L (20), interneuron R (10), FAR3 (10) and OUT1 (10); the unannotated body is ignored.
    assert coupling_between(arrays, 1, 9) == pytest.approx(20 / 50)
    assert coupling_between(arrays, 8, 9) == pytest.approx(-10 / 50)
    assert np.isfinite(arrays["coupling"]).all()


def test_roles_sides_and_amplitude_motors(synthetic_flight):
    arrays, _ = build_flight_arrays(*synthetic_flight)
    role_of = {body: ROLES[arrays["role"][index_of(arrays, body)]] for body in (1, 3, 5, 7, 9)}
    assert role_of == {1: "haltere", 3: "ocellar", 5: "hs", 7: "interneuron", 9: "wing_motor"}
    assert arrays["side"][index_of(arrays, 1)] == -1 and arrays["side"][index_of(arrays, 2)] == 1
    assert [bool(arrays["amplitude"][index_of(arrays, body)]) for body in (9, 11, 13)] == [True, True, False]


def test_sensors_without_soma_fall_back_to_where_they_enter(synthetic_flight):
    arrays, _ = build_flight_arrays(*synthetic_flight)
    assert arrays["soma"][index_of(arrays, 1)].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(arrays["soma"][index_of(arrays, 13)]).all()
