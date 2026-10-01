"""Cut the fly's flight-stabilisation circuit out of MaleCNS v1.0: halteres, ocelli and HS cells to the wing steering motors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .extract import BASE_URL, ANNOTATIONS_FILE, SOURCE, WEIGHTS_FILE, download_file, read_edges, save_circuit
from .paths import CACHE_DIR, FLIGHT_PATH

if TYPE_CHECKING:
    import pandas as pd

TRANSMITTERS_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
ROLES = ("interneuron", "haltere", "ocellar", "hs", "wing_motor")
SENSOR_ROLES = ("haltere", "ocellar", "hs")
# Basalare motor neurons that raise the wing's stroke amplitude.
AMPLITUDE_MOTORS = ("b1 MN", "b2 MN")
SIGNS = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
SIDES = {"L": -1, "R": 1}


def transmitter_signs(consensus: pd.Series) -> np.ndarray:
    return consensus.astype(str).str.lower().map(SIGNS).fillna(0.0).to_numpy(dtype=np.float64)


def hop_distances(start: np.ndarray, source: np.ndarray, target: np.ndarray, size: int, max_hops: int) -> np.ndarray:
    distance = np.full(size, max_hops + 1)
    distance[start] = 0
    frontier = np.zeros(size, dtype=bool)
    frontier[start] = True
    for hop in range(1, max_hops + 1):
        reached = np.zeros(size, dtype=bool)
        reached[target[frontier[source]]] = True
        reached &= distance > max_hops
        distance[reached] = hop
        frontier = reached
    return distance


def on_paths(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, sources: np.ndarray, targets: np.ndarray,
             size: int, max_hops: int, min_synapses: int) -> np.ndarray:
    """Neurons on some path of at most max_hops strong connections from a source to a target."""
    strong = weight >= min_synapses
    from_sources = hop_distances(sources, pre[strong], post[strong], size, max_hops)
    to_targets = hop_distances(targets, post[strong], pre[strong], size, max_hops)
    return from_sources + to_targets <= max_hops


def location(primary, fallback) -> np.ndarray:
    for value in (primary, fallback):
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
            return np.asarray(value, dtype=np.float64)
    return np.full(3, np.nan)


def side_of(soma_side, root_side) -> int:
    return SIDES.get(soma_side) or SIDES.get(root_side) or 0


def role_codes(neurons: pd.DataFrame) -> np.ndarray:
    cell_type = neurons["type"].fillna("")
    subclass = neurons["subclass"].fillna("")
    roles = np.zeros(len(neurons), dtype=np.int8)
    roles[(subclass == "haltere").to_numpy()] = ROLES.index("haltere")
    roles[cell_type.str.match(r"^OCG").to_numpy()] = ROLES.index("ocellar")
    roles[cell_type.str.match(r"^HS").to_numpy()] = ROLES.index("hs")
    roles[((neurons["superclass"] == "vnc_motor") & (subclass == "wm")).to_numpy()] = ROLES.index("wing_motor")
    return roles


def build_flight_arrays(annotations: pd.DataFrame, edges: pd.DataFrame, transmitters: pd.DataFrame,
                        max_hops: int = 3, min_synapses: int = 5) -> tuple[dict[str, np.ndarray], dict]:
    import pandas as pd

    neurons = annotations[annotations["superclass"].fillna("") != ""].reset_index(drop=True)
    body_index = pd.Index(neurons["bodyId"].to_numpy(dtype=np.int64))
    pre = body_index.get_indexer(edges["body_pre"])
    post = body_index.get_indexer(edges["body_post"])
    annotated = (pre >= 0) & (post >= 0)
    pre, post = pre[annotated], post[annotated]
    weight = edges["weight"].to_numpy()[annotated]
    size = len(neurons)
    # The circuit's share of each neuron's input: every annotated partner counts, kept or not.
    total_input = np.bincount(post, weights=weight, minlength=size)

    roles = role_codes(neurons)
    sensors = np.flatnonzero(np.isin(roles, [ROLES.index(role) for role in SENSOR_ROLES]))
    motors = np.flatnonzero(roles == ROLES.index("wing_motor"))
    kept = np.flatnonzero(on_paths(pre, post, weight, sensors, motors, size, max_hops, min_synapses))
    renumber = np.full(size, -1)
    renumber[kept] = np.arange(len(kept))
    in_circuit = (renumber[pre] >= 0) & (renumber[post] >= 0) & (weight >= min_synapses)

    signs = transmitter_signs(transmitters.set_index("body")["consensus_nt"].reindex(neurons["bodyId"].to_numpy()))
    circuit = neurons.iloc[kept]
    edge_pre, edge_post, edge_weight = pre[in_circuit], post[in_circuit], weight[in_circuit]
    arrays: dict[str, np.ndarray] = {
        "body_id": circuit["bodyId"].to_numpy(dtype=np.int64),
        "type": circuit["type"].fillna("").to_numpy(dtype=str),
        "role": roles[kept],
        "side": np.array([side_of(soma, root) for soma, root in zip(circuit["somaSide"], circuit["rootSide"])], dtype=np.int8),
        "amplitude": circuit["type"].isin(AMPLITUDE_MOTORS).to_numpy(),
        "soma": np.array([location(soma, tosoma) for soma, tosoma in zip(circuit["somaLocation"], circuit["tosomaLocation"])]).reshape(-1, 3),
        "pre": renumber[edge_pre].astype(np.int32),
        "post": renumber[edge_post].astype(np.int32),
        "coupling": signs[edge_pre] * edge_weight / total_input[edge_post],
    }
    manifest = {
        "source": SOURCE,
        "files": [BASE_URL + ANNOTATIONS_FILE, BASE_URL + WEIGHTS_FILE, BASE_URL + TRANSMITTERS_FILE],
        "max_hops": max_hops,
        "min_synapses": min_synapses,
        "counts": {"neurons": len(kept), **{role: int((arrays["role"] == code).sum()) for code, role in enumerate(ROLES)}},
        "connections": int(in_circuit.sum()),
        "synapses": int(edge_weight.sum()),
        "excitatory_neurons": int((signs[kept] > 0).sum()),
        "inhibitory_neurons": int((signs[kept] < 0).sum()),
        "neurons_without_position": int(np.isnan(arrays["soma"][:, 0]).sum()),
    }
    return arrays, manifest


def extract_flight(cache_dir: Path = CACHE_DIR, path: Path = FLIGHT_PATH) -> dict:
    import pandas as pd

    annotations_path = download_file(BASE_URL + ANNOTATIONS_FILE, cache_dir / ANNOTATIONS_FILE)
    weights_path = download_file(BASE_URL + WEIGHTS_FILE, cache_dir / WEIGHTS_FILE)
    transmitters_path = download_file(BASE_URL + TRANSMITTERS_FILE, cache_dir / TRANSMITTERS_FILE)
    annotations = pd.read_feather(annotations_path)
    annotated = annotations.loc[annotations["superclass"].fillna("") != "", "bodyId"].to_numpy(dtype=np.int64)
    arrays, manifest = build_flight_arrays(annotations, read_edges(weights_path, annotated), pd.read_feather(transmitters_path))
    save_circuit(arrays, manifest, path)
    return manifest
