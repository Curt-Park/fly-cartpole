"""Cut the left and right mushroom bodies out of the MaleCNS v1.0 connectome."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .paths import CACHE_DIR, CIRCUIT_PATH, DATA_DIR

if TYPE_CHECKING:
    import pandas as pd

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SOURCE = "MaleCNS v1.0 (HHMI Janelia, Google Research and collaborators), CC BY 4.0"
POPULATIONS = ("pn", "kc", "mbon", "dan")
BLOCKS = ("pn_kc", "kc_mbon", "dan_mbon", "mbon_dan")


def download_file(url: str, target: Path) -> Path:
    """Download through a .part file so an interrupted run is never taken for a complete file."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)
    return target


def select_populations(annotations: pd.DataFrame, side: str = "R") -> dict[str, np.ndarray]:
    cell_class = annotations["class"].fillna("")
    cell_type = annotations["type"].fillna("")
    on_side = annotations["somaSide"].fillna("") == side

    def body_ids(mask) -> np.ndarray:
        return np.sort(annotations.loc[mask, "bodyId"].to_numpy(dtype=np.int64))

    return {
        "pn": body_ids((cell_class == "ALPN") & on_side),
        "kc": body_ids((cell_class == "Kenyon_Cell") & on_side),
        "mbon": body_ids((cell_class == "MBON") & on_side),
        "pam": body_ids(cell_type.str.startswith("PAM")),
        "ppl1": body_ids(cell_type.str.startswith("PPL1")),
    }


def read_edges(weights_path: Path, body_ids: np.ndarray) -> pd.DataFrame:
    """Stream the edge table batch by batch; the full table does not fit comfortably in memory."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    wanted = pa.array(np.unique(body_ids), type=pa.int64())
    reader = ipc.open_file(pa.memory_map(str(weights_path)))
    kept = []
    for index in range(reader.num_record_batches):
        batch = reader.get_batch(index)
        mask = pc.and_(
            pc.is_in(batch["body_pre"], value_set=wanted),
            pc.is_in(batch["body_post"], value_set=wanted),
        )
        kept.append(batch.filter(mask))
    return pa.Table.from_batches(kept, schema=reader.schema).to_pandas()


def connection_block(edges: pd.DataFrame, pre_ids: np.ndarray, post_ids: np.ndarray) -> np.ndarray:
    import pandas as pd

    rows = pd.Index(pre_ids).get_indexer(edges["body_pre"])
    cols = pd.Index(post_ids).get_indexer(edges["body_post"])
    inside = (rows >= 0) & (cols >= 0)
    block = np.zeros((len(pre_ids), len(post_ids)), dtype=np.int32)
    np.add.at(block, (rows[inside], cols[inside]), edges["weight"].to_numpy()[inside])
    return block


def soma_positions(annotations: pd.DataFrame, body_ids: np.ndarray) -> np.ndarray:
    locations = annotations.set_index("bodyId").loc[body_ids, "somaLocation"]
    positions = np.full((len(body_ids), 3), np.nan)
    for row, location in enumerate(locations):
        if isinstance(location, (list, tuple, np.ndarray)) and len(location) == 3:
            positions[row] = np.asarray(location, dtype=np.float64)
    return positions


def build_circuit_arrays(annotations: pd.DataFrame, edges: pd.DataFrame, side: str = "R") -> tuple[dict[str, np.ndarray], dict]:
    populations = select_populations(annotations, side)
    pn, kc_candidates, mbon = populations["pn"], populations["kc"], populations["mbon"]
    dan = np.concatenate([populations["pam"], populations["ppl1"]])
    dan_is_punishment = np.concatenate(
        [np.zeros(len(populations["pam"]), dtype=bool), np.ones(len(populations["ppl1"]), dtype=bool)]
    )

    pn_kc = connection_block(edges, pn, kc_candidates)
    kc_mbon = connection_block(edges, kc_candidates, mbon)
    on_pathway = (pn_kc.sum(axis=0) > 0) & (kc_mbon.sum(axis=1) > 0)
    kc = kc_candidates[on_pathway]
    pn_kc, kc_mbon = pn_kc[:, on_pathway], kc_mbon[on_pathway]

    dan_mbon = connection_block(edges, dan, mbon)
    reaches_mbon = dan_mbon.sum(axis=1) > 0
    dan, dan_mbon, dan_is_punishment = dan[reaches_mbon], dan_mbon[reaches_mbon], dan_is_punishment[reaches_mbon]
    mbon_dan = connection_block(edges, mbon, dan)

    # Punishment dopamine lands on approach compartments, reward dopamine on avoidance ones.
    punishment_input = dan_mbon[dan_is_punishment].sum(axis=0)
    reward_input = dan_mbon[~dan_is_punishment].sum(axis=0)
    mbon_valence = np.sign(punishment_input - reward_input).astype(np.int8)

    pn_types = annotations.set_index("bodyId").loc[pn, "type"].fillna("unknown").astype(str)
    pn_glomerulus = np.array([name.split("_")[0] for name in pn_types])

    arrays: dict[str, np.ndarray] = {
        "pn_kc": pn_kc,
        "kc_mbon": kc_mbon,
        "dan_mbon": dan_mbon,
        "mbon_dan": mbon_dan,
        "pn_glomerulus": pn_glomerulus,
        "mbon_valence": mbon_valence,
        "dan_is_punishment": dan_is_punishment.astype(np.int8),
    }
    for name, body_ids in zip(POPULATIONS, (pn, kc, mbon, dan)):
        arrays[f"{name}_body_id"] = body_ids
        arrays[f"{name}_soma"] = soma_positions(annotations, body_ids)

    pn_per_kc = (pn_kc > 0).sum(axis=0)
    manifest = {
        "source": SOURCE,
        "files": [BASE_URL + ANNOTATIONS_FILE, BASE_URL + WEIGHTS_FILE],
        "counts": {
            "pn": len(pn),
            "kc": len(kc),
            "kc_dropped_off_pathway": int((~on_pathway).sum()),
            "mbon": len(mbon),
            "dan": len(dan),
            "pam": int((~dan_is_punishment).sum()),
            "ppl1": int(dan_is_punishment.sum()),
        },
        "edges": {name: int((arrays[name] > 0).sum()) for name in BLOCKS},
        "synapses": {name: int(arrays[name].sum()) for name in BLOCKS},
        "pn_per_kc": {
            "mean": round(float(pn_per_kc.mean()), 2) if len(kc) else 0.0,
            "median": float(np.median(pn_per_kc)) if len(kc) else 0.0,
        },
        "glomeruli": len(set(pn_glomerulus.tolist())),
        "mbon_valence": {
            "approach": int((mbon_valence > 0).sum()),
            "avoidance": int((mbon_valence < 0).sum()),
            "neutral": int((mbon_valence == 0).sum()),
        },
        "cells_without_soma": {name: int(np.isnan(arrays[f"{name}_soma"][:, 0]).sum()) for name in POPULATIONS},
    }
    return arrays, manifest


def save_circuit(arrays: dict[str, np.ndarray], manifest: dict, path: Path = CIRCUIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def extract(cache_dir: Path = CACHE_DIR, data_dir: Path = DATA_DIR) -> dict[str, dict]:
    import pandas as pd

    annotations_path = download_file(BASE_URL + ANNOTATIONS_FILE, cache_dir / ANNOTATIONS_FILE)
    weights_path = download_file(BASE_URL + WEIGHTS_FILE, cache_dir / WEIGHTS_FILE)
    annotations = pd.read_feather(annotations_path)
    manifests = {}
    for side, name in (("R", "mb_right.npz"), ("L", "mb_left.npz")):
        populations = select_populations(annotations, side)
        edges = read_edges(weights_path, np.concatenate(list(populations.values())))
        arrays, manifests[side] = build_circuit_arrays(annotations, edges, side)
        save_circuit(arrays, manifests[side], data_dir / name)
    return manifests
