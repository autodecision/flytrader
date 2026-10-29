"""Sparse connectome files with an explicit, stable identity for every neuron."""
from dataclasses import dataclass
import csv
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .provenance import fingerprint, write_json


@dataclass
class Connectivity:
    node_ids: tuple
    weights: sparse.csr_matrix  # [postsynaptic, presynaptic], raw contact counts


def from_edges(edges):
    ids = tuple(sorted({str(n) for a, b, _ in edges for n in (a, b)}))
    if not ids:
        raise ValueError("Connectivity is empty")
    index = {n: i for i, n in enumerate(ids)}
    data = np.asarray([w for _, _, w in edges], dtype=float)
    if not np.isfinite(data).all() or (data <= 0).any():
        raise ValueError("Synapse counts must be finite and positive")
    matrix = sparse.csr_matrix((data, ([index[str(b)] for _, b, _ in edges],
                                      [index[str(a)] for a, _, _ in edges])), shape=(len(ids), len(ids)))
    return Connectivity(ids, matrix)


def load_graph(path):
    path = Path(path)
    if path.suffix != ".npz":
        with path.open(newline="") as stream:
            return from_edges([(r["bodyId_pre"], r["bodyId_post"], float(r["weight"])) for r in csv.DictReader(stream)])
    with np.load(path, allow_pickle=False) as arrays:
        if int(arrays["format_version"]) != 1:
            raise ValueError("Unsupported connectome file version")
        ids = tuple(arrays["node_ids"].astype(str).tolist())
        data, indices, indptr = arrays["data"], arrays["indices"], arrays["indptr"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Connectome neuron identities must be nonempty and unique")
    if data.ndim != 1 or indices.shape != data.shape or indptr.shape != (len(ids) + 1,):
        raise ValueError("Invalid sparse connectome dimensions")
    if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(indptr.dtype, np.integer):
        raise ValueError("Sparse indices must be integers")
    if indptr[0] != 0 or indptr[-1] != len(data) or (np.diff(indptr) < 0).any():
        raise ValueError("Invalid sparse row offsets")
    if (indices < 0).any() or (indices >= len(ids)).any():
        raise ValueError("Sparse connection refers to an unknown neuron")
    if not np.isfinite(data).all() or (data <= 0).any():
        raise ValueError("Synapse counts must be finite and positive")
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(ids), len(ids)))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return Connectivity(ids, matrix)


def save_graph(path, graph):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Compiled connectomes must use an .npz filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    try:
        with temporary.open("wb") as stream:
            # Uncompressed arrays keep repeated loads quick; never deserialize Python objects.
            np.savez(stream, format_version=np.array(1), node_ids=np.asarray(graph.node_ids),
                     data=graph.weights.data, indices=graph.weights.indices, indptr=graph.weights.indptr)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def compile_full(weights_path, annotations_path, out, progress=print):
    """Retain every assigned neuronal superclass and every released internal edge."""
    import pyarrow as pa
    import pyarrow.feather as feather
    started = time.monotonic()
    rows = feather.read_table(annotations_path, columns=["bodyId", "superclass", "status"]).to_pylist()
    selected = [r for r in rows if r["superclass"] and str(r["superclass"]).lower() != "glia"
                and str(r["status"]).lower() != "glia"]
    ids = tuple(sorted(str(r["bodyId"]) for r in selected))
    if not ids:
        raise ValueError("No annotated neuronal superclasses found")
    numeric = np.asarray(ids, dtype=np.int64)
    order = np.argsort(numeric)
    lookup = numeric[order]

    def map_ids(values):
        slots = np.searchsorted(lookup, values)
        safe = np.minimum(slots, len(lookup) - 1)
        valid = (slots < len(lookup)) & (lookup[safe] == values)
        return order[safe].astype(np.int32), valid

    pre_chunks, post_chunks, weight_chunks = [], [], []
    source_rows = 0
    with pa.memory_map(str(weights_path), "r") as source:
        reader = pa.ipc.open_file(source)
        names = reader.schema.names
        pre = "body_pre" if "body_pre" in names else "bodyId_pre"
        post = "body_post" if "body_post" in names else "bodyId_post"
        if not {pre, post, "weight"}.issubset(names):
            raise ValueError("Unexpected connectivity schema")
        progress(f"Retaining {len(ids):,} annotated neurons. Scanning all released connections…")
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            source_rows += batch.num_rows
            pre_indices, pre_valid = map_ids(batch[pre].to_numpy())
            post_indices, post_valid = map_ids(batch[post].to_numpy())
            keep = pre_valid & post_valid
            weights = batch["weight"].to_numpy()[keep]
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Retained synapse counts must be finite and positive")
            pre_chunks.append(pre_indices[keep])
            post_chunks.append(post_indices[keep])
            weight_chunks.append(weights.astype(np.float32))
            if i % max(1, reader.num_record_batches // 10) == 0:
                progress(f"Connectivity scan: {100 * (i + 1) / reader.num_record_batches:.0f}%")
    pre_indices = np.concatenate(pre_chunks)
    post_indices = np.concatenate(post_chunks)
    weights = np.concatenate(weight_chunks)
    del pre_chunks, post_chunks, weight_chunks
    if not len(weights):
        raise ValueError("Retained neurons have no connections")
    progress(f"Building sparse matrix from {len(weights):,} retained connections…")
    matrix = sparse.coo_matrix((weights, (post_indices, pre_indices)), shape=(len(ids), len(ids))).tocsr()
    del pre_indices, post_indices, weights
    graph = Connectivity(ids, matrix)
    connected = (np.diff(matrix.indptr) > 0) | (np.bincount(matrix.indices, minlength=len(ids)) > 0)
    metadata = {"dataset": "male-cns:v1.0", "scope": "full-retained", "format": "sparse-csr-v1",
                "matched_neurons": len(ids), "edges": matrix.nnz,
                "synaptic_contacts": int(matrix.data.sum(dtype=np.float64)),
                "isolated_neurons": int((~connected).sum()), "source_connection_rows": source_rows,
                "selection": "Every assigned neuronal superclass, including uncertain classes; excludes glia and unresolved objects. All internal released edges and self-connections retained.",
                "source": "https://male-cns.janelia.org/download/", "license": "CC-BY",
                "attribution": "MaleCNS collaboration: HHMI Janelia, Cambridge, MRC LMB, Google Research",
                "source_fingerprints": {"weights": fingerprint(weights_path), "annotations": fingerprint(annotations_path)},
                "sparse_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes}
    save_graph(out, graph)
    metadata["graph_fingerprint"] = fingerprint(out)
    metadata["build_seconds"] = round(time.monotonic() - started, 3)
    write_json(Path(out).with_suffix(".metadata.json"), metadata)
    progress(f"Ready: {len(ids):,} neurons, {matrix.nnz:,} directed connections → {out}")
    return metadata
