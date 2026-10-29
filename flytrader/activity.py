"""Anatomical context and sampled actual states from the complete running model."""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from .connectome import load_graph
from .provenance import fingerprint, write_json

DISPLAY_LIMIT = 2000


def graph_layout(edges_path, annotations_path, connectivity=None, display_limit=DISPLAY_LIMIT, priority_ids=()):
    graph = connectivity if connectivity is not None else load_graph(edges_path)
    ids = graph.node_ids
    if display_limit < 1:
        raise ValueError("Display limit must be positive")
    displayed = np.arange(len(ids)) if len(ids) <= display_limit else np.sort(
        np.random.default_rng(7).choice(len(ids), display_limit, replace=False))
    if priority_ids and len(ids) > display_limit:
        priority = set(priority_ids)
        selected = np.asarray([i for i, node in enumerate(ids) if node in priority], dtype=int)[:display_limit]
        rest = np.setdiff1d(np.arange(len(ids)), selected)
        displayed = np.sort(np.r_[selected, np.random.default_rng(7).choice(rest, display_limit - len(selected), replace=False)])
    annotations = {}
    context = []
    if Path(annotations_path).exists():
        import pyarrow.feather as feather
        table = feather.read_table(annotations_path, columns=["bodyId", "type", "somaLocation", "somaSide"])
        for row in table.to_pylist():
            position = row["somaLocation"]
            if position is not None and len(position) == 3 and all(math.isfinite(v) for v in position):
                row["position"] = [round(v * .008, 3) for v in position]
                context.append((str(row["bodyId"]), row["position"]))
            else:
                row["position"] = None
            annotations[str(row["bodyId"])] = row
    nodes = []
    for i in displayed:
        node_id = ids[i]
        row = annotations.get(node_id, {})
        nodes.append({"id": node_id, "type": row.get("type") or "Unannotated",
                      "side": row.get("somaSide") or "Unknown", "position": row.get("position")})
    located = sum(n["position"] is not None for n in nodes)
    model_located = sum(annotations.get(n, {}).get("position") is not None for n in ids)
    # The model retains every edge. Only the rendering is sampled and bounded.
    visible = graph.weights[displayed, :][:, displayed].tocoo()
    strongest = np.argsort(visible.data, kind="stable")[-1200:][::-1]
    shown_ids = {n["id"] for n in nodes}
    backdrop = [p for node_id, p in context if node_id not in shown_ids]
    # Cap the gray backdrop so thousands of context dots do not overplot into
    # a solid mass that buries the colored model cells.
    backdrop = backdrop[::max(1, math.ceil(len(backdrop) / 3000))] if located else []
    return {"nodes": nodes, "context": backdrop, "located": located, "unlocated": len(nodes) - located,
            "model_neurons": len(ids), "model_located": model_located, "model_unlocated": len(ids) - model_located,
            "display_indices": displayed.tolist(), "sampled": len(nodes) < len(ids), "sample_seed": 7,
            "sample_strategy": "Dopamine and MBON cells included; remaining cells sampled" if priority_ids else "Uniform sample",
            "edges": [[int(visible.col[i]), int(visible.row[i])] for i in strongest],
            "total_edges": graph.weights.nnz, "layout": "anatomical" if located else "schematic",
            "source": "MaleCNS v1.0 soma positions (micrometres)" if located else "Schematic arrangement; no anatomical positions",
            "source_url": "https://male-cns.janelia.org/download/" if located else None,
            "license": "CC-BY" if located else None,
            "attribution": "MaleCNS collaboration: HHMI Janelia, Cambridge, MRC LMB, Google Research" if located else None,
            "graph_fingerprint": fingerprint(edges_path), "activity_kind": "tanh reservoir state; not spikes",
            "context_note": "Gray background cell bodies are uncolored anatomical context, not individual activity readings"}


class ActivityRecorder:
    def __init__(self, path, edges, annotations, total, delay=0, bars_path=None, connectivity=None, display_limit=DISPLAY_LIMIT,
                 priority_ids=()):
        self.path = Path(path)
        self.live_path = self.path.with_suffix(".live.json")
        self.total = total
        self.delay = min(max(delay, 0), 20 / total)
        self.stride = max(1, math.ceil(total / 400))
        self.last_write = 0
        self.latest = None
        graph = connectivity if connectivity is not None else load_graph(edges)
        self.expected_ids = graph.node_ids
        self.validated_ids = None
        self.payload = {"version": 2, "status": "running", "total": total, "sample_stride": self.stride,
                        "graph": graph_layout(edges, annotations, graph, display_limit, priority_ids), "frames": []}
        self.display_indices = np.asarray(self.payload["graph"]["display_indices"])
        if bars_path:
            bars_path = Path(bars_path)
            with bars_path.open() as stream:
                first = next(csv.DictReader(stream), {})
            metadata = {}
            try:
                metadata = json.loads(bars_path.with_suffix(".metadata.json").read_text())
            except (OSError, ValueError):
                pass
            unit = metadata.get("quote_currency") or ("SOL per token" if "volume_sol" in first else "recording units")
            if first.get("mint") == "SYNTHETIC_NOT_A_TOKEN":
                unit = "synthetic units"
            self.payload["recording"] = {"path": str(bars_path), "asset": first.get("mint", ""), "price_unit": unit}
        write_json(self.path, self.payload)

    def __call__(self, step, bar, ids, values, *, phase=None, paper=None, mint=None, learning=None, coin_index=None,
                 chart_context=None, risk=None, market=None, reward=None):
        if ids is not self.validated_ids:
            if tuple(map(str, ids)) != self.expected_ids:
                raise ValueError("Activity node order does not match the model")
            # Reservoir node IDs are immutable and retain their identity across observations.
            self.validated_ids = ids if isinstance(ids, tuple) else None
        values = np.asarray(values)
        if values.shape != (len(self.expected_ids),):
            raise ValueError("Activity values do not match the full model size")
        frame = {"step": step + 1, "timestamp": float(bar[0]), "price": float(bar[1]),
                 "phase": phase or ("training inputs" if step < int(self.total * .7) else "held-out inputs"),
                 "active_count": int(np.count_nonzero(np.abs(values) > .05)),
                 "max_abs_activation": float(np.max(np.abs(values))),
                 "values": [round(float(v), 6) for v in values[self.display_indices]]}
        if paper is not None:
            frame["paper"] = paper
        if mint is not None:
            frame["mint"] = mint
        if learning is not None:
            frame["learning"] = learning
        if coin_index is not None:
            frame["coin_index"] = coin_index
        if chart_context is not None:
            frame.update(chart_context=chart_context, risk=risk, market=market)
        if reward is not None:
            frame["reward"] = reward
        self.latest = frame
        if step % self.stride == 0 or step + 1 == self.total:
            self.payload["frames"].append(frame)
        now = time.monotonic()
        if now - self.last_write >= .10 or step + 1 == self.total:
            write_json(self.live_path, {"status": "running", "total": self.total, "frame": frame})
            self.last_write = now
        if self.delay:
            time.sleep(self.delay)

    def finish(self, status):
        if self.latest and (not self.payload["frames"] or self.payload["frames"][-1]["step"] != self.latest["step"]):
            self.payload["frames"].append(self.latest)
        self.payload["status"] = status
        write_json(self.path, self.payload)
        write_json(self.live_path, {"status": status, "total": self.total, "frame": self.latest})
