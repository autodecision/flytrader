import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from flytrader.activity import ActivityRecorder
from flytrader.connectome import Connectivity, compile_full, from_edges, load_graph, save_graph
from flytrader.model import Reservoir, evaluate
from flytrader.gui import Workspace


class ConnectomeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "graph.npz"

    def test_sparse_roundtrip_preserves_ids_and_direction(self):
        original = from_edges([("2", "10", 3), ("10", "2", 1), ("2", "2", 2)])
        save_graph(self.path, original)
        restored = load_graph(self.path)
        self.assertEqual(restored.node_ids, ("10", "2"))
        np.testing.assert_array_equal(restored.weights.toarray(), [[0, 3], [1, 2]])
        left, right = Reservoir(original), Reservoir(restored)
        for _ in range(3):
            np.testing.assert_array_equal(left.step(np.array([.1, -.1, .4])), right.step(np.array([.1, -.1, .4])))

    def test_invalid_sparse_connection_is_rejected(self):
        np.savez(self.path, format_version=1, node_ids=np.array(["a", "b"]), data=np.array([1.]),
                 indices=np.array([2]), indptr=np.array([0, 1, 1]))
        with self.assertRaisesRegex(ValueError, "unknown neuron"):
            load_graph(self.path)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "connectome extra required")
    def test_full_compiler_keeps_uncertain_neurons_weak_edges_self_edges_and_isolates(self):
        import pyarrow as pa
        import pyarrow.feather as feather
        annotations, weights = self.root / "annotations.feather", self.root / "weights.feather"
        feather.write_feather(pa.table({"bodyId": [10, 20, 30, 40, 50, 60],
                                       "superclass": ["cb_intrinsic", "vnc_tbc", "cb_motor", "cb_intrinsic", None, "glia"],
                                       "status": ["Traced", "Anchor", "Traced", "Glia", "Orphan", "Traced"]}), annotations)
        feather.write_feather(pa.table({"body_pre": [10, 20, 10, 10, 40, 50, 60],
                                       "body_post": [20, 10, 10, 20, 10, 10, 10],
                                       "weight": [2, 1, 3, 4, 100, 200, 300]}), weights, chunksize=2)
        metadata = compile_full(weights, annotations, self.path, progress=lambda _: None)
        graph = load_graph(self.path)
        self.assertEqual(graph.node_ids, ("10", "20", "30"))
        np.testing.assert_array_equal(graph.weights.toarray(), [[3, 1, 0], [6, 0, 0], [0, 0, 0]])
        self.assertEqual(metadata["edges"], 3)
        self.assertEqual(metadata["synaptic_contacts"], 10)
        self.assertEqual(metadata["isolated_neurons"], 1)
        self.assertEqual(metadata["source_connection_rows"], 7)
        self.assertEqual(metadata["scope"], "full-retained")

    def test_every_large_graph_cell_contributes_to_readout(self):
        n = 2051
        ids = tuple(str(i) for i in range(n))
        # A directed cycle; the test excites one cell outside the direct readout.
        graph = Connectivity(ids, sparse.csr_matrix((np.ones(n, dtype=np.float32),
                             ((np.arange(n) + 1) % n, np.arange(n))), shape=(n, n)))
        model = Reservoir(graph)
        unselected = next(i for i in range(n) if i not in model.readout_nodes)
        model.inputs[:] = 0
        model.inputs[unselected, 0] = 1
        output = model.step(np.array([1., 0., 0.]))
        self.assertEqual(len(model.state), n)
        self.assertEqual(model.graph.nnz, n)
        np.testing.assert_array_equal(output[4:132], np.zeros(128))
        self.assertGreater(output[132 + unselected % 64], 0)
        model.step(np.zeros(3))
        self.assertGreater(model.state[(unselected + 1) % n], 0)

    def test_display_sampling_keeps_full_model_statistics_and_exact_identity(self):
        graph = from_edges([(str(i), str((i + 1) % 12), 1) for i in range(12)])
        save_graph(self.path, graph)
        recording = ActivityRecorder(self.root / "test.activity.json", self.path, self.root / "missing.feather",
                                     100, connectivity=graph, display_limit=3)
        values = np.linspace(.1, .9, 12)
        recording(0, [0, 1, 1], graph.node_ids, values)
        recording.finish("interrupted")
        data = json.loads(recording.path.read_text())
        self.assertEqual(data["graph"]["model_neurons"], 12)
        self.assertEqual(len(data["graph"]["nodes"]), 3)
        self.assertEqual(data["frames"][0]["active_count"], 12)
        indices = data["graph"]["display_indices"]
        self.assertEqual([node["id"] for node in data["graph"]["nodes"]], [graph.node_ids[i] for i in indices])
        np.testing.assert_allclose(data["frames"][0]["values"], values[indices], rtol=0, atol=5e-7)

    def test_sampling_cannot_change_large_model_paper_results(self):
        n = 2051
        graph = Connectivity(tuple(str(i) for i in range(n)), sparse.eye(n, dtype=np.float32, format="csr"))
        save_graph(self.path, graph)
        bars = np.array([[i, 1 + i / 1000, i % 3] for i in range(100)], dtype=float)
        recorder = ActivityRecorder(self.root / "test.activity.json", self.path, self.root / "missing.feather",
                                    100, connectivity=graph, display_limit=5)
        self.assertEqual(evaluate(bars, graph), evaluate(bars, graph, observer=recorder))
        self.assertEqual(len(recorder.latest["values"]), 5)

    def test_gui_discovers_sparse_graph_and_defaults_to_full_model(self):
        data = self.root / "data"
        data.mkdir()
        full = data / "full.npz"
        save_graph(full, from_edges([("a", "b", 1)]))
        full.with_suffix(".metadata.json").write_text(json.dumps({"scope": "full-retained", "matched_neurons": 2}))
        inventory = Workspace(self.root).inventory()
        self.assertEqual(inventory["default_graph"], "data/full.npz")
        self.assertEqual(inventory["files"][0]["kind"], "edges")
        self.assertIn("full retained", inventory["files"][0]["label"])

    def test_gui_prefers_a_real_graph_over_the_synthetic_demo(self):
        data = self.root / "data"
        data.mkdir()
        (data / "synthetic-edges.csv").write_text("bodyId_pre,bodyId_post,weight\na,b,1\n")
        (data / "malecns-edges.csv").write_text("bodyId_pre,bodyId_post,weight\n1,2,1\n")
        inventory = Workspace(self.root).inventory()
        self.assertEqual(inventory["default_graph"], "data/malecns-edges.csv")
