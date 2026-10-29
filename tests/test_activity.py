import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flytrader.activity import ActivityRecorder, graph_layout
from flytrader.model import Reservoir, load_edges, states_for


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.edges = self.root / "edges.csv"
        self.edges.write_text("bodyId_pre,bodyId_post,weight\n2,10,3\n10,2,1\n")
        self.annotations = self.root / "annotations.feather"
        self.output = self.root / "test.activity.json"

    def test_no_positions_means_explicit_schematic(self):
        graph = graph_layout(self.edges, self.annotations)
        self.assertEqual(graph["layout"], "schematic")
        self.assertEqual([n["id"] for n in graph["nodes"]], ["10", "2"])
        self.assertEqual(graph["edges"], [[1, 0], [0, 1]])
        self.assertTrue(all(n["position"] is None for n in graph["nodes"]))
        self.assertEqual(graph["context"], [])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "connectome extra required")
    def test_real_positions_preserve_missing_cells_and_separate_context(self):
        import pyarrow as pa
        import pyarrow.feather as feather
        feather.write_feather(pa.table({"bodyId": [10, 2, 3], "type": ["MBON01", "APL", "Context"],
                                       "somaSide": ["L", "R", "L"],
                                       "somaLocation": [[100, 200, 300], None, [400, 500, 600]]}), self.annotations)
        graph = graph_layout(self.edges, self.annotations)
        self.assertEqual(graph["layout"], "anatomical")
        self.assertEqual(graph["nodes"][0]["position"], [.8, 1.6, 2.4])
        self.assertIsNone(graph["nodes"][1]["position"])
        self.assertEqual((graph["located"], graph["unlocated"]), (1, 1))
        self.assertEqual(graph["context"], [[3.2, 4., 4.8]])

    def test_saved_and_live_frames_match_the_model(self):
        bars = np.array([[i, 1 + i / 1000, 1 + i % 3] for i in range(120)], dtype=float)
        recorder = ActivityRecorder(self.output, self.edges, self.annotations, len(bars))
        states = states_for(bars, Reservoir(load_edges(self.edges)), recorder)
        recorder.finish("completed")
        recording = json.loads(self.output.read_text())
        self.assertEqual(recording["status"], "completed")
        self.assertEqual(len(recording["frames"]), 120)
        np.testing.assert_allclose([f["values"] for f in recording["frames"]], states[:, 4:], rtol=0, atol=5e-7)
        self.assertEqual(recording["frames"][83]["phase"], "training inputs")
        self.assertEqual(recording["frames"][84]["phase"], "held-out inputs")
        live = json.loads(recorder.live_path.read_text())
        self.assertEqual(live["frame"], recording["frames"][-1])
        self.assertEqual(live["status"], "completed")

    def test_wrong_node_order_is_rejected(self):
        recorder = ActivityRecorder(self.output, self.edges, self.annotations, 100)
        with self.assertRaisesRegex(ValueError, "node order"):
            recorder(0, [0, 1, 1], ["2", "10"], [.1, .2])

    def test_recording_preserves_market_price_units(self):
        bars = self.root / "imported-prices.csv"
        bars.write_text("timestamp,mint,close,volume\n0,BTC-USDC,78000,1\n")
        bars.with_suffix(".metadata.json").write_text(json.dumps({"quote_currency": "USDC"}))
        recorder = ActivityRecorder(self.output, self.edges, self.annotations, 100, bars_path=bars)
        self.assertEqual(recorder.payload["recording"]["price_unit"], "USDC")
        self.assertEqual(recorder.payload["recording"]["asset"], "BTC-USDC")

    def test_synthetic_prices_are_labeled_as_synthetic(self):
        bars = self.root / "synthetic.csv"
        bars.write_text("timestamp,mint,close,volume_sol\n0,SYNTHETIC_NOT_A_TOKEN,1,1\n")
        recorder = ActivityRecorder(self.output, self.edges, self.annotations, 100, bars_path=bars)
        self.assertEqual(recorder.payload["recording"]["price_unit"], "synthetic units")

    def test_interruption_keeps_latest_unsampled_frame(self):
        recorder = ActivityRecorder(self.output, self.edges, self.annotations, 1000)
        for step in range(101):
            recorder(step, [step, 1, 1], ["10", "2"], [.1, -.2])
        recorder.finish("interrupted")
        recording = json.loads(self.output.read_text())
        self.assertEqual(recording["status"], "interrupted")
        self.assertEqual(recording["frames"][-1]["step"], 101)
        self.assertEqual(json.loads(recorder.live_path.read_text())["frame"]["step"], 101)

    def test_long_recordings_are_bounded_and_keep_the_last_state(self):
        recorder = ActivityRecorder(self.output, self.edges, self.annotations, 1005)
        for step in range(1005):
            recorder(step, [step, 1, 1], ["10", "2"], [.1, -.2])
        recorder.finish("completed")
        recording = json.loads(self.output.read_text())
        self.assertLessEqual(len(recording["frames"]), 401)
        self.assertEqual(recording["frames"][0]["step"], 1)
        self.assertEqual(recording["frames"][-1]["step"], 1005)
