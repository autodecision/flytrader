import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from flytrader.gui import Handler, Workspace
from flytrader.provenance import fingerprint, write_json


class GuiTests(unittest.TestCase):
    def test_pump_link_pin_and_risk_settings_reach_the_trial_worker(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / 'data'; data.mkdir()
            (data / 'graph.csv').write_text('bodyId_pre,bodyId_post,weight\n1,2,3\n')
            workspace = Workspace(root)
            mint = 'A' * 32
            with patch('flytrader.gui.subprocess.Popen') as popen, patch('flytrader.gui.threading.Thread.start'):
                workspace.launch({'action': 'trial', 'edges': 'data/graph.csv', 'source': 'live', 'continuous': True,
                                  'coin_policy': 'pin', 'mint': f'https://pump.fun/coin/{mint}?ref=x',
                                  'min_cap_sol': 200, 'min_reserves_sol': 25})
                command = popen.call_args.args[0]
                self.assertIn('--pin', command)
                self.assertEqual(command[command.index('--mint') + 1], mint)
                self.assertEqual(command[command.index('--min-cap-sol') + 1], '200.0')
                self.assertEqual(command[command.index('--min-reserves-sol') + 1], '25.0')

    def test_pin_without_coin_and_nonfinite_market_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / 'data'; data.mkdir()
            (data / 'graph.csv').write_text('bodyId_pre,bodyId_post,weight\n1,2,3\n')
            workspace = Workspace(root)
            base = {'action': 'trial', 'edges': 'data/graph.csv', 'source': 'live', 'continuous': True}
            for extras in [{'coin_policy': 'pin'}, {'min_cap_sol': 'nan'}, {'min_reserves_sol': -1}, {'mint': 'https://evil.invalid/coin/' + 'A' * 32}]:
                with self.assertRaises(ValueError): workspace.launch({**base, **extras})

    def test_reward_controls_reach_worker_and_finite_reward_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / 'data'; data.mkdir()
            (data / 'graph.csv').write_text('bodyId_pre,bodyId_post,weight\n1,2,3\n')
            workspace = Workspace(root)
            base = {'action': 'trial', 'edges': 'data/graph.csv', 'source': 'live', 'continuous': True,
                    'profile': 'mushroom-body-reward-v1', 'reward_gain': 2, 'exploration': .65}
            for extra in [{'continuous': False}, {'reward_gain': 'nan'}, {'exploration': 1.1}]:
                with self.assertRaises(ValueError): workspace.launch({**base, **extra})
            with patch('flytrader.gui.subprocess.Popen') as popen, patch('flytrader.gui.threading.Thread.start'):
                workspace.launch(base)
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index('--profile') + 1], 'mushroom-body-reward-v1')
                self.assertEqual(command[command.index('--reward-gain') + 1], '2.0')
                self.assertEqual(command[command.index('--exploration') + 1], '0.65')

    def test_path_cannot_escape_data(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root)
            for name in ("../secret", "/etc/passwd", "reports/file.json"):
                with self.assertRaises(ValueError):
                    workspace.path(name)

    def test_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data"
            path.mkdir()
            (path / "escape").symlink_to("/etc")
            with self.assertRaises(ValueError):
                Workspace(root).path("data/escape/passwd")

    def test_bad_action_and_nonfinite_cost_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            w = Workspace(root)
            for action in ("shell", "coinbase"):
                with self.assertRaisesRegex(ValueError, "Unknown action"):
                    w.launch({"action": action})
            with self.assertRaises(ValueError):
                w.launch({"action": "capture", "count": 100, "interval": "nan"})

    def test_public_jobs_are_finite_recorded_replays_and_use_a_prunable_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"; data.mkdir()
            reports = Path(root) / "reports"; reports.mkdir()
            (data / "graph.csv").write_text("bodyId_pre,bodyId_post,weight\n1,2,3\n")
            (data / "prices.csv").write_text(
                "timestamp,mint,close,volume_sol\n" + "".join(f"{i},TEST,1,0\n" for i in range(100)))
            workspace = Workspace(root)
            with patch("flytrader.gui.subprocess.Popen") as popen, patch("flytrader.gui.threading.Thread.start"):
                job = workspace.launch({"action": "replay", "edges": "data/graph.csv", "bars": "data/prices.csv",
                                        "cost": 150, "public": True, "watch": False})
                self.assertTrue(job["output"].startswith("reports/public-replay-"))
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index("--watch-delay") + 1], "0")
            with self.assertRaisesRegex(ValueError, "supports paper trials and replays only"):
                workspace.launch({"action": "capture", "public": True})
            with self.assertRaisesRegex(ValueError, "bounded costs"):
                workspace.launch({"action": "replay", "edges": "data/graph.csv", "bars": "data/prices.csv",
                                  "cost": 150, "public": True, "watch": True})

    def test_public_output_pruning_never_removes_private_reports(self):
        with tempfile.TemporaryDirectory() as root:
            reports = Path(root) / "reports"; reports.mkdir()
            public = reports / "public-replay-old.json"
            private = reports / "trial-private.json"
            public.write_text("public")
            private.write_text("private")
            Workspace(root).prune_public_outputs(max_files=0, max_bytes=0)
            self.assertFalse(public.exists())
            self.assertTrue(private.exists())

    def test_fingerprint_detects_changed_content(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "test.json"
            write_json(path, {"value": 1})
            first = fingerprint(path)
            write_json(path, {"value": 2})
            self.assertNotEqual(first["sha256"], fingerprint(path)["sha256"])
            self.assertEqual(json.loads(path.read_text()), {"value": 2})
            self.assertFalse(path.with_suffix(".json.partial").exists())

    def test_recording_readiness_updates_when_new_prices_arrive(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / "data"
            data.mkdir()
            bars = data / "prices.csv"
            bars.write_text("timestamp,mint,close,volume_sol\n" + "".join(f"{i},TEST,1,0\n" for i in range(99)))
            workspace = Workspace(root)
            self.assertFalse(workspace.file_info(bars)["ready"])
            with bars.open("a") as stream:
                stream.write("99,TEST,1,0\n")
            self.assertTrue(workspace.file_info(bars)["ready"])
            self.assertEqual(workspace.file_info(bars)["count"], 100)

    def test_activity_sidecars_are_not_displayed_as_reports(self):
        with tempfile.TemporaryDirectory() as root:
            reports = Path(root) / "reports"
            reports.mkdir()
            for name in ("test.json", "test.activity.json", "test.activity.live.json"):
                write_json(reports / name, {"mode": "paper-replay"})
            self.assertEqual([r["path"] for r in Workspace(root).inventory()["reports"]], ["reports/test.json"])


class ChartHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        workspace = Workspace(self.directory.name)
        data = Path(self.directory.name) / "data"
        data.mkdir(exist_ok=True)
        self.bars = data / "prices.csv"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.workspace = workspace
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.shutdown)

    def history(self, full=False):
        url = f"http://127.0.0.1:{self.server.server_port}/api/bars?path=data/prices.csv"
        with urlopen(url + ("&full=1" if full else ""), timeout=3) as response:
            return json.load(response)

    def test_full_coin_history_keeps_brief_price_move_and_refreshes(self):
        self.bars.write_text("timestamp,mint,close,volume_sol\n" + "".join(
            f"{1700000000 + i * 10},TEST,{2 if i == 499 else 1},0\n" for i in range(1000)))
        history = self.history(full=True)
        self.assertEqual(history["count"], 1000)
        self.assertEqual(history["steps"], list(range(1, 1001)))
        self.assertEqual(history["points"][499], [1700004990, 2])
        with self.bars.open("a") as stream:
            stream.write("1700010000,TEST,1.5,0\n")
        refreshed = self.history(full=True)
        self.assertEqual(refreshed["count"], 1001)
        self.assertEqual(refreshed["points"][-1], [1700010000, 1.5])
        self.assertEqual(refreshed["steps"][-1], 1001)
        self.assertEqual(refreshed["points"][:-1], history["points"])

    def test_sampled_history_keeps_latest_quote_and_original_frame_numbers(self):
        self.bars.write_text("timestamp,mint,close,volume_sol\n" + "".join(
            f"{1700000000 + i * 10},TEST,{i + 1},0\n" for i in range(1001)))
        history = self.history()
        self.assertLessEqual(len(history["points"]), 401)
        self.assertEqual(history["steps"][0], 1)
        self.assertEqual(history["steps"][-1], 1001)
        self.assertEqual(history["points"][-1], [1700010000, 1001])
        for step, point in zip(history["steps"], history["points"]):
            self.assertEqual(point, [1700000000 + (step - 1) * 10, step])
