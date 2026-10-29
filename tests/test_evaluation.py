import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flytrader.evaluation import plan_evaluation, recording_rows, summarize, POLICIES
from flytrader.gui import Workspace
from flytrader.reward_eval import load_split, validate_split


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "data").mkdir()
        (self.root / "reports").mkdir()

    def recording(self, name, mint, start, status="stopped"):
        rows = [{"mint": mint, "bar": [start + i * 10, 1 + i / 100, 0],
                 "market": {"regime": "pumpswap", "quote_reserves_sol": 100 + i,
                            "token_reserves": 100, "real_tokens": 100, "real_sol": 100}} for i in range(20)]
        path = self.root / f"data/{name}.market.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.root / f"reports/trial-{name}.json").write_text(json.dumps({
            "mode": "paper-trial", "status": status, "model_profile": "mushroom-body-reward-v1",
            "coins": [{"market_path": str(path.relative_to(self.root))}]}))
        return path

    def test_split_uses_later_unseen_tokens_excludes_repeats_and_running_sessions(self):
        for i, mint in enumerate("ABCDE"):
            self.recording(mint, mint, i * 1000)
        self.recording("repeat", "A", 6000)
        self.recording("running", "F", 7000, "running")
        plan = plan_evaluation(self.root)
        self.assertTrue(plan["ready"])
        self.assertEqual([r["mint"] for r in plan["train"]], list("ABC"))
        self.assertEqual([r["mint"] for r in plan["test"]], list("DE"))
        self.assertEqual(len(plan["excluded"]), 1)
        self.assertEqual(plan["test_observations"], 40)
        validate_split(load_split(self.root / r["path"] for r in plan["train"]),
                       load_split(self.root / r["path"] for r in plan["test"]))

    def test_empty_single_coin_and_overlapping_recordings_do_not_create_fake_holdout(self):
        self.assertFalse(plan_evaluation(self.root)["ready"])
        a = self.recording("a", "A", 1000)
        b = self.recording("overlap", "B", 1100)
        self.recording("repeat", "A", 2000)
        self.assertFalse(plan_evaluation(self.root)["ready"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_split([a, b])
        with self.assertRaisesRegex(ValueError, "required"):
            validate_split([], [])

    def test_invalid_reserves_and_path_escape_cannot_enter_evaluation(self):
        path = self.recording("bad", "A", 1000)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for reserves in (0, -1, float("nan"), float("inf")):
            rows[3]["market"]["token_reserves"] = reserves
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError): recording_rows(path)
        (self.root / "reports/trial-escape.json").write_text(json.dumps({
            "mode": "paper-trial", "status": "stopped", "model_profile": "mushroom-body-reward-v1",
            "coins": [{"market_path": "../outside.market.jsonl"}]}))
        plan = plan_evaluation(self.root)
        self.assertFalse(plan["ready"])
        self.assertEqual(len(plan["excluded"]), 1)

    def test_verdict_does_not_call_smaller_loss_profit_or_seeds_independent_samples(self):
        scores = {p: {"net_pnl_sol": -.02, "closed_trades": 2, "costs_sol": .01, "marked_drawdown_pct": 1.,
                      "learned_entries": 0, "exploration_entries": 2, "unresolved": 0} for p in (*POLICIES, "cash")}
        scores["no_learning"] = {**scores["no_learning"], "net_pnl_sol": -.03}
        scores["cash"] = {k: 0 for k in scores["cash"]}
        aggregate, summary = summarize([{"policies": scores}], {"test_coins": 2})
        self.assertIn("holding cash still won", summary["verdict"])
        self.assertIn("not independent", summary["note"])
        self.assertAlmostEqual(summary["advantage_vs_untrained_sol"], .01)
        scores["opponent"]["unresolved"] = 1
        _, summary = summarize([{"policies": scores}], {"test_coins": 2})
        self.assertIn("Incomplete exits", summary["verdict"])

    def test_gui_launches_isolated_worker_and_passes_cost_learning_and_risk_settings(self):
        graph = self.root / "data/full.csv"
        graph.write_text("bodyId_pre,bodyId_post,weight\n1,2,3\n")
        graph.with_suffix(".metadata.json").write_text(json.dumps({"scope": "full-retained", "matched_neurons": 2}))
        workspace = Workspace(self.root)
        with patch.object(workspace, "evaluation_plan", return_value={"ready": True}), \
                patch("flytrader.gui.subprocess.Popen") as popen, patch("flytrader.gui.threading.Thread.start"):
            body = {"action": "learning-check", "edges": "data/full.csv", "cost": 175,
                    "reward_gain": 2, "exploration": .65, "min_cap_sol": 200, "min_reserves_sol": 15}
            workspace.launch(body)
            command = popen.call_args.args[0]
            self.assertIn("flytrader.evaluation", command)
            for key, value in (("--cost-bps", "175.0"), ("--reward-gain", "2.0"), ("--exploration", "0.65"),
                               ("--min-cap-sol", "200.0"), ("--min-reserves-sol", "15.0")):
                self.assertEqual(command[command.index(key) + 1], value)
            self.assertNotIn("--continuous", command)
            with self.assertRaisesRegex(ValueError, "current operation"):
                workspace.launch(body)
        with self.assertRaises(ValueError):
            workspace.launch({**body, "cost": "nan"})


if __name__ == "__main__":
    unittest.main()
