from argparse import Namespace
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from flytrader.connectome import from_edges
from flytrader.context import ChartContext
from flytrader.reward import MushroomBody, RewardMemory, RewardCoinModel, TrainingGuard, PROFILE
from flytrader.trial import PaperBroker, TrialRunner
from flytrader.results import session_summary
from flytrader.reward_eval import validate_split, run_split, parameter_hash
from flytrader.evaluation import run_check, POLICIES

MINT = "A" * 32
TYPES = {"1": "KCg", "2": "KCab", "3": "PAM01", "4": "PPL101", "5": "MBON01", "6": "MBON02", "7": "other"}
EDGES = [("1", "5", 4.), ("2", "5", 3.), ("1", "6", 2.), ("2", "6", 5.),
         ("3", "5", 2.), ("3", "6", 1.), ("4", "5", 1.), ("4", "6", 2.),
         ("5", "7", 2.), ("7", "3", 2.), ("7", "4", 2.), ("5", "1", 1.)]
MARKET = {"market_cap_sol": 500, "real_sol": 100, "authorities_verified": True,
          "regime": "bonding_curve", "quote_reserves_sol": 100, "token_reserves": 100, "real_tokens": 100,
          "mint_authority": None, "freeze_authority": None, "extensions": [], "largest_account_pct": 3}


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(chdir(self.directory))

    def setup_model(self, name="test", **kwargs):
        brain = MushroomBody(from_edges(EDGES), types=TYPES)
        learner = RewardMemory(brain, f"learning/{name}.npz", {"test": name}, **kwargs)
        self.addCleanup(learner.close)
        return brain, learner, RewardCoinModel(brain, learner, MINT), PaperBroker()

    def tick(self, model, broker, i, price=1, *, risk=None, local_step=None, exit_reason=None):
        timestamp = 1700000000 + i * 10
        prediction = model.observe((timestamp, price, 0), MARKET)
        risk = risk or TrainingGuard().assess(model.context.snapshot, MARKET, timestamp)
        plan = model.plan(timestamp, risk, broker)
        previous = broker.event_count
        broker.observe(i if local_step is None else local_step, timestamp, MINT, price, prediction,
                       policy_controlled=True, entry_plan=plan, exit_reason=exit_reason,
                       entry_block_reason="; ".join(risk["reasons"]) if not risk["entry_allowed"] else None,
                       allocation_fraction=risk["allocation_fraction"], max_order_sol=risk["max_order_sol"])
        model.settle([e for e in broker.events if e["id"] > previous])

    def test_explores_before_forecast_fit_and_learns_only_after_a_real_exit(self):
        brain, learner, model, broker = self.setup_model()
        for i in range(6): self.tick(model, broker, i)
        self.assertIsNotNone(broker.pending)
        self.assertFalse(broker.events)
        self.assertEqual(learner.meta["examples"], 0)
        np.testing.assert_array_equal(learner.weights, 0)
        self.tick(model, broker, 6)
        self.assertEqual(broker.events[0]["entry_kind"], "exploration")
        self.assertAlmostEqual(broker.events[0]["amount_sol"], .1)
        self.assertEqual(learner.meta["examples"], 0)
        self.tick(model, broker, 7)
        self.tick(model, broker, 8)
        self.assertEqual(broker.trades, 0)
        self.tick(model, broker, 9)
        self.assertEqual(learner.meta["examples"], broker.trades)
        self.assertEqual(broker.trades, 1)
        self.assertLess(learner.meta["last_outcome"]["observed"], 0)  # flat price loses both fees
        self.assertAlmostEqual(learner.meta["last_outcome"]["observed"], np.log(.985 / 1.015))
        self.assertGreater(brain.snapshot()["changed_connections"], 0)
        self.assertGreater(brain.pulse["PPL1"], 0)
        self.assertEqual(brain.pulse["PAM"], 0)

    def test_only_real_kc_mbon_edges_change_and_gain_changes_actual_dynamics(self):
        brains = []
        for gain in (0, 1, 2):
            brain = MushroomBody(from_edges(EDGES), types=TYPES)
            for _ in range(6): brain.encode(np.ones(3), np.ones(15))
            old = brain.graph.data.copy()
            head = np.array([0., 1., -1.])
            brain.teach(brain.eligibility(head), .02, gain)
            unchanged = np.ones(len(old), dtype=bool); unchanged[brain.slots] = False
            np.testing.assert_array_equal(brain.graph.data[unchanged], old[unchanged])
            brains.append(brain)
        np.testing.assert_array_equal(brains[0].scales, 1)
        np.testing.assert_allclose(brains[2].scales - 1, 2 * (brains[1].scales - 1))
        brains[1].state.fill(0); brains[2].state.fill(0)
        brains[1].reference.fill(0); brains[2].reference.fill(0)
        brains[1].pulse = {"PAM": 0., "PPL1": 0.}; brains[2].pulse = dict(brains[1].pulse)
        for _ in range(8):
            a = brains[1].encode(np.ones(3), np.ones(15))
            b = brains[2].encode(np.ones(3), np.ones(15))
        self.assertGreater(float(np.max(np.abs(a - b))), 1e-7)

    def test_cancellation_and_unresolved_positions_do_not_earn_rewards(self):
        _, learner, model, broker = self.setup_model()
        for i in range(6): self.tick(model, broker, i)
        broker.pause("feed unavailable")
        self.tick(model, broker, 6, risk={"entry_allowed": False, "reasons": ["Unavailable"], "allocation_fraction": .01, "max_order_sol": 1})
        self.assertFalse(broker.events)
        self.assertFalse(model.pending)
        self.assertEqual(learner.meta["examples"], 0)
        learner.exploration = 1
        self.tick(model, broker, 7); self.tick(model, broker, 8)
        self.assertIsNotNone(broker.position)
        broker.park_unresolved("No quote for a sale")
        self.tick(model, broker, 9)
        self.assertEqual(learner.meta["examples"], 0)

    def test_opponent_depression_changes_the_correct_pathway_and_future_bias(self):
        for error, positive in ((.05, True), (-.05, False)):
            brain = MushroomBody(from_edges(EDGES), types=TYPES)
            for _ in range(12): brain.encode(np.ones(3), np.ones(15))
            brain.teach(brain.eligibility(), error, 1)
            target = brain.avoid if positive else brain.approach
            np.testing.assert_array_equal(brain.scales[~target[brain.post]], 1)
            self.assertTrue(np.any(brain.scales[target[brain.post]] < 1))
            brain.state.fill(0); brain.reference.fill(0)
            brain.pulse = {"PAM": 0., "PPL1": 0.}
            for _ in range(20): brain.encode(np.ones(3), np.ones(15))
            self.assertGreater(brain.bias, 0) if positive else self.assertLess(brain.bias, 0)

    def test_sparse_regime_patterns_use_different_kcs_and_recovery_is_bounded(self):
        types = {**TYPES, **{str(100 + i): "KC" for i in range(100)}}
        edges = EDGES + [(str(100 + i), "5" if i % 2 else "6", 1.) for i in range(100)]
        brain = MushroomBody(from_edges(edges), types=types)
        brain.encode(np.ones(3), np.ones(15), {"regime": "bonding_curve"})
        first = set(brain.snapshot()["active_kc_ids"])
        brain.encode(np.ones(3), np.ones(15), {"regime": "pumpswap"})
        second = set(brain.snapshot()["active_kc_ids"])
        self.assertFalse(first & second)
        self.assertLess(brain.snapshot()["kc_active_fraction"], .06)
        brain.teach(brain.eligibility(), .1, 1)
        old = brain.scales.copy(); brain.recover(3600)
        self.assertTrue(np.all(brain.scales >= old)); self.assertTrue(np.all(brain.scales <= 1))

    def test_timed_position_survives_pinned_segment_step_reset(self):
        _, learner, model, broker = self.setup_model()
        for i in range(7): self.tick(model, broker, i)
        self.tick(model, broker, 7, local_step=0)
        self.tick(model, broker, 8, local_step=1)
        self.assertEqual(broker.trades, 0)
        self.tick(model, broker, 9, local_step=2)
        self.assertEqual(broker.trades, 1)
        self.assertEqual(learner.meta["examples"], 1)

    def test_checkpoint_restores_both_circuit_and_policy_and_rejects_wrong_identity(self):
        brain, learner, model, broker = self.setup_model()
        for i in range(10): self.tick(model, broker, i, 1 + i * .01)
        learner.close()
        other = MushroomBody(from_edges(EDGES), types=TYPES)
        restored = RewardMemory(other, learner.path, learner.identity)
        self.addCleanup(restored.close)
        np.testing.assert_array_equal(brain.scales, other.scales)
        np.testing.assert_array_equal(learner.weights, restored.weights)
        self.assertEqual(restored.resumed_examples, 1)
        self.assertEqual(learner.rng.random(), restored.rng.random())
        restored.close()
        with self.assertRaises(ValueError): RewardMemory(other, learner.path, {"wrong": True})

    def test_future_prices_do_not_change_earlier_actions_or_learning(self):
        histories = []
        for name, changed in (("a", False), ("b", True)):
            _, learner, model, broker = self.setup_model(name)
            history = []
            for i in range(60):
                self.tick(model, broker, i, (1 + .002 * i) * (.5 if changed and i >= 40 else 1))
                history.append((broker.event_count, learner.meta["examples"], model.values.copy()))
            histories.append(history)
        for a, b in zip(histories[0][:40], histories[1][:40]):
            self.assertEqual(a[:2], b[:2]); np.testing.assert_array_equal(a[2], b[2])

    def test_pump_is_a_size_warning_but_controls_and_new_crash_block_entries(self):
        context, guard = ChartContext(), TrainingGuard()
        for i in range(20):
            context.observe(i * 10, 1 + .04 * i, MARKET)
            result = guard.assess(context.snapshot, MARKET, i * 10)
        self.assertTrue(result["entry_allowed"])
        self.assertTrue(result["warnings"])
        self.assertEqual(result["allocation_fraction"], .01)
        self.assertFalse(guard.assess(context.snapshot, {**MARKET, "freeze_authority": "creator"}, 190)["entry_allowed"])
        context.observe(200, .5, MARKET)
        result = guard.assess(context.snapshot, MARKET, 200)
        self.assertFalse(result["entry_allowed"]); self.assertTrue(result["emergency_exit"])
        # An old high is not grounds to restart the cooldown indefinitely.
        for i in range(21, 28):
            context.observe(i * 10, .5, MARKET)
            result = guard.assess(context.snapshot, MARKET, i * 10)
        self.assertTrue(result["entry_allowed"])

    def test_runner_persists_trade_rewards_and_actual_circuit_frames(self):
        Path("data").mkdir()
        Path("data/graph.csv").write_text("bodyId_pre,bodyId_post,weight\n" + "".join(f"{a},{b},{w}\n" for a, b, w in EDGES))
        feather.write_feather(pa.table({"bodyId": list(TYPES), "type": list(TYPES.values()),
                                        "somaLocation": [[1., 2., 3.]] * len(TYPES), "somaSide": ["L"] * len(TYPES)}),
                              "data/malecns-annotations.feather")
        class Feed:
            curve = "test-curve"
            slot = 0
            snapshot = MARKET
            def read(self):
                if self.slot >= 68: raise KeyboardInterrupt()
                self.slot += 1
                self.snapshot = {**MARKET, "quote_reserves_sol": (1 + .006 * self.slot) * 100}
                return (1700000000 + self.slot * 10, 1 + .006 * self.slot, 0)
        args = Namespace(source="live", continuous=True, profile=PROFILE, coins=1, count=65, interval=10,
                         capital=10, cost_bps=150, watch_delay=0, mint=MINT, pin=True, rpc="unused", bars=[],
                         edges="data/graph.csv", out="reports/test.json")
        runner = TrialRunner(args, feed_factory=lambda *args: Feed(), sleeper=lambda _: None)
        with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt): runner.run()
        report = json.loads(Path(args.out).read_text())
        self.assertGreater(report["paper"]["closed_trades"], 1)
        self.assertEqual(report["learning"]["examples"], report["paper"]["closed_trades"])
        self.assertEqual(report["status"], "stopped")
        activity = json.loads(Path(report["coins"][0]["activity_path"]).with_suffix(".activity.json").read_text())
        self.assertEqual(activity["graph"]["sample_strategy"], "Dopamine and MBON cells included; remaining cells sampled")
        self.assertTrue(any(f.get("reward", {}).get("changed_connections", 0) > 0 for f in activity["frames"]))
        summary = session_summary(Path.cwd(), report)
        self.assertEqual(summary["learning"]["checkpoint_status"], "verified")
        self.assertEqual(summary["no_trade_reason"], "traded")

    def test_holdout_is_frozen_and_rejects_token_or_time_leakage(self):
        _, learner, _, _ = self.setup_model()
        train = [[{"mint": MINT, "bar": [100 + 10 * i, 1 + .01 * i, 0],
                   "market": {**MARKET, "quote_reserves_sol": 100 + i}} for i in range(70)]]
        test = [[{"mint": "B" * 32, "bar": [1000 + 10 * i, 1 + .01 * i, 0],
                  "market": {**MARKET, "quote_reserves_sol": 100 + i}} for i in range(70)]]
        validate_split(train, test)
        with self.assertRaises(ValueError): validate_split(train, train)
        with self.assertRaises(ValueError): validate_split(test, train)
        run_split(train, learner, "opponent")
        self.assertGreater(learner.meta["examples"], 0)
        learner.frozen = True
        before = parameter_hash(learner); examples = learner.meta["examples"]
        score = run_split(test, learner, "opponent")
        self.assertGreater(score["closed_trades"], 0)
        self.assertEqual(parameter_hash(learner), before)
        self.assertEqual(learner.meta["examples"], examples)

    def test_learning_check_writes_comparison_without_touching_live_memory(self):
        Path("data").mkdir(); Path("reports").mkdir(); Path("learning").mkdir()
        checkpoint = Path("learning/live.npz"); checkpoint.write_bytes(b"Existing live memory must remain intact")
        graph = Path("data/graph.csv")
        graph.write_text("bodyId_pre,bodyId_post,weight\n" + "".join(f"{a},{b},{w}\n" for a, b, w in EDGES))
        for number, mint in enumerate((MINT, "B" * 32)):
            rows = [{"mint": mint, "bar": [1000 + number * 1000 + i * 10, 1 + .01 * i, 0],
                     "market": {**MARKET, "quote_reserves_sol": 100 + i}} for i in range(70)]
            path = Path(f"data/coin-{number}.market.jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            Path(f"reports/trial-{number}.json").write_text(json.dumps({"mode": "paper-trial", "status": "stopped",
                "model_profile": PROFILE, "coins": [{"market_path": str(path)}]}))
        with patch("flytrader.reward.MushroomBody", side_effect=lambda graph: MushroomBody(graph, types=TYPES)), redirect_stdout(io.StringIO()):
            result = run_check(Path.cwd(), graph, "reports/check.json", seeds=(19,))
        self.assertEqual(checkpoint.read_bytes(), b"Existing live memory must remain intact")
        self.assertEqual(list(Path("learning").iterdir()), [checkpoint])
        self.assertEqual(result["data"]["test_coins"], 1)
        scores = result["runs"][0]["policies"]
        self.assertEqual(set(scores), {*POLICIES, "cash"})
        self.assertTrue(all(s["parameters_frozen"] for s in scores.values()))
        self.assertGreater(scores["opponent"]["training_trades"], 0)
        self.assertEqual(scores["readout_only"]["circuit"]["changed_connections"], 0)
        self.assertTrue(Path("reports/check.json").is_file())
        with self.assertRaisesRegex(ValueError, "new benchmark"):
            run_check(Path.cwd(), graph, "reports/check.json", seeds=(19,))
        with patch("flytrader.reward.MushroomBody", side_effect=lambda graph: MushroomBody(graph, types=TYPES)), \
                patch("flytrader.reward_eval.run_split", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                run_check(Path.cwd(), graph, "reports/interrupted.json", seeds=(19,))
        self.assertFalse(Path("reports/interrupted.json").exists())
        self.assertEqual(checkpoint.read_bytes(), b"Existing live memory must remain intact")
        self.assertEqual(json.loads(Path("reports/check.json").read_text())["summary"], result["summary"])

    def test_readout_ablation_can_act_without_faking_a_learned_opponent_bias(self):
        brain, memory, _, _ = self.setup_model(exploration=0, plasticity=False)
        memory.weights[:, 0] = .05
        memory.meta["head_counts"] = [3, 3, 3]
        memory.frozen = True
        rows = [[{"mint": MINT, "bar": [1000 + i * 10, 1 + .01 * i, 0],
                  "market": {**MARKET, "quote_reserves_sol": 100 + i}} for i in range(70)]]
        before = parameter_hash(memory)
        result = run_split(rows, memory, "readout_only")
        self.assertGreater(result["learned_entries"], 0)
        self.assertEqual(result["exploration_entries"], 0)
        self.assertEqual(parameter_hash(memory), before)
        self.assertAlmostEqual(brain.bias, 0.)


if __name__ == "__main__":
    unittest.main()
