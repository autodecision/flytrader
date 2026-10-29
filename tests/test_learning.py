from argparse import Namespace
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flytrader.learning import AdaptiveCoinModel, OnlineReadout
from flytrader.market import UnsupportedCurve
from flytrader.model import Reservoir, states_for
from flytrader.trial import PaperBroker, TrialRunner

EDGES = [("1", "2", 3.), ("2", "1", 2.)]


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(chdir(self.temp))

    def learner(self, name="memory.npz", size=6):
        learner = OnlineReadout(size, name, {"model": "test"})
        self.addCleanup(learner.close)
        return learner

    def bars(self, count=120):
        rng = np.random.default_rng(8)
        return np.column_stack([1700000000 + np.arange(count) * 10,
                                np.exp(rng.normal(.01, .08, count).cumsum()), np.zeros(count)])

    def test_online_fit_matches_exponentially_weighted_batch_ridge(self):
        learner = self.learner(size=3)
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=(70, 3)), rng.normal(size=70)
        for t, (state, outcome) in enumerate(zip(x, y)):
            learner.update(state, outcome, prediction=None, timestamp=t + 1, mint="A", signal_step=t)
        weights = .995 ** np.arange(69, -1, -1)
        expected = np.linalg.solve(x.T @ (weights[:, None] * x) + np.eye(3), x.T @ (weights * y))
        np.testing.assert_allclose(learner.beta, expected, atol=1e-12)

    def test_labels_wait_two_prices_and_metrics_score_original_prediction(self):
        learner = self.learner()
        model = AdaptiveCoinModel(Reservoir(EDGES), learner, "A")
        bars = self.bars()
        predictions = []
        for t, bar in enumerate(bars):
            predictions.append(model.observe(bar))
            self.assertEqual(learner.meta["examples"], max(0, t - 21))
        self.assertIsNone(predictions[58])
        self.assertIsNotNone(predictions[59])
        errors, baselines = [], []
        for t in range(20, len(bars) - 2):
            if predictions[t] is not None:
                label = np.log(bars[t + 2, 1] / bars[t + 1, 1])
                errors.append(abs(predictions[t] - label))
                baselines.append(abs(label))
        snapshot = learner.snapshot()
        self.assertEqual(snapshot["evaluated"], len(errors))
        self.assertAlmostEqual(snapshot["prediction_mae"], np.mean(errors))
        self.assertAlmostEqual(snapshot["baseline_mae"], np.mean(baselines))
        self.assertLessEqual(len(model.bars), 21)
        self.assertEqual(len(model.pending), 2)

    def test_future_changes_do_not_affect_earlier_predictions_or_neurons(self):
        bars = self.bars()
        first = AdaptiveCoinModel(Reservoir(EDGES), self.learner("first.npz"), "A")
        predictions, states = [], []
        for bar in bars:
            predictions.append(first.observe(bar))
            states.append(first.reservoir.state.copy())
        original = []
        states_for(bars, Reservoir(EDGES), lambda t, bar, ids, values: original.append(values))
        np.testing.assert_allclose(states, original)
        changed = bars.copy()
        changed[90:, 1] *= 4
        second = AdaptiveCoinModel(Reservoir(EDGES), self.learner("second.npz"), "A")
        other = [second.observe(bar) for bar in changed]
        np.testing.assert_allclose(predictions[59:90], other[59:90])
        self.assertFalse(np.allclose(first.learner.beta, second.learner.beta))

    def test_persistence_and_duplicate_outcomes_are_not_retrained(self):
        learner = self.learner()
        model = AdaptiveCoinModel(Reservoir(EDGES), learner, "A")
        for bar in self.bars():
            model.observe(bar)
        expected, count = learner.beta.copy(), learner.meta["examples"]
        learner.close()
        restored = self.learner()
        np.testing.assert_allclose(restored.beta, expected)
        self.assertEqual(restored.resumed_examples, count)
        replay = AdaptiveCoinModel(Reservoir(EDGES), restored, "A")
        for bar in self.bars():
            replay.observe(bar)
        self.assertEqual(restored.meta["examples"], count)
        self.assertEqual(restored.snapshot()["session_updates"], 0)
        np.testing.assert_allclose(restored.beta, expected)

    def test_only_one_writer_and_model_identity_must_match(self):
        first = self.learner()
        with self.assertRaises(BlockingIOError):
            OnlineReadout(6, "memory.npz", {"model": "test"})
        first.close()
        with self.assertRaises(ValueError):
            OnlineReadout(6, "memory.npz", {"model": "different"})


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(chdir(self.temp))
        self.enterContext(redirect_stdout(io.StringIO()))
        Path("data").mkdir()
        Path("data/graph.csv").write_text("bodyId_pre,bodyId_post,weight\n1,2,3\n2,1,2\n")

    def args(self, **kwargs):
        return Namespace(**{**dict(source="live", profile="legacy", continuous=True, coins=1, count=65, interval=10,
                                   capital=10, cost_bps=150, watch_delay=0, mint="", rpc="unused",
                                   bars=[], edges="data/graph.csv", out="reports/test.json"), **kwargs})

    def runner(self, coins, *, args=None, first_timestamp=1700000000):
        clock = [first_timestamp]
        mints = iter("ABCDEFGHJKLMNPQRSTUVWXYZ"[:coins])
        def discover(seen):
            try:
                return next(mints) * 32
            except StopIteration:
                raise KeyboardInterrupt()
        class Feed:
            curve = "test-curve"
            slot = 0
            def read(self):
                self.slot += 1
                clock[0] += 10
                return clock[0], np.exp(self.slot * .07), 0.
        return TrialRunner(args or self.args(), feed_factory=lambda *a: Feed(),
                           discover_mint=discover, sleeper=lambda _: None)

    def test_runs_beyond_requested_coins_until_stop_and_carries_learning(self):
        runner = self.runner(6)
        with self.assertRaises(KeyboardInterrupt):
            runner.run()
        session = json.loads(Path(runner.args.out).read_text())
        self.assertEqual(session["status"], "stopped")
        self.assertEqual(session["coin_count"], 6)
        self.assertEqual(session["observations"], 390)
        self.assertEqual(session["learning"]["examples"], 6 * 43)
        self.assertGreater(session["paper"]["closed_trades"], 0)
        self.assertIsNone(session["paper"]["position"])
        self.assertIsNone(session["paper"]["pending"])
        second = json.loads(Path(session["coins"][1]["activity_path"]).with_suffix(".activity.json").read_text())
        self.assertIsNotNone(second["frames"][20]["paper"]["prediction"])
        self.assertEqual(second["frames"][20]["learning"]["examples"], 43)
        self.assertEqual(second["frames"][22]["learning"]["examples"], 44)
        ledger = [json.loads(line) for line in Path(session["ledger_path"]).read_text().splitlines()]
        self.assertEqual(len(ledger), session["paper"]["orders"])
        self.assertEqual(len({e["id"] for e in ledger}), len(ledger))
        self.assertTrue(all("coin_index" in e for e in ledger))
        # A later session restores trained parameters rather than starting at zero.
        later = self.runner(1, args=self.args(out="reports/later.json"), first_timestamp=1700100000)
        with self.assertRaises(KeyboardInterrupt):
            later.run()
        self.assertEqual(later.session["learning"]["resumed_examples"], 258)
        self.assertEqual(later.session["learning"]["session_updates"], 43)

    def test_feed_outage_retries_and_recovers_instead_of_ending(self):
        runner = self.runner(1)
        original = runner.discover_mint
        calls = [0]
        def discover(seen):
            calls[0] += 1
            if calls[0] <= 4:
                raise OSError("Offline")
            return original(seen)
        runner.discover_mint = discover
        with self.assertRaises(KeyboardInterrupt):
            runner.run()
        self.assertEqual(runner.session["status"], "stopped")
        self.assertEqual(runner.session["observations"], 65)
        self.assertEqual(runner.session["learning"]["examples"], 43)

    def test_migration_keeps_cash_locked_and_continues_to_next_coin(self):
        runner = self.runner(2)
        original = runner.feed_factory
        def factory(mint, rpc):
            feed = original(mint, rpc)
            read = feed.read
            def interrupted_read():
                if mint == "A" * 32 and feed.slot == 61:
                    raise UnsupportedCurve("Migrated with an open position")
                return read()
            feed.read = interrupted_read
            return feed
        runner.feed_factory = factory
        with self.assertRaises(KeyboardInterrupt):
            runner.run()
        paper = runner.session["paper"]
        self.assertEqual(len(paper["unresolved"]), 1)
        self.assertEqual(paper["unresolved"][0]["mint"], "A" * 32)
        self.assertTrue(paper["valuation_stale"])
        self.assertEqual(runner.session["coin_count"], 2)
        self.assertFalse(any(e["mint"] == "A" * 32 and e["side"] == "SELL" for e in runner.session["events"]))
        self.assertAlmostEqual(paper["cash_sol"], 10 - .5 + paper["realized_sol"])
        self.assertGreater(runner.session["learning"]["examples"], 39)

    def test_saved_trials_cannot_contaminate_live_learning(self):
        with self.assertRaises(ValueError):
            TrialRunner(self.args(source="recordings"))

    def test_failure_to_save_learning_stops_with_visible_error(self):
        runner = self.runner(2)
        with patch.object(OnlineReadout, "save", side_effect=OSError("Checkpoint storage unavailable")):
            with self.assertRaises(OSError):
                runner.run()
        session = json.loads(Path(runner.args.out).read_text())
        self.assertEqual(session["status"], "failed")
        self.assertIn("Checkpoint storage unavailable", session["error"])
        self.assertEqual(session["coin_count"], 1)

    def test_risk_halt_keeps_observing_and_learning(self):
        runner = self.runner(2)
        runner.broker.halted = True
        with self.assertRaises(KeyboardInterrupt):
            runner.run()
        self.assertEqual(runner.session["learning"]["examples"], 86)
        self.assertEqual(runner.session["paper"]["orders"], 0)
        self.assertTrue(runner.session["paper"]["halted"])

    def test_quiet_rotation_cancels_entries_and_closes_due_positions_before_next_coin(self):
        for changing_prices in (3, 4):
            with self.subTest(changing_prices=changing_prices):
                runner = self.runner(2, args=self.args(out=f"reports/quiet-{changing_prices}.json"))
                original = runner.feed_factory
                def factory(mint, rpc):
                    feed = original(mint, rpc)
                    read = feed.read
                    def quiet_read():
                        bar = read()
                        return (bar[0], min(feed.slot, changing_prices), 0) if mint == "A" * 32 else bar
                    feed.read = quiet_read
                    return feed
                runner.feed_factory = factory
                with patch.object(AdaptiveCoinModel, "observe", return_value=1):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.run()
                coin = runner.session["coins"][0]
                self.assertEqual(coin["status"], "quiet")
                self.assertEqual(runner.session["coin_count"], 2)
                self.assertIsNone(runner.session["paper"]["position"])
                self.assertIsNone(runner.session["paper"]["pending"])
                self.assertEqual(runner.session["paper"]["unresolved"], [])
                events = [event for event in runner.session["events"] if event["mint"] == "A" * 32]
                self.assertFalse(any(event["side"] == "BUY" and event["step"] == coin["observations"] for event in events))
                self.assertEqual(sum(event["side"] == "BUY" for event in events), sum(event["side"] == "SELL" for event in events))
                if changing_prices == 4:
                    self.assertEqual(events[-1]["side"], "SELL")
                    self.assertEqual(events[-1]["step"], coin["observations"])

    def test_long_session_bounds_dashboard_history_and_keeps_complete_ledger(self):
        runner = self.runner(1)
        discoveries = [0]
        def discover(seen):
            discoveries[0] += 1
            if discoveries[0] > 52:
                raise KeyboardInterrupt()
            # Valid base58 strings, each a distinct fake test mint.
            alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            return "A" * 31 + alphabet[discoveries[0]]
        runner.discover_mint = discover
        with self.assertRaises(KeyboardInterrupt):
            runner.run()
        session = runner.session
        self.assertEqual(session["coin_count"], 52)
        self.assertEqual(len(session["coins"]), 50)
        self.assertEqual(session["coins"][0]["index"], 2)
        self.assertEqual(session["coins"][session["active_coin"]]["index"], 51)
        self.assertEqual(session["learning"]["examples"], 52 * 43)
        self.assertGreater(session["paper"]["orders"], 500)
        self.assertEqual(len(session["events"]), 500)
        orders = Path(session["ledger_path"]).read_text().splitlines()
        self.assertEqual(len(orders), session["paper"]["orders"])
        self.assertEqual(len(list(Path("reports").glob("*.coin.json"))), 52)
