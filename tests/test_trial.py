from argparse import Namespace
from contextlib import chdir, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flytrader.activity import ActivityRecorder
from flytrader.gui import Workspace
from flytrader.market import CurveFeed, UnsupportedCurve, pump_creation
from flytrader.model import Reservoir, paper_score, states_for
from flytrader.provenance import write_json
from flytrader.trial import CoinModel, PaperBroker, TrialRunner, pump_recording

MINT = "A" * 32
EDGES = [("1", "2", 3.0), ("2", "1", 2.0)]


class BrokerTests(unittest.TestCase):
    def test_signal_uses_later_entry_and_exit_quotes_including_costs(self):
        broker = PaperBroker(10, 150)
        signal = broker.observe(0, 1, MINT, 1, .1)
        self.assertEqual(broker.events, [])
        self.assertIsNotNone(signal["pending"])
        bought = broker.observe(1, 2, MINT, 2, 0)
        self.assertEqual(broker.events[0]["step"], 2)
        self.assertAlmostEqual(bought["cash_sol"], 9.5)
        units = .5 / (2 * 1.015)
        sold = broker.observe(2, 3, MINT, 3, 0)
        self.assertAlmostEqual(sold["cash_sol"], 9.5 + units * 3 * .985)
        self.assertAlmostEqual(sold["pnl_sol"], sold["realized_sol"])
        self.assertEqual(sold["closed_trades"], 1)
        self.assertIsNone(sold["position"])

    def test_streaming_scores_match_existing_replay_rules(self):
        prices = np.exp(np.random.default_rng(4).normal(0, .12, 100).cumsum())
        predictions = np.random.default_rng(5).normal(.04, .05, 100)
        broker = PaperBroker(1, 150)
        for t, price in enumerate(prices):
            broker.observe(t, t + 1, MINT, price, predictions[t], allow_entry=t + 2 < len(prices))
        expected = paper_score(predictions, prices)
        self.assertEqual(broker.trades, expected["trades"])
        self.assertAlmostEqual(broker.snapshot()["return_pct"], expected["return_pct"])
        self.assertAlmostEqual(broker.drawdown * 100, expected["closed_trade_drawdown_pct"])

    def test_disconnection_cancels_queued_buys_and_keeps_open_positions_unresolved(self):
        broker = PaperBroker()
        broker.observe(0, 1, MINT, 1, .1)
        broker.pause("Unavailable")
        broker.observe(1, 2, MINT, 2, 0)
        self.assertFalse(broker.events)
        broker.observe(2, 3, MINT, 2, .1)
        broker.observe(3, 4, MINT, 2, 0)
        broker.pause("Migrated")
        snapshot = broker.snapshot()
        self.assertTrue(snapshot["valuation_stale"])
        self.assertIsNotNone(snapshot["position"])
        self.assertEqual(snapshot["closed_trades"], 0)
        self.assertEqual(len(broker.events), 1)

    def test_risk_halt_stops_further_entries(self):
        broker = PaperBroker(1, 0)
        for t in range(20):
            broker.observe(3 * t, 3 * t + 1, MINT, 10, 1)
            broker.observe(3 * t + 1, 3 * t + 2, MINT, 10, 0)
            broker.observe(3 * t + 2, 3 * t + 3, MINT, 1, 0)
        self.assertTrue(broker.halted)
        self.assertEqual(broker.trades, 3)
        self.assertIsNone(broker.pending)

    def test_invalid_price_or_cross_coin_fill_rejected(self):
        broker = PaperBroker()
        with self.assertRaises(ValueError):
            broker.observe(0, 1, MINT, float("nan"), 1)
        broker.observe(0, 1, MINT, 1, 1)
        with self.assertRaises(ValueError):
            broker.observe(1, 2, "B" * 32, 1, 0)


class ModelTests(unittest.TestCase):
    def bars(self, count=120):
        rng = np.random.default_rng(8)
        return np.column_stack([np.arange(count) + 1700000000, np.exp(rng.normal(0, .1, count).cumsum()), np.zeros(count)])

    def test_trial_states_equal_original_model_and_training_is_causal(self):
        bars = self.bars()
        model = CoinModel(Reservoir(EDGES), 84)
        actual = []
        predictions = []
        for bar in bars:
            predictions.append(model.observe(bar))
            actual.append(model.reservoir.state.copy())
        original = []
        states_for(bars, Reservoir(EDGES), lambda t, bar, ids, values: original.append(values))
        np.testing.assert_allclose(actual, original)
        self.assertTrue(all(p is None for p in predictions[:84]))
        modified = bars.copy()
        modified[100:, 1] *= 1000
        other = CoinModel(Reservoir(EDGES), 84)
        second = [other.observe(bar) for bar in modified]
        np.testing.assert_allclose(model.beta, other.beta)
        np.testing.assert_allclose(predictions[84:100], second[84:100])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(chdir(self.temp))
        self.enterContext(redirect_stdout(io.StringIO()))
        Path("data").mkdir()
        Path("data/graph.csv").write_text("bodyId_pre,bodyId_post,weight\n1,2,3\n2,1,2\n")

    def recording(self, mint=MINT, name="prices", count=120):
        path = Path(f"data/{name}.csv")
        rng = np.random.default_rng(17)
        prices = np.exp(rng.normal(.025, .14, count).cumsum()) * 1e-7
        with path.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "mint", "close", "volume_sol"])
            writer.writerows((1700000000 + 10 * i, mint, price, 0) for i, price in enumerate(prices))
        write_json(path.with_suffix(".metadata.json"), {"mint": mint, "kind": "reserve-price-snapshots"})
        return str(path)

    def args(self, **kwargs):
        return Namespace(**{**dict(source="recordings", profile="legacy", coins=3, count=120, interval=10, capital=10,
                                   cost_bps=150, watch_delay=0, mint="", rpc="https://unused.invalid",
                                   bars=[self.recording()], edges="data/graph.csv", out="reports/test.json"), **kwargs})

    def test_multi_coin_trial_records_exact_account_and_neuron_frames(self):
        args = self.args()
        args.bars += [self.recording("B" * 32, "second"), self.recording(MINT, "duplicate")]
        session = TrialRunner(args).run()
        self.assertEqual(session["status"], "completed")
        self.assertFalse(session["coin_selection"]["activity_screening"])
        self.assertEqual(session["planned_coins"], 2)
        self.assertEqual(session["observations"], 240)
        for coin in session["coins"]:
            activity = json.loads(Path(coin["activity_path"]).with_suffix(".activity.json").read_text())
            self.assertEqual(len(activity["frames"]), 120)
            for frame in activity["frames"][:84]:
                self.assertIsNone(frame["paper"]["prediction"])
            self.assertTrue(any(f["active_count"] > 0 for f in activity["frames"]))
            self.assertEqual(activity["frames"][-1]["paper"]["closed_trades"],
                             sum(e["side"] == "SELL" for e in session["events"] if e["mint"] in [c["mint"] for c in session["coins"][:coin["index"] + 1]]))
        self.assertTrue(all(e["step"] > 85 for e in session["events"]))
        self.assertIsNone(session["paper"]["position"])
        self.assertAlmostEqual(session["paper"]["cash_sol"] - 10, sum(e["pnl_sol"] for e in session["events"] if e["side"] == "SELL"))

    def test_live_trials_use_fresh_observations_and_persist_prices(self):
        args = self.args(source="live", coins=1, count=65)
        bars = ModelTests().bars(65)
        class Feed:
            curve = "test-curve"
            slot = 0
            def read(self):
                self.slot += 1
                return bars[self.slot - 1]
        runner = TrialRunner(args, feed_factory=lambda *args: Feed(), discover_mint=lambda seen: MINT, sleeper=lambda _: None)
        session = runner.run()
        coin = session["coins"][0]
        self.assertEqual(session["status"], "completed")
        self.assertTrue(session["coin_selection"]["activity_screening"])
        self.assertEqual(session["coin_selection"]["quiet_seconds"], 90)
        self.assertEqual(coin["observations"], 65)
        self.assertEqual(coin["training_count"], 60)
        metadata = json.loads(Path(coin["bars"]).with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["observations"], 65)
        self.assertEqual(coin["last_slot"], 65)
        self.assertTrue(all(event["step"] >= 62 for event in session["events"]))

    def test_feed_failure_retries_without_advancing_model(self):
        args = self.args(source="live", coins=1, mint=MINT)
        class Feed:
            curve = "test-curve"
            slot = 1
            def read(self):
                raise OSError("Offline")
        sleeps = []
        session = TrialRunner(args, feed_factory=lambda *args: Feed(), discover_mint=lambda seen: MINT, sleeper=sleeps.append).run()
        self.assertEqual(session["status"], "failed")
        self.assertEqual(session["observations"], 0)
        self.assertEqual(session["events"], [])
        self.assertEqual(sleeps, [5, 10, 20])

    def test_pinned_unsupported_coin_fails_once_instead_of_retrying_forever(self):
        args = self.args(source="live", continuous=True, pin=True, mint=MINT)
        sleeps = []

        def unsupported(*_):
            raise UnsupportedCurve("Unsupported launchpad")

        session = TrialRunner(args, feed_factory=unsupported, sleeper=sleeps.append).run()
        self.assertEqual(session["status"], "failed")
        self.assertEqual(session["feed_status"], "unsupported")
        self.assertEqual(session["feed_message"], "Unsupported launchpad")
        self.assertEqual(session["coins"][0]["error"], "Unsupported launchpad")
        self.assertEqual(session["observations"], 0)
        self.assertEqual(sleeps, [])

    def test_flat_candidate_is_skipped_before_any_neuron_training(self):
        args = self.args(source="live", coins=1, count=65)
        candidates = iter([MINT, "B" * 32])
        class Feed:
            curve = "test-curve"
            slot = 0
            def __init__(self, mint):
                self.mint = mint
            def read(self):
                self.slot += 1
                return (1700000000 + self.slot * 10, 1 if self.mint == MINT else self.slot, 0)
        session = TrialRunner(args, feed_factory=lambda mint, _: Feed(mint),
                              discover_mint=lambda _: next(candidates), sleeper=lambda _: None).run()
        self.assertEqual(session["coins"][0]["mint"], "B" * 32)
        self.assertEqual(session["observations"], 65)
        self.assertIn("quiet candidate", session["skipped_candidates"][0]["error"])
        self.assertFalse(any(e["mint"] == MINT for e in session["events"]))

    def test_coin_that_goes_quiet_ends_early_with_its_real_prices_and_reason(self):
        args = self.args(source="live", coins=1, count=120)
        class Feed:
            curve = "test-curve"
            slot = 0
            def read(self):
                self.slot += 1
                return (1700000000 + self.slot * 10, min(self.slot, 3), 0)
        session = TrialRunner(args, feed_factory=lambda *_: Feed(), discover_mint=lambda _: MINT,
                              sleeper=lambda _: None).run()
        coin = session["coins"][0]
        self.assertEqual(coin["status"], "quiet")
        self.assertEqual(coin["observations"], 12)
        self.assertEqual(coin["activity"]["seconds_since_change"], 90)
        self.assertIn("90 seconds", coin["rotation_reason"])
        self.assertEqual(session["paper"]["orders"], 0)
        activity = json.loads(Path(coin["activity_path"]).with_suffix(".activity.json").read_text())
        self.assertEqual(activity["frames"][-1]["step"], 12)
        with Path(coin["bars"]).open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)

    def test_auto_discovery_skips_unsupported_coin_before_starting_neurons(self):
        args = self.args(source="live", coins=1, count=65)
        candidates = iter([MINT, "B" * 32])
        bars = ModelTests().bars(65)
        class Feed:
            curve = "test-curve"
            slot = 0
            def read(self):
                self.slot += 1
                return bars[self.slot - 1]
        def factory(mint, endpoint):
            if mint == MINT:
                raise UnsupportedCurve("Migrated")
            return Feed()
        session = TrialRunner(args, feed_factory=factory, discover_mint=lambda seen: next(candidates), sleeper=lambda _: None).run()
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["coins"][0]["mint"], "B" * 32)
        self.assertEqual(session["skipped_candidates"][0]["mint"], MINT)
        self.assertEqual(session["observations"], 65)

    def test_migration_with_open_position_requires_attention_without_a_fake_sale(self):
        args = self.args(source="live", coins=1, count=65)
        class Feed:
            curve = "test-curve"
            slot = 0
            def read(self):
                if self.slot == 62:
                    raise UnsupportedCurve("Curve migrated")
                self.slot += 1
                return (1700000000 + self.slot * 10, float(np.exp(self.slot * .1) * 1e-7), 0)
        session = TrialRunner(args, feed_factory=lambda *args: Feed(), discover_mint=lambda seen: MINT, sleeper=lambda _: None).run()
        self.assertEqual(session["status"], "needs_attention")
        self.assertEqual(session["observations"], 62)
        self.assertEqual([event["side"] for event in session["events"]], ["BUY"])
        self.assertTrue(session["paper"]["valuation_stale"])
        self.assertEqual(session["paper"]["closed_trades"], 0)
        self.assertAlmostEqual(session["paper"]["cash_sol"], 9.5)

    def test_stop_saves_last_frame_and_is_reopenable_after_restart(self):
        args = self.args(watch_delay=.1)
        with self.assertRaises(KeyboardInterrupt):
            TrialRunner(args, sleeper=lambda _: (_ for _ in ()).throw(KeyboardInterrupt())).run()
        report = Workspace(self.temp).trial(args.out)
        self.assertEqual(report["status"], "stopped")
        activity = json.loads(Path(report["activity_path"]).with_suffix(".activity.json").read_text())
        self.assertEqual(activity["status"], "interrupted")
        self.assertEqual(len(activity["frames"]), 1)
        self.assertEqual(len(Workspace(self.temp).inventory()["trials"]), 1)

    def test_finished_or_dead_process_is_not_shown_as_running(self):
        args = self.args()
        session = TrialRunner(args).run()
        session.update(status="running", pid=999999999)
        session["paper"]["pending"] = {"mint": MINT}
        write_json(args.out, session)
        recovered = Workspace(self.temp).trial(args.out)
        self.assertEqual(recovered["status"], "interrupted")
        self.assertIsNone(recovered["paper"]["pending"])

    def test_gui_rejects_invalid_settings_before_starting_a_process(self):
        args = self.args()
        workspace = Workspace(self.temp)
        for kwargs in ({"source": "wallet"}, {"capital": "nan"}, {"coins": 6}, {"interval": 0}, {"cost": "inf"},
                       {"source": "live", "mint": MINT, "coins": 3}):
            with self.assertRaises(ValueError):
                workspace.launch({"action": "trial", "edges": args.edges, **kwargs})
        self.assertEqual(workspace.pump_recordings(), args.bars)

    def test_other_market_recordings_are_not_used_as_pump_prices(self):
        args = self.args()
        path = Path(args.bars[0])
        write_json(path.with_suffix(".metadata.json"), {"product": "BTC-USDC"})
        with self.assertRaises(ValueError):
            pump_recording(path)
        self.assertEqual(Workspace(self.temp).pump_recordings(), [])


class FeedTests(unittest.TestCase):
    def test_discovery_excludes_other_launch_platforms_and_malformed_events(self):
        event = {"pool": "pump", "txType": "create", "mint": MINT}
        self.assertTrue(pump_creation(event))
        for value in ({**event, "pool": "bonk"}, {**event, "mint": None}, {**event, "txType": "buy"}, [], {}):
            self.assertFalse(pump_creation(value))

    def test_same_slot_or_missing_account_never_produces_a_new_price(self):
        feed = CurveFeed.__new__(CurveFeed)
        feed.slot, feed.curve, feed.endpoint, feed.decimals, feed.mint = 5, "curve", "endpoint", 6, MINT
        feed.mode, feed.holder_attempt_at = "curve", float("inf")
        with patch("flytrader.market.rpc", return_value={"context": {"slot": 5}, "value": None}):
            with self.assertRaises(ValueError):
                feed.read()
        with patch("flytrader.market.rpc", return_value={"context": {"slot": 6}, "value": None}), \
             patch("flytrader.market.raydium_sol_pool", side_effect=UnsupportedCurve("No verified native-SOL pool")):
            with self.assertRaisesRegex(UnsupportedCurve, "No verified native-SOL pool"): feed.read()
        self.assertEqual(feed.slot, 5)
