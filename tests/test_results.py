import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flytrader.gui import Workspace
from flytrader.provenance import write_json
from flytrader.results import learning_comparison, new_diagnostics, record_diagnostics, session_summary


class SessionResultTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.report = {"mode": "paper-trial", "status": "stopped", "source": "live", "continuous": True,
                       "started": "2026-09-17T21:00:00+00:00", "finished": "2026-09-17T22:00:00+00:00",
                       "observations": 4, "coin_count": 1, "active_coin": 0, "interval_seconds": 10,
                       "cost_bps_per_side": 150, "graph_fingerprint": {"sha256": "graph"},
                       "paper": {"initial_sol": 10, "cash_sol": 10, "equity_sol": 10, "pnl_sol": 0,
                                 "orders": 0, "closed_trades": 0}, "events": [],
                       "learning": {"enabled": True, "examples": 105, "session_updates": 5, "resumed_examples": 100,
                                    "evaluated": 40, "prediction_mae": .2, "baseline_mae": .1,
                                    "checkpoint": "learning/test.npz"},
                       "coins": [{"index": 0, "activity_path": "reports/test-coin-1.json", "observations": 4}]}
        self.frames = [
            {"step": 1, "price": 1, "paper": {"prediction": None, "decision": "Training — no orders"},
             "learning": {"evaluated": 20, "prediction_mae": .2, "baseline_mae": .1}},
            {"step": 2, "price": 1, "paper": {"prediction": .00024, "decision": "Hold cash — signal below entry threshold"}},
            {"step": 3, "price": 1.01, "paper": {"prediction": -.001, "decision": "Hold cash — signal below entry threshold"}},
            {"step": 4, "price": 1.01, "paper": {"prediction": 0, "decision": "Hold cash — too few observations remain to finish a new trade"}},
        ]
        self.activity = self.root / "reports/test-coin-1.activity.json"
        write_json(self.activity, {"frames": self.frames})
        (self.root / "learning").mkdir()
        self.checkpoint = self.root / "learning/test.npz"
        np.savez(self.checkpoint, metadata=json.dumps({"version": 1, "examples": 105, "identity": {"graph": "graph"}}),
                 gram=np.eye(2), target=np.ones(2))
        self.path = self.root / "reports/test.json"
        write_json(self.path, self.report)

    def test_stopped_session_explains_zero_profit_and_verified_learning(self):
        original = self.path.read_bytes()
        summary = Workspace(self.root).trial("reports/test.json")["summary"]
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(summary["duration_seconds"], 3600)
        self.assertEqual(summary["money"]["pnl_sol"], 0)
        self.assertEqual(summary["money"]["orders"], 0)
        self.assertEqual(summary["learning"]["new_updates"], 5)
        self.assertEqual(summary["learning"]["checkpoint_status"], "verified")
        self.assertEqual(summary["learning"]["comparison"]["assessment"], "worse")
        self.assertEqual(summary["learning"]["comparison"]["scope"], "session")
        self.assertEqual(summary["learning"]["comparison"]["count"], 20)
        self.assertEqual(summary["no_trade_reason"], "below_threshold")
        self.assertEqual(summary["activity"]["forecasts"], 3)
        self.assertEqual(summary["activity"]["max_forecast"], .00024)
        self.assertAlmostEqual(summary["activity"]["unchanged_price_pct"], 200/3)

    def test_resumed_learning_quality_uses_session_errors_not_lifetime_average(self):
        comparison = learning_comparison({"evaluated": 100, "prediction_mae": .5, "baseline_mae": .4},
                                         {"evaluated": 80, "prediction_mae": .6, "baseline_mae": .4})
        self.assertEqual(comparison["assessment"], "better")
        self.assertEqual(comparison["count"], 20)
        self.assertAlmostEqual(comparison["model_error"], .1)
        self.assertEqual(learning_comparison({"evaluated": 0}, None)["assessment"], "insufficient")

    def test_short_or_zero_error_comparisons_do_not_claim_success(self):
        for count in (0, 5, 19):
            result = learning_comparison({"evaluated": count, "prediction_mae": 0, "baseline_mae": .1}, None)
            self.assertEqual(result["assessment"], "insufficient")
        result = learning_comparison({"evaluated": 40, "prediction_mae": .1, "baseline_mae": 0}, None)
        self.assertEqual(result["assessment"], "worse")
        self.assertIsNone(result["error_ratio"])

    def test_sampled_history_does_not_claim_no_signals_ever_qualified(self):
        write_json(self.activity, {"frames": [self.frames[0], self.frames[3]]})
        summary = session_summary(self.root, self.report)
        self.assertEqual(summary["activity"]["coverage"], "sampled")
        self.assertEqual(summary["no_trade_reason"], "sampled_below_threshold")
        self.assertEqual(summary["activity"]["price_intervals"], 0)

    def test_runtime_diagnostics_remain_exact_when_activity_is_sampled(self):
        data = new_diagnostics()
        for step in range(1000):
            record_diagnostics(data, 1, None if step == 0 else 1,
                               {"prediction": .08 if step == 567 else 0, "decision": "Buy queued" if step == 567 else "Hold cash"}, .04)
        self.report.update(diagnostics=data, observations=1000)
        summary = session_summary(self.root, self.report)
        self.assertEqual(summary["activity"]["coverage"], "complete")
        self.assertEqual(summary["activity"]["qualifying_signals"], 1)
        self.assertEqual(summary["activity"]["max_forecast"], .08)
        self.assertEqual(summary["no_trade_reason"], "no_fills")

    def test_risk_pauses_are_explained_separately_from_weak_forecasts(self):
        data = new_diagnostics()
        record_diagnostics(data, 1, None, {'prediction': .1, 'decision': 'Paper entry paused — Rapid price drop'}, .04)
        self.report.update(diagnostics=data, observations=1)
        self.assertEqual(session_summary(self.root, self.report)['no_trade_reason'], 'risk_checks')

    def test_money_totals_use_entire_account_including_open_unresolved_positions(self):
        self.report["paper"].update(orders=602, closed_trades=300, equity_sol=11, cash_sol=9,
                                    pnl_sol=1, realized_sol=-.5, unrealized_sol=1.5, valuation_stale=True,
                                    position={"mint": "A"}, unresolved=[{"mint": "B"}])
        summary = session_summary(self.root, self.report)
        self.assertEqual(summary["money"]["orders"], 602)
        self.assertEqual(summary["money"]["buys"], 302)
        self.assertEqual(summary["money"]["sells"], 300)
        self.assertEqual(summary["money"]["open_positions"], 2)
        self.assertTrue(summary["money"]["valuation_stale"])
        self.assertEqual(summary["money"]["realized_sol"], -.5)

    def test_summary_cache_invalidates_when_checkpoint_or_activity_changes(self):
        workspace = Workspace(self.root)
        first = workspace.trial("reports/test.json")["summary"]
        with patch("flytrader.results.session_summary", side_effect=AssertionError("should be cached")):
            self.assertEqual(workspace.trial("reports/test.json")["summary"], first)
        self.checkpoint.unlink()
        self.assertEqual(workspace.trial("reports/test.json")["summary"]["learning"]["checkpoint_status"], "missing")
        write_json(self.activity, {"frames": []})
        self.assertEqual(workspace.trial("reports/test.json")["summary"]["activity"]["coverage"], "unavailable")

    def test_finite_trials_do_not_claim_persistent_learning(self):
        self.report.pop("learning")
        summary = session_summary(self.root, self.report)
        self.assertFalse(summary["learning"]["enabled"])
        self.assertEqual(summary["learning"]["checkpoint_status"], "not_used")

    def test_legacy_fills_are_counted_and_checkpoint_corruption_is_visible(self):
        self.report["paper"].pop("orders")
        self.report["paper"]["closed_trades"] = 1
        self.report["events"] = [{"side": "BUY"}, {"side": "SELL"}]
        self.checkpoint.write_bytes(b"broken checkpoint")
        summary = session_summary(self.root, self.report)
        self.assertEqual(summary["money"]["buys"], 1)
        self.assertEqual(summary["money"]["sells"], 1)
        self.assertEqual(summary["learning"]["checkpoint_status"], "unverified")
