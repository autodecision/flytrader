import unittest

from flytrader.selection import ExploreDiscovery, QuoteActivity, eligible_candidates, price_changed

NOW = 1700000000
MINT = "A" * 32


def candidate(**changes):
    return {"mint": MINT, "initialized": True, "complete": False, "program": "pump",
            "created_timestamp": (NOW - 300) * 1000, "last_trade_timestamp": (NOW - 5) * 1000,
            "real_sol_reserves": 10_000_000_000, **changes}


class DiscoveryTests(unittest.TestCase):
    def test_context_selection_prefers_larger_caps_but_still_requires_real_curve_reserves(self):
        rows = [candidate(market_cap=150), candidate(mint='B' * 32, market_cap=400),
                candidate(mint='C' * 32, market_cap=10000, real_sol_reserves=1)]
        selected = eligible_candidates(rows, NOW, min_cap=100, min_reserves=5)
        self.assertEqual([row['mint'] for row in selected], ['B' * 32, MINT])
        migrated = candidate(complete=True, market_cap=500, real_sol_reserves=0)
        self.assertEqual(len(eligible_candidates([migrated], NOW, min_cap=100, include_migrated=True)), 1)
        self.assertEqual(eligible_candidates([migrated], NOW, min_cap=100), [])

    def test_only_recent_supported_funded_curves_are_shortlisted(self):
        self.assertEqual(eligible_candidates([candidate()], NOW)[0]["mint"], MINT)
        invalid = [{"complete": True}, {"initialized": False}, {"is_banned": True},
                   {"program": "bonk"}, {"quote_mint": "USDC"}, {"mint": "bad/address"},
                   {"created_timestamp": (NOW - 20) * 1000}, {"last_trade_timestamp": (NOW - 91) * 1000},
                   {"last_trade_timestamp": (NOW + 60) * 1000}, {"real_sol_reserves": 1},
                   {"real_sol_reserves": "nan"}, {"last_trade_timestamp": None}, {"raydium_pool": "pool"},
                   {"pump_swap_pool": "pool"}]
        for change in invalid:
            with self.subTest(change=change):
                self.assertEqual(eligible_candidates([candidate(**change)], NOW), [])
        self.assertEqual(eligible_candidates([candidate(), None, {}], NOW, {MINT}), [])

    def test_cached_candidates_are_rechecked_for_age_and_exclusions(self):
        clock = [NOW]
        calls = []
        rows = [candidate(), candidate(mint="B" * 32, last_trade_timestamp=(NOW - 6) * 1000)]
        def fetch(offset):
            calls.append(offset)
            return rows
        discovery = ExploreDiscovery(fetch=fetch, clock=lambda: clock[0])
        self.assertEqual(discovery(), MINT)
        self.assertEqual(discovery({MINT}), "B" * 32)
        self.assertEqual(calls, [0])
        clock[0] += 100
        with self.assertRaisesRegex(RuntimeError, "No recently traded"):
            discovery()
        self.assertEqual(calls, [0, 0, 50])

    def test_empty_or_failed_catalog_never_falls_back_to_an_unchecked_coin(self):
        with self.assertRaises(RuntimeError):
            ExploreDiscovery(fetch=lambda _: [], clock=lambda: NOW)()
        def offline(_):
            raise OSError("offline")
        with self.assertRaises(OSError):
            ExploreDiscovery(fetch=offline)()


class ActivityTests(unittest.TestCase):
    def test_quiet_coin_is_detected_after_ninety_seconds_not_before(self):
        activity = QuoteActivity()
        for t in range(0, 90, 10):
            self.assertNotEqual(activity.observe(t, 1)["status"], "quiet")
        result = activity.observe(90, 1)
        self.assertEqual(result["status"], "quiet")
        self.assertIn("90 seconds", result["reason"])

    def test_recent_single_move_does_not_disguise_a_mostly_inactive_coin(self):
        activity = QuoteActivity()
        for t in range(0, 121, 10):
            result = activity.observe(t, 1 if t < 80 else 1.01)
        self.assertEqual(result["status"], "quiet")
        self.assertEqual(result["changes_in_window"], 1)

    def test_active_coin_can_stay_for_a_longer_training_run_with_bounded_memory(self):
        activity = QuoteActivity()
        for t in range(0, 10000, 10):
            result = activity.observe(t, 1 + (t // 10 % 2) * .01)
            self.assertNotEqual(result["status"], "quiet")
        self.assertEqual(result["status"], "active")
        self.assertLessEqual(len(activity.changes), 12)

    def test_tiny_rounding_changes_do_not_keep_a_dead_coin_alive_but_accumulated_moves_count(self):
        self.assertFalse(price_changed(1, 1.0000001))
        activity = QuoteActivity()
        activity.observe(0, 1)
        self.assertEqual(activity.observe(10, 1.0000005)["changes_in_window"], 0)
        self.assertEqual(activity.observe(20, 1.000002)["changes_in_window"], 1)

    def test_slower_sampling_allows_enough_quotes_before_assessing_activity(self):
        for interval in (60, 300, 3600):
            with self.subTest(interval=interval):
                activity = QuoteActivity(interval)
                for step in range(20):
                    result = activity.observe(step * interval, 1 + step * .01)
                    self.assertIsNone(result["reason"])
                self.assertEqual(result["status"], "active")
                self.assertLessEqual(len(activity.changes), 4)
                for step in range(1, 4):
                    result = activity.observe((19 + step) * interval, 1.19)
                self.assertEqual(result["status"], "quiet")
                self.assertEqual(result["seconds_since_change"], 3 * interval)

    def test_invalid_or_stale_quotes_are_not_activity(self):
        activity = QuoteActivity()
        activity.observe(10, 1)
        for time, price in [(10, 2), (9, 2), (11, 0), (11, float("nan"))]:
            with self.assertRaises(ValueError):
                activity.observe(time, price)
