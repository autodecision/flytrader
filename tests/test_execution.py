import unittest

from flytrader.execution import ReserveExecution, FillUnavailable
from flytrader.trial import PaperBroker

MARKET = {"regime": "bonding_curve", "quote_reserves_sol": 100., "token_reserves": 1000.,
          "real_sol": 70., "real_tokens": 800.}


class ExecutionTests(unittest.TestCase):
    def test_constant_product_buy_and_shadow_roundtrip_include_all_costs(self):
        book = ReserveExecution(.015)
        buy = book.buy("mint", MARKET, 1)
        net = (1 - .000005) / 1.015
        units = 1000 * net / (100 + net)
        self.assertAlmostEqual(buy["units"], units)
        self.assertGreater(buy["fill_price"], .1 * 1.015)
        sell = book.sell("mint", MARKET, units)
        self.assertAlmostEqual(sell["amount_sol"], net * .985 - .000005)
        self.assertAlmostEqual(1 - sell["amount_sol"], buy["cost_sol"] + sell["cost_sol"])
        self.assertAlmostEqual(book.offsets[("mint", "bonding_curve")][0], 0)
        self.assertAlmostEqual(book.offsets[("mint", "bonding_curve")][1], 0)

    def test_impact_limit_missing_reserves_and_insufficient_exit_have_no_fill(self):
        book = ReserveExecution(.015)
        for market, budget in (({}, 1), (MARKET, 3), ({**MARKET, "real_tokens": .01}, 1)):
            with self.assertRaises(FillUnavailable): book.buy("mint", market, budget)
        self.assertFalse(book.offsets)
        with self.assertRaises(FillUnavailable): book.sell("mint", {**MARKET, "real_sol": .01}, 100)
        self.assertFalse(book.offsets)

    def test_raydium_transfer_fee_reduces_both_buy_and_sell_fills(self):
        market = {**MARKET, "regime": "raydium_clmm", "transfer_fee_bps": 300}
        book = ReserveExecution(0, fixed_fee_sol=0)
        buy = book.buy("mint", market, 1)
        gross_units = 1000 / 101
        self.assertAlmostEqual(buy["units"], gross_units * .97)
        self.assertEqual(buy["transfer_fee_bps"], 300)
        sell = book.sell("mint", market, buy["units"])
        self.assertLess(sell["amount_sol"], 1)
        self.assertEqual(sell["execution_model"], "active-liquidity shadow reserves")

    def test_invalid_transfer_fee_cannot_produce_a_fill(self):
        book = ReserveExecution(0)
        with self.assertRaisesRegex(FillUnavailable, "transfer fee"):
            book.buy("mint", {**MARKET, "transfer_fee_bps": 10_000}, 1)

    def test_mild_price_gain_can_still_lose_after_costs_and_fill_marks_match(self):
        broker = PaperBroker(10, 150, impact_fills=True)
        plan = {"signal_id": "one", "entry_kind": "exploration", "hold_seconds": 30, "allocation_fraction": .01}
        broker.observe(0, 100, "mint", .1, 0, policy_controlled=True, entry_plan=plan, market=MARKET)
        broker.observe(1, 110, "mint", .1, 0, policy_controlled=True, market=MARKET)
        self.assertEqual(broker.trades, 0)
        market = {**MARKET, "quote_reserves_sol": 101.}
        marked = broker.observe(2, 120, "mint", .101, 0, policy_controlled=True, market=market)["equity_sol"]
        sold = broker.observe(3, 140, "mint", .101, 0, policy_controlled=True, market=market)
        self.assertAlmostEqual(marked, sold["equity_sol"])
        self.assertLess(sold["pnl_sol"], 0)
        self.assertEqual(sold["closed_trades"], 1)
        self.assertGreater(sold["costs_paid_sol"], 0)
        self.assertLess(broker.events[-1]["max_adverse_pct"], 0)

    def test_missing_exit_reserves_preserve_unresolved_tokens_and_do_not_sell(self):
        broker = PaperBroker(10, 150, impact_fills=True)
        plan = {"signal_id": "one", "entry_kind": "exploration", "hold_seconds": 30, "allocation_fraction": .01}
        broker.observe(0, 100, "mint", .1, 0, policy_controlled=True, entry_plan=plan, market=MARKET)
        broker.observe(1, 110, "mint", .1, 0, policy_controlled=True, market=MARKET)
        before = broker.snapshot()["equity_sol"]
        after = broker.observe(2, 150, "mint", .01, 0, policy_controlled=True, market={})
        self.assertTrue(after["valuation_stale"])
        self.assertEqual(after["closed_trades"], 0)
        self.assertIsNotNone(after["position"])
        self.assertEqual(after["equity_sol"], before)


if __name__ == "__main__": unittest.main()
