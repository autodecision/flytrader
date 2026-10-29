from argparse import Namespace
from contextlib import chdir, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flytrader.context import ChartContext, ContextCoinModel, EXTRA_FEATURES, RiskGuard, context_learner, prior_history
from flytrader.learning import OnlineReadout
from flytrader.market import parse_coin
from flytrader.model import Reservoir
from flytrader.provenance import fingerprint, write_json
from flytrader.trial import PaperBroker, TrialRunner

EDGES = [("1", "2", 3.), ("2", "1", 2.)]
MINT = "A" * 32
MARKET = {"market_cap_sol": 500, "real_sol": 100, "authorities_verified": True,
          "mint_authority": None, "freeze_authority": None, "extensions": [], "largest_account_pct": 3}


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(chdir(self.directory))

    def test_coin_urls_extract_only_a_valid_pump_mint(self):
        for value in [MINT, f'https://pump.fun/coin/{MINT}?ref=test', f'pump.fun/coin/{MINT}', f'https://www.pump.fun/coin/{MINT}/']:
            self.assertEqual(parse_coin(value), MINT)
        for value in ['https://evil.invalid/coin/' + MINT, 'https://pump.fun.evil.invalid/coin/' + MINT,
                      'http://pump.fun/coin/' + MINT, 'https://user@pump.fun/coin/' + MINT,
                      'https://pump.fun/explore', 'https://pump.fun/coin/../secret', 'https://pump.fun:99/coin/' + MINT]:
            with self.assertRaises(ValueError):
                parse_coin(value)

    def test_longer_history_changes_neurons_even_with_the_same_recent_prices(self):
        history_a = [(i * 10, 1 + i * .0001) for i in range(100)]
        history_b = [(i * 10, 2 - i * .008) for i in range(79)] + history_a[79:]
        a = ContextCoinModel(Reservoir(EDGES), None, MINT, history=history_a)
        b = ContextCoinModel(Reservoir(EDGES), None, MINT, history=history_b)
        for i in range(100, 130):
            a.observe((i * 10, 1.02, 0), MARKET)
            b.observe((i * 10, 1.02, 0), MARKET)
        self.assertFalse(np.allclose(a.reservoir.state, b.reservoir.state))
        self.assertEqual(a.context.snapshot['observations'], 130)
        self.assertNotEqual(a.context.snapshot['drawdown_pct'], b.context.snapshot['drawdown_pct'])

    def test_windows_use_time_and_do_not_present_a_gap_as_a_one_minute_return(self):
        context = ChartContext()
        for t in range(0, 901, 10): context.observe(t, 1 + t / 10000)
        self.assertTrue(all(context.snapshot['window_complete'].values()))
        context.observe(2000, 1.2)
        self.assertFalse(context.snapshot['window_complete']['1m'])
        self.assertIsNone(context.snapshot['returns_pct']['1m'])
        self.assertEqual(context.snapshot['first_timestamp'], 0)

    def test_context_learning_is_causal_and_has_separate_persistent_memory(self):
        Path('graph.csv').write_text('bodyId_pre,bodyId_post,weight\n1,2,3\n2,1,2\n')
        original = OnlineReadout.for_reservoir(Reservoir(EDGES), fingerprint('graph.csv'), 10)
        original.save(); original.close()
        old = Path(original.path).read_bytes()
        bars = [(10000 + t * 10, float(np.exp(.002 * t + .03 * np.sin(t))), 0) for t in range(100)]
        results = []
        for name, shift in [('first', False), ('second', True)]:
            reservoir = Reservoir(EDGES)
            learner = context_learner(reservoir, fingerprint('graph.csv'), 10, directory=name)
            model = ContextCoinModel(reservoir, learner, MINT)
            values = []
            for t, bar in enumerate(bars):
                if shift and t >= 80: bar = (bar[0], bar[1] * .4, 0)
                values.append(model.observe(bar, MARKET))
            self.assertEqual(learner.meta['examples'], 78)
            self.assertEqual(learner.size, 6 + EXTRA_FEATURES)
            self.assertLessEqual(len(model.pending), 2)
            learner.close(); results.append(values)
        np.testing.assert_allclose(results[0][59:80], results[1][59:80])
        self.assertNotEqual(results[0][90], results[1][90])
        self.assertEqual(Path(original.path).read_bytes(), old)

    def test_outage_does_not_train_a_long_gap_as_a_normal_interval(self):
        learner = OnlineReadout(6 + EXTRA_FEATURES, 'test.npz', {'profile': 'test'})
        self.addCleanup(learner.close)
        model = ContextCoinModel(Reservoir(EDGES), learner, MINT)
        for t in range(65): model.observe((1000 + t * 10, 1 + .001 * t, 0), MARKET)
        learned = learner.meta['examples']
        model.observe((3000, .1, 0), MARKET)
        self.assertEqual(learner.meta['examples'], learned)
        model.observe((3010, .11, 0), MARKET)
        self.assertEqual(learner.meta['examples'], learned)
        model.observe((3020, .12, 0), MARKET)
        self.assertEqual(learner.meta['examples'], learned + 1)

    def test_prior_history_excludes_future_other_mints_and_unverified_files(self):
        Path('data').mkdir()
        for name, mint, kind in [('real', MINT, 'reserve-price-snapshots'), ('other', 'B' * 32, 'reserve-price-snapshots'), ('context', MINT, 'context-history')]:
            path = Path(f'data/{name}.csv')
            with path.open('w') as stream:
                writer = csv.writer(stream); writer.writerow(['timestamp', 'mint', 'close', 'volume_sol'])
                writer.writerows((i, mint, 100 if i == 123 else 1 + i * .001, 0) for i in range(1000))
            write_json(path.with_suffix('.metadata.json'), {'mint': mint, 'kind': kind})
        points, paths = prior_history(MINT, 900, limit=30)
        self.assertLessEqual(len(points), 30)
        self.assertEqual(points[0][0], 1)
        self.assertEqual(points[-1][0], 899)
        self.assertIn((123, 100), points)
        self.assertEqual(paths, ['data/real.csv'])


class RiskTests(unittest.TestCase):
    def context(self, prices):
        context = ChartContext()
        for i, price in enumerate(prices): context.observe(i * 10, price, MARKET)
        return context.snapshot

    def test_pumping_dumping_reversal_and_cooldown_are_explicit_not_predictions(self):
        for prices, expected in [([1] * 90 + [.8], 'Dumping'), ([1] * 85 + [1.04, 1.08, 1.12, 1.16, 1.2], 'Pumping'),
                                 ([1] * 85 + [1.5, 1.45, 1.4, 1.38, 1.36], 'Pump reversing')]:
            with self.subTest(expected=expected):
                context = self.context(prices)
                guard = RiskGuard()
                risk = guard.assess(context, MARKET, context['last_timestamp'])
                self.assertEqual(risk['trend'], expected)
                self.assertFalse(risk['entry_allowed'])
                self.assertNotIn('probability', risk)

    def test_large_cap_cannot_override_liquidity_controls_or_concentration(self):
        context = self.context([1] * 91)
        for change in [{'real_sol': 1}, {'mint_authority': 'creator'}, {'freeze_authority': 'creator'},
                       {'largest_account_pct': 30}, {'extensions': ['transferHook']}, {'authorities_verified': False},
                       {'liquidity_status': 'inactive'}, {'transfer_fee_authority': 'creator'}]:
            risk = RiskGuard().assess(context, {**MARKET, 'market_cap_sol': 100000, **change}, 900)
            self.assertFalse(risk['entry_allowed'])
            self.assertTrue(risk['reasons'])
        risk = RiskGuard().assess(context, {**MARKET, 'largest_account_pct': None}, 900)
        self.assertTrue(risk['entry_allowed'])
        self.assertEqual(risk['allocation_fraction'], .01)
        self.assertTrue(risk['unknowns'])
        self.assertTrue(RiskGuard().assess(context, {**MARKET, 'extensions': ['transferFeeConfig']}, 900)['entry_allowed'])
        self.assertFalse(RiskGuard().assess(context, {}, 900)['entry_allowed'])

    def test_liquidity_loss_is_visible_even_when_the_last_price_is_flat(self):
        context = ChartContext()
        context.observe(0, 1, MARKET)
        context.observe(10, 1, {**MARKET, 'real_sol': 60})
        risk = RiskGuard().assess(context.snapshot, {**MARKET, 'real_sol': 60}, 10)
        self.assertFalse(risk['entry_allowed'])
        self.assertTrue(any('reserves fell' in reason for reason in risk['reasons']))

    def test_new_risk_cancels_pending_buy_and_exits_at_the_actual_dump_price(self):
        broker = PaperBroker(10, 0)
        broker.observe(0, 0, MINT, 1, .1)
        broker.observe(1, 10, MINT, .5, .1, entry_block_reason='Dumping')
        self.assertFalse(broker.events)
        broker.observe(2, 20, MINT, 1, .1)
        broker.observe(3, 30, MINT, 1, .1, max_order_sol=.02)
        self.assertEqual(broker.events[0]['amount_sol'], .02)
        broker.observe(4, 40, MINT, .1, .1, entry_block_reason='Dumping')
        self.assertEqual(broker.events[-1]['quote_price'], .1)
        self.assertAlmostEqual(broker.snapshot()['pnl_sol'], -.018)


class PinnedTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(chdir(self.directory)); self.enterContext(redirect_stdout(io.StringIO()))
        Path('data').mkdir(); Path('data/graph.csv').write_text('bodyId_pre,bodyId_post,weight\n1,2,3\n2,1,2\n')

    def test_pinned_coin_keeps_state_and_learning_beyond_segment_limit_and_quiet_periods(self):
        args = Namespace(source='live', continuous=True, pin=True, coins=1, count=65, interval=10,
                         mint=f'https://pump.fun/coin/{MINT}', capital=10, cost_bps=150, watch_delay=0,
                         rpc='unused', bars=[], edges='data/graph.csv', out='reports/pinned.json')
        class Feed:
            curve = 'test'
            slot = 0
            snapshot = MARKET
            def read(self):
                if self.slot == 130: raise KeyboardInterrupt()
                self.slot += 1
                return 100000 + self.slot * 10, 1 if self.slot < 100 else 1 + (self.slot - 100) * .001, 0
        feeds = []
        def factory(*_):
            feed = Feed(); feeds.append(feed); return feed
        def no_discovery(_): raise AssertionError('Pinned coin must never switch automatically')
        runner = TrialRunner(args, feed_factory=factory, discover_mint=no_discovery, sleeper=lambda _: None)
        with self.assertRaises(KeyboardInterrupt): runner.run()
        self.assertEqual(len(feeds), 1)
        self.assertEqual(runner.pinned_model.step, 130)
        self.assertEqual(runner.session['observations'], 130)
        self.assertEqual(runner.session['coin_count'], 1)
        self.assertEqual(runner.session['segment_count'], 2)
        self.assertEqual(runner.session['learning']['examples'], 108)
        self.assertEqual({coin['mint'] for coin in runner.session['coins']}, {MINT})
        self.assertEqual(runner.session['coins'][1]['history']['points'], 65)
        first = json.loads(Path(runner.session['coins'][0]['activity_path']).with_suffix('.activity.json').read_text())
        self.assertTrue(any('Pinned coin is quiet' in frame['paper']['decision'] for frame in first['frames']))
        self.assertTrue(all('chart_context' in frame and 'risk' in frame for frame in first['frames']))
        self.assertEqual(runner.session['status'], 'stopped')
