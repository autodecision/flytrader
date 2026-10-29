import unittest

import numpy as np

from flytrader.model import Reservoir, evaluate, paper_score, states_for


class ModelTests(unittest.TestCase):
    def test_future_prices_do_not_change_past_states(self):
        bars = np.array([[i, 1 + i / 1000, 1] for i in range(150)], dtype=float)
        edges = [("a", "b", 2), ("b", "a", 1)]
        first = states_for(bars, Reservoir(edges))
        bars[100:, 1:] *= 100
        second = states_for(bars, Reservoir(edges))
        np.testing.assert_array_equal(first[:100], second[:100])

    def test_signal_fills_on_next_bar(self):
        # The jump to entry price is unavailable profit.
        result = paper_score(np.array([1, 0, 0]), np.array([1, 10, 10]), cost_bps=0)
        self.assertEqual(result["return_pct"], 0)
        self.assertEqual(result["trades"], 1)

    def test_both_sides_pay_costs(self):
        result = paper_score(np.ones(3), np.ones(3), cost_bps=100)
        expected = (0.05 * 0.99 / 1.01 - 0.05) * 100
        self.assertAlmostEqual(result["return_pct"], expected)

    def test_terminal_signals_cannot_fill(self):
        self.assertEqual(paper_score(np.array([0, 1, 1]), np.ones(3))["trades"], 0)

    def test_loss_limit_halts_new_positions(self):
        result = paper_score(np.ones(100), np.power(0.5, np.arange(100)), cost_bps=0)
        self.assertTrue(result["halted"])
        self.assertLess(result["trades"], 10)

    def test_graph_direction(self):
        model = Reservoir([("a", "b", 2)])
        np.testing.assert_array_equal(model.graph.toarray(), [[0, 0], [0.9, 0]])

    def test_activity_reports_exact_states_in_node_order(self):
        edges = [("2", "10", 3), ("10", "2", 1)]
        bars = np.array([[i, 1 + i / 1000, 1 + i % 3] for i in range(120)], dtype=float)
        observed = []
        model = Reservoir(edges)
        states = states_for(bars, model, lambda t, bar, ids, values: observed.append((t, tuple(ids), values)))
        self.assertEqual(observed[-1][0], 119)
        self.assertEqual(observed[0][1], ("10", "2"))
        np.testing.assert_array_equal(np.stack([item[2] for item in observed]), states[:, 4:])
        self.assertFalse(np.shares_memory(observed[-1][2], model.state))

    def test_recording_does_not_change_paper_results(self):
        edges = [("a", "b", 2), ("b", "a", 1)]
        bars = np.array([[i, 1 + i / 1000, i % 3] for i in range(120)], dtype=float)
        # An observer receives a copy, so even accidental mutation cannot change the model state.
        def observer(t, bar, ids, values):
            values[:] = 999
        self.assertEqual(evaluate(bars, edges), evaluate(bars, edges, observer=observer))

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            Reservoir([("a", "b", float("nan"))])


if __name__ == "__main__":
    unittest.main()
