"""A deliberately non-biophysical reservoir with fixed graph connectivity."""
import csv

import numpy as np
from scipy import sparse

from .connectome import Connectivity, from_edges


class Reservoir:
    def __init__(self, edges, seed=7):
        # Preserve direction: a presynaptic node contributes to a postsynaptic node.
        connectivity = edges if isinstance(edges, Connectivity) else from_edges(edges)
        ids = connectivity.node_ids
        self.node_ids = ids
        self.graph = connectivity.weights.copy()
        # A bounded row sum, rather than claiming synapse counts are biological conductances.
        sums = np.asarray(self.graph.sum(axis=1)).ravel()
        self.graph.data *= np.repeat(0.9 / np.maximum(sums, 1), np.diff(self.graph.indptr))
        rng = np.random.default_rng(seed)
        self.inputs = rng.normal(0, 0.3, (len(ids), 3)).astype(self.graph.dtype)
        self.state = np.zeros(len(ids), dtype=self.graph.dtype)
        # Bound readout size even for a large imported graph.
        self.readout_nodes = np.sort(rng.choice(len(ids), min(len(ids), 128), replace=False))
        # Large graphs also pool every cell, so unselected readout cells still contribute.
        self.pool_groups = np.arange(len(ids)) % 64 if len(ids) > 2000 else None
        self.pool_counts = np.bincount(self.pool_groups) if self.pool_groups is not None else None

    def step(self, features):
        if self.pool_groups is None:
            drive = self.inputs @ features
        else:
            drive = self.inputs[:, 0] * features[0] + self.inputs[:, 1] * features[1] + self.inputs[:, 2] * features[2]
        self.state = (0.5 * self.state + 0.5 * np.tanh(self.graph @ self.state + drive)).astype(self.graph.dtype)
        pooled = np.bincount(self.pool_groups, weights=self.state) / self.pool_counts if self.pool_groups is not None else []
        return np.r_[1.0, features, self.state[self.readout_nodes], pooled]


def load_edges(path):
    with open(path, newline="") as stream:
        return [(r["bodyId_pre"], r["bodyId_post"], float(r["weight"])) for r in csv.DictReader(stream)]


def read_bars(path):
    """One mint per file. Strict chronological order; no resampling assumptions."""
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 100:
        raise ValueError("Need at least 100 chronological bars")
    if len({r["mint"] for r in rows}) != 1:
        raise ValueError("Use one mint per file to avoid mixing token histories")
    volume_key = "volume" if "volume" in rows[0] else "volume_sol"
    values = np.array([[float(r[k]) for k in ("timestamp", "close", volume_key)] for r in rows])
    if not np.isfinite(values).all() or (np.diff(values[:, 0]) <= 0).any():
        raise ValueError("Bars must be finite and timestamps strictly increasing")
    if (values[:, 1] <= 0).any() or (values[:, 2] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    return values


def states_for(bars, reservoir, observer=None):
    prices, volumes = bars[:, 1], bars[:, 2]
    result = []
    for t in range(len(bars)):
        ret = np.log(prices[t] / prices[t - 1]) if t else 0.0
        momentum = np.log(prices[t] / prices[max(0, t - 5)])
        prior_volume = np.mean(volumes[max(0, t - 20):t]) if t else volumes[t]
        volume_change = np.log1p(volumes[t]) - np.log1p(prior_volume)
        features = np.clip([ret * 20, momentum * 10, volume_change], -3, 3)
        result.append(reservoir.step(features))
        if observer is not None:
            observer(t, bars[t], reservoir.node_ids, reservoir.state.copy())
    return np.asarray(result)


def paper_score(predictions, prices, cost_bps=150, threshold=0.04):
    """Signal at close t fills at t+1; hold one bar, exit at t+2.

    Nonoverlapping positions; configurable adverse slippage+fees on each side.
    Terminal positions are closed. This is a fill proxy, not bonding-curve execution.
    """
    if not 0 <= cost_bps < 10000 or threshold < 0:
        raise ValueError("Invalid costs or threshold")
    cash, peak, max_drawdown, trades = 1.0, 1.0, 0.0, 0
    cost = cost_bps / 10000
    t = 0
    halted = False
    while t + 2 < len(prices):
        if predictions[t] > threshold and not halted:
            allocation = 0.05 * cash
            units = allocation / (prices[t + 1] * (1 + cost))
            cash += units * prices[t + 2] * (1 - cost) - allocation
            trades += 1
            peak = max(peak, cash)
            max_drawdown = max(max_drawdown, 1 - cash / peak)
            halted = max_drawdown >= 0.1
            t += 2
        else:
            t += 1
    return {"return_pct": (cash - 1) * 100, "closed_trade_drawdown_pct": max_drawdown * 100,
            "trades": trades, "halted": halted}


def evaluate(bars, edges, cost_bps=150, observer=None):
    reservoir = Reservoir(edges)
    states = states_for(bars, reservoir, observer)
    prices = bars[:, 1]
    # Predict the executable interval t+1 -> t+2, not an already observed return.
    labels = np.log(prices[2:] / prices[1:-1])
    split = int(len(bars) * 0.7)
    # Purge labels crossing the chronological split; discard reservoir warmup.
    x, y = states[20:split - 2], labels[20:split - 2]
    regularizer = np.eye(x.shape[1]) * 1.0
    beta = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
    prediction = states[split:] @ beta
    test_prices = prices[split:]
    linear_x = x[:, :4]
    linear_beta = np.linalg.solve(linear_x.T @ linear_x + np.eye(4), linear_x.T @ y)
    return {
        "mode": "paper-replay", "train_bars": split, "test_bars": len(test_prices),
        "neurons": len(reservoir.node_ids), "connections": reservoir.graph.nnz,
        "readout_neurons": len(reservoir.readout_nodes),
        "pooled_readout_groups": 64 if reservoir.pool_groups is not None else 0,
        "cost_bps_per_side": cost_bps, "entry_threshold_log_return": 0.04,
        "allocation_fraction": 0.05,
        "reservoir": paper_score(prediction, test_prices, cost_bps),
        "linear_baseline": paper_score(states[split:, :4] @ linear_beta, test_prices, cost_bps),
        "cash_baseline": {"return_pct": 0.0},
        "limitations": ["Single chronological holdout; not evidence of profitability",
                        "Price-based fill proxy; no liquidity, MEV, failures or curve impact simulation",
                        "Unsigned normalized connectivity; artificial dynamics and input mapping"],
    }
