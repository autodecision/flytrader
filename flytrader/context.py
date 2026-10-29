"""Causal chart context and a separately versioned learning profile."""
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .learning import OnlineReadout
from .provenance import fingerprint

PROFILE = "chart-context-v1"
EXTRA_FEATURES = 15


def prior_history(mint, before, directory="data", limit=20000):
    """Only previously captured prices, never current listing statistics in the past."""
    prices = {}
    sources = []
    for meta in sorted(Path(directory).glob("*.metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:500]:
        try:
            info = json.loads(meta.read_text())
            if info.get("mint") != mint or info.get("kind") != "reserve-price-snapshots":
                continue
            path = meta.with_name(meta.name.removesuffix(".metadata.json") + ".csv")
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    timestamp, price = float(row["timestamp"]), float(row["close"])
                    if row.get("mint") == mint and 0 < timestamp < before and math.isfinite(price) and price > 0:
                        prices.setdefault(timestamp, price)
            sources.append(str(path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    ordered = sorted(prices.items())
    # Keep the endpoints and evenly sample very long histories; never invent candles.
    if len(ordered) > limit:
        indices = set(np.linspace(0, len(ordered) - 1, max(2, limit - 2), dtype=int))
        indices.update((min(range(len(ordered)), key=lambda i: ordered[i][1]), max(range(len(ordered)), key=lambda i: ordered[i][1])))
        ordered = [ordered[i] for i in sorted(indices)]
    return ordered, sources


class ChartContext:
    """All-history extrema/trend plus time-based 1/5/15 minute windows."""
    def __init__(self, interval=10):
        self.interval = interval
        self.recent = deque()
        self.first_time = self.first_price = self.high = self.low = self.previous = None
        self.last_time = None
        self.count = 0
        self.sum_log = self.sum_log2 = 0.0
        self.liquidity_peak = None
        self.snapshot = {}

    def observe(self, timestamp, price, market=None):
        if not math.isfinite(timestamp) or not math.isfinite(price) or price <= 0 or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("Chart context requires positive prices and increasing finite timestamps")
        market = market or {}
        if self.first_time is None:
            self.first_time, self.first_price = timestamp, price
            self.high = self.low = price
        gap = timestamp - self.last_time if self.last_time is not None else 0
        one = math.log(price / self.previous) if self.previous else 0
        self.high, self.low = max(self.high, price), min(self.low, price)
        self.count += 1
        self.sum_log += one
        self.sum_log2 += one * one
        self.recent.append((timestamp, price))
        # Keep one boundary anchor at or before the longest window.
        while len(self.recent) > 2 and self.recent[1][0] <= timestamp - 900:
            self.recent.popleft()
        returns, coverage = {}, {}
        for label, seconds in (("1m", 60), ("5m", 300), ("15m", 900)):
            anchor = next((p for p in reversed(self.recent) if p[0] <= timestamp - seconds), self.recent[0])
            complete = timestamp - anchor[0] >= seconds and timestamp - seconds - anchor[0] <= max(self.interval * 1.75, seconds * .1)
            coverage[label] = complete
            returns[label] = math.log(price / anchor[1]) if complete else None
        recent_high = max(p for _, p in self.recent)
        recent_low = min(p for _, p in self.recent)
        liquidity = market.get("real_sol")
        if liquidity is not None and math.isfinite(liquidity) and liquidity >= 0:
            self.liquidity_peak = max(self.liquidity_peak or 0, liquidity)
        liquidity_drop = (1 - liquidity / self.liquidity_peak) if liquidity is not None and self.liquidity_peak else None
        total_return = math.log(price / self.first_price)
        drawdown = price / self.high - 1
        deviation = math.sqrt(max(0, self.sum_log2 / self.count - (self.sum_log / self.count) ** 2))
        self.snapshot = {"observations": self.count, "first_timestamp": self.first_time, "last_timestamp": timestamp,
                         "history_seconds": timestamp - self.first_time, "return_since_start_pct": 100 * math.expm1(total_return),
                         "drawdown_pct": 100 * drawdown, "high_price": self.high, "low_price": self.low,
                         "last_change_pct": 100 * math.expm1(one), "volatility_log": deviation,
                         "returns_pct": {k: None if v is None else 100 * math.expm1(v) for k, v in returns.items()},
                         "window_complete": coverage, "recent_runup_pct": 100 * (recent_high / recent_low - 1),
                         "recent_drawdown_pct": 100 * (price / recent_high - 1),
                         "liquidity_drop_pct": None if liquidity_drop is None else 100 * liquidity_drop,
                         "gap_seconds": gap, "interval_seconds": self.interval}
        cap = market.get("market_cap_sol")
        concentration = market.get("largest_account_pct")
        extra = [returns["1m"] or 0, returns["5m"] or 0, returns["15m"] or 0,
                 total_return, drawdown, deviation, (price / recent_high - 1),
                 math.log1p(cap) / 10 if cap is not None and cap >= 0 else 0,
                 math.log1p(liquidity) / 10 if liquidity is not None and liquidity >= 0 else 0,
                 liquidity_drop or 0, concentration / 100 if concentration is not None else 0,
                 float(cap is not None), float(liquidity is not None), float(concentration is not None),
                 min(1, (timestamp - self.first_time) / 900)]
        # Long and short context drive the actual reservoir, as well as the readout.
        drive = [one * 20, ((returns["1m"] or one) + .5 * (returns["5m"] or 0) + .25 * (returns["15m"] or 0)) * 5,
                 total_return + 2 * drawdown - (liquidity_drop or 0)]
        self.previous, self.last_time = price, timestamp
        return np.clip(drive, -3, 3), np.clip(extra, -3, 3)


def context_learner(reservoir, graph_fingerprint, interval, directory="learning"):
    identity = {"profile": PROFILE, "graph": graph_fingerprint["sha256"],
                "dynamics": fingerprint(Path(__file__).with_name("model.py"))["sha256"],
                "context": fingerprint(__file__)["sha256"],
                "learner": fingerprint(Path(__file__).with_name("learning.py"))["sha256"],
                "interval": float(interval), "seed": 7}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    size = 4 + len(reservoir.readout_nodes) + (64 if reservoir.pool_groups is not None else 0) + EXTRA_FEATURES
    return OnlineReadout(size, Path(directory) / f"pump-context-{key}.npz", identity)


class ContextCoinModel:
    """New inputs have separate memory; original checkpoints remain untouched."""
    def __init__(self, reservoir, learner, mint, interval=10, training_count=60, history=()):
        self.reservoir, self.learner, self.mint = reservoir, learner, mint
        reservoir.state.fill(0)
        self.context = ChartContext(interval)
        self.pending = deque()
        self.step = 0
        self.previous = self.last_time = None
        self.training_count, self.training_states, self.training_prices = training_count, [], []
        self.beta = None
        warmup = deque(maxlen=21)
        for timestamp, price in history:
            drive, _ = self.context.observe(timestamp, price)
            warmup.append(drive)
        for drive in warmup:
            reservoir.step(drive)

    def observe(self, bar, market=None):
        timestamp, price = float(bar[0]), float(bar[1])
        drive, extra = self.context.observe(timestamp, price, market)
        state = np.r_[self.reservoir.step(drive), extra]
        t = self.step
        if self.last_time is not None and timestamp - self.last_time > self.context.interval * 1.75:
            self.pending.clear()  # Do not learn a delayed outage return as a normal interval.
        if self.learner:
            if self.pending and self.pending[0][0] == t - 2:
                signal, old_state, prediction = self.pending.popleft()
                self.learner.update(old_state, math.log(price / self.previous), prediction=prediction,
                                    timestamp=timestamp, mint=self.mint, signal_step=signal + 1)
            prediction = float(state @ self.learner.beta) if t >= 20 and self.learner.meta["examples"] >= 38 else None
            if t >= 20:
                self.pending.append((t, state, prediction))
        else:
            prediction = None
            if t < self.training_count:
                self.training_states.append(state)
                self.training_prices.append(price)
                if t + 1 == self.training_count:
                    x = np.asarray(self.training_states)[20:-2]
                    prices = np.asarray(self.training_prices)
                    labels = np.log(prices[2:] / prices[1:-1])[20:]
                    self.beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ labels)
                    self.training_states.clear()
            elif self.beta is not None:
                prediction = float(state @ self.beta)
        self.previous, self.last_time = price, timestamp
        self.step += 1
        return prediction


class RiskGuard:
    """Observable risk rules, never a probability or guarantee about a future rug."""
    def __init__(self, min_cap=100, min_reserves=10):
        self.min_cap, self.min_reserves = min_cap, min_reserves
        self.cooldown_until = 0

    def assess(self, context, market, timestamp, *, live=True):
        reasons, unknown = [], []
        changes = context["returns_pct"]
        dumping = context["last_change_pct"] <= -8 or (changes["1m"] is not None and changes["1m"] <= -12)
        pumping = (changes["1m"] is not None and changes["1m"] >= 15) or (changes["5m"] is not None and changes["5m"] >= 35)
        reversal = context["recent_runup_pct"] >= 25 and context["recent_drawdown_pct"] <= -8
        if dumping: reasons.append("Rapid price drop: possible dump")
        if pumping: reasons.append("Rapid pump: entries paused to avoid chasing a spike")
        if reversal: reasons.append("Large run-up followed by a reversal")
        if context["drawdown_pct"] <= -35: reasons.append("Price is at least 35% below the observed high")
        if (context["liquidity_drop_pct"] or 0) >= 25: reasons.append("Observed SOL reserves fell at least 25% from their high")
        if dumping or reversal:
            self.cooldown_until = max(self.cooldown_until, timestamp + max(120, context["interval_seconds"] * 3))
        if timestamp < self.cooldown_until: reasons.append("Waiting for the reversal cooldown to finish")
        cap, reserves = market.get("market_cap_sol"), market.get("real_sol")
        if live:
            if market.get("liquidity_status") == "inactive":
                reasons.append("Raydium pool has no active liquidity; paper fills are blocked")
            for label, value, minimum in (("Market cap", cap, self.min_cap), ("Real SOL reserves", reserves, self.min_reserves)):
                if value is None:
                    reasons.append(f"{label} is unknown; paper entries paused")
                elif value <= 0 or value < minimum:
                    reasons.append(f"{label} below the {minimum:g} SOL minimum")
            if not market.get("authorities_verified"):
                reasons.append("Token mint/freeze controls have not been verified")
            elif market.get("mint_authority") or market.get("freeze_authority"):
                reasons.append("An active mint or freeze authority is present")
            risky_extensions = set(market.get("extensions", [])) - {"metadataPointer", "tokenMetadata", "immutableOwner", "transferFeeConfig"}
            if risky_extensions: reasons.append("Token extensions need review: " + ", ".join(sorted(risky_extensions)))
            if market.get("transfer_fee_authority"):
                reasons.append("An active authority can change the token transfer fee")
            largest = market.get("largest_account_pct")
            if largest is None: unknown.append("Ownership concentration unknown; smaller paper allocation")
            elif largest >= 20: reasons.append("One non-curve token account holds at least 20% of minted supply")
        else:
            unknown.append("Historical recording has no verified past token controls or reserves")
        if not all(context["window_complete"].values()): unknown.append("Longer chart windows are still filling")
        trend = "Dumping" if dumping else "Pump reversing" if reversal else "Pumping" if pumping else "No rapid move detected"
        return {"trend": trend, "entry_allowed": not reasons, "reasons": reasons, "unknowns": unknown,
                "label": "Entries paused" if reasons else "Caution · incomplete checks" if unknown else "No configured warning triggered",
                "cooldown_seconds": max(0, self.cooldown_until - timestamp),
                "allocation_fraction": .01 if unknown else .05,
                "max_order_sol": max(0, reserves * .01) if reserves is not None else None,
                "limitations": "Heuristics can miss sudden dumps and coordinated selling. Market cap is not liquidity."}
