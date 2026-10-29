"""Observable, causal paper trials. This module cannot sign or submit orders."""
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .activity import ActivityRecorder
from .connectome import load_graph
from .market import CurveFeed, UnsupportedCurve, valid_mint, parse_coin
from .learning import AdaptiveCoinModel, OnlineReadout
from .model import Reservoir, read_bars
from .provenance import fingerprint, write_json
from .results import new_diagnostics, record_diagnostics
from .selection import (ExploreDiscovery, QuoteActivity, price_changed, activity_limits,
                        MIN_WINDOW_CHANGES, EXPLORE_URL, MIN_AGE_SECONDS, MAX_TRADE_AGE_SECONDS, MIN_REAL_SOL)
from .context import PROFILE, ContextCoinModel, RiskGuard, context_learner, prior_history
from .reward import (PROFILE as REWARD_PROFILE, MushroomBody, RewardMemory,
                     RewardCoinModel, TrainingGuard, WARMUP, HORIZONS)
from .execution import ReserveExecution, FillUnavailable

ENTRY_THRESHOLD = .04


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class FeedUnavailable(RuntimeError):
    """A read-only data operation failed; local model/storage errors are separate."""


class QuietCandidate(UnsupportedCurve):
    """Fresh quotes arrived, but the candidate did not show sustained movement."""


class PaperBroker:
    """Signal at t, buy at t+1, sell at t+2. Cash is shared across trial coins."""

    def __init__(self, capital=10.0, cost_bps=150.0, *, impact_fills=False):
        if not math.isfinite(capital) or capital <= 0 or not math.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
            raise ValueError("Use positive paper capital and finite costs below 10,000 bps")
        self.initial = self.cash = self.peak = float(capital)
        self.cost = cost_bps / 10000
        self.position = self.pending = None
        self.events = []
        self.event_count = 0
        self.unresolved = []
        self.trades = self.wins = 0
        self.realized = self.drawdown = 0.0
        self.halted = False
        self.last_price = None
        self.prediction = None
        self.decision = "Waiting for prices"
        self.valuation_stale = False
        self.execution = ReserveExecution(self.cost) if impact_fills else None
        self.execution_market = {}
        self.position_value = None
        self.costs_paid = 0.

    def pause(self, reason):
        self.pending = None
        self.valuation_stale = self.position is not None
        self.decision = reason

    def record(self, event):
        self.event_count += 1
        self.events.append({**event, "id": self.event_count})

    def park_unresolved(self, reason):
        """Keep unavailable tokens and their cost locked; never manufacture a sale."""
        self.pause(reason)
        if self.position:
            self.unresolved.append({**self.position, "last_price": self.last_price,
                                    "last_value_sol": self.position_value if self.execution and self.position_value is not None else self.position["units"] * self.last_price * (1 - self.cost),
                                    "reason": reason})
            self.position = None

    def observe(self, step, timestamp, mint, price, prediction=None, allow_entry=True, *,
                entry_block_reason=None, allocation_fraction=.05, max_order_sol=None,
                policy_controlled=False, entry_plan=None, exit_reason=None, market=None):
        if not math.isfinite(price) or price <= 0 or (prediction is not None and not math.isfinite(prediction)):
            raise ValueError("Paper orders require finite signals and positive fresh prices")
        self.last_price, self.prediction, self.valuation_stale = float(price), prediction, False
        self.execution_market = market or {}
        # Update excursions/valuation on this fresh quote before any exit event.
        if self.position:
            self.snapshot()
        self.decision = "Training — no orders" if prediction is None else "Hold cash — signal below entry threshold"
        due = self.position and (timestamp >= self.position["exit_timestamp"] if "exit_timestamp" in self.position
                                 else step >= self.position["exit_step"])
        if self.position and (due or exit_reason):
            position = self.position
            if position["mint"] != mint:
                raise ValueError("Cannot fill an open position using another coin's price")
            proceeds = position["units"] * price * (1 - self.cost)
            fill = {}
            if self.execution:
                try:
                    fill = self.execution.sell(mint, self.execution_market, position["units"])
                    proceeds = fill["amount_sol"]
                except FillUnavailable as exc:
                    self.pause(f"Paper exit unavailable — {exc}")
                    return self.snapshot()
                self.costs_paid += fill["cost_sol"]
            pnl = proceeds - position["allocation_sol"]
            self.cash += proceeds
            self.realized += pnl
            self.trades += 1
            self.wins += int(pnl > 0)
            closed_equity = self.initial + self.realized
            self.peak = max(self.peak, closed_equity)
            self.drawdown = max(self.drawdown, 1 - closed_equity / self.peak)
            self.halted = self.drawdown >= .1
            self.record({"side": "SELL", "mint": mint, "step": step + 1, "timestamp": timestamp,
                                "quote_price": price, "fill_price": price * (1 - self.cost),
                                "units": position["units"], "amount_sol": proceeds, "pnl_sol": pnl,
                                **fill,
                                **{k: position[k] for k in ("max_favorable_pct", "max_adverse_pct") if k in position},
                                "cash_sol": self.cash,
                                **{k: position[k] for k in ("signal_id", "entry_kind", "hold_seconds") if k in position},
                                "reason": exit_reason or ("Planned holding time ended" if "exit_timestamp" in position else
                                                           "One-observation holding period ended")})
            self.position = None
            self.position_value = None
            self.decision = "Sold — paper position closed"
        if entry_block_reason:
            self.pending = None
        ready = self.pending and (timestamp > self.pending["signal_timestamp"] if "signal_timestamp" in self.pending
                                  else step >= self.pending["fill_step"])
        if ready and not self.halted:
            if self.pending["mint"] != mint:
                raise ValueError("Cannot fill a pending order using another coin's price")
            allocation = self.cash * min(.05, max(0, allocation_fraction))
            if "allocation_fraction" in self.pending:
                allocation = min(allocation, self.cash * self.pending["allocation_fraction"])
            if max_order_sol is not None:
                allocation = min(allocation, max(0, max_order_sol))
            if allocation <= 0:
                self.pending = None
                self.decision = "Paper entry paused — no available allocation"
                return self.snapshot()
            plan = self.pending
            units = allocation / (price * (1 + self.cost))
            fill = {}
            if self.execution:
                try:
                    fill = self.execution.buy(mint, self.execution_market, allocation)
                    units = fill["units"]
                except FillUnavailable as exc:
                    self.pause(f"Paper entry paused — {exc}")
                    return self.snapshot()
                self.costs_paid += fill["cost_sol"]
            self.cash -= allocation
            self.position = {"mint": mint, "units": units, "allocation_sol": allocation,
                             "entry_price": allocation / units, "entry_step": step + 1,
                             "max_favorable_pct": 0., "max_adverse_pct": 0.,
                             "exit_step": step + 1}
            if "hold_seconds" in plan:
                self.position.update({k: plan[k] for k in ("signal_id", "entry_kind", "hold_seconds")})
                self.position.update(entry_timestamp=timestamp, exit_timestamp=timestamp + plan["hold_seconds"])
            self.record({"side": "BUY", "mint": mint, "step": step + 1, "timestamp": timestamp,
                                "quote_price": price, "fill_price": price * (1 + self.cost),
                                "units": units, "amount_sol": allocation, "pnl_sol": None,
                                **fill,
                                "cash_sol": self.cash,
                                **{k: plan[k] for k in ("signal_id", "entry_kind", "hold_seconds") if k in plan},
                                "reason": plan.get("reason", "Previous observation's model signal")})
            self.pending = None
            self.decision = (f"Bought · {plan['entry_kind']} · planned hold {plan['hold_seconds']}s" if "hold_seconds" in plan
                             else "Bought — exit on the next fresh observation")
        if self.halted:
            self.decision = "Risk halt — 10% closed-trade drawdown reached"
        elif sum(p["allocation_sol"] for p in self.unresolved) >= self.initial * .1:
            self.decision = "Entries paused — 10% of starting cash is locked in unresolved positions; learning continues"
        elif entry_block_reason:
            self.decision = "Paper entry paused — " + entry_block_reason
        elif (entry_plan is not None if policy_controlled else prediction is not None and prediction > ENTRY_THRESHOLD) and allow_entry and self.position is None:
            self.pending = {"mint": mint, "fill_step": step + 1, "signal_step": step + 1,
                            **({**entry_plan, "signal_timestamp": timestamp} if entry_plan is not None else {})}
            self.decision = "Buy queued — waiting for the next observation"
        elif prediction is not None and not allow_entry and self.position is None and self.pending is None:
            self.decision = "Hold cash — too few observations remain to finish a new trade"
        elif policy_controlled and self.position:
            self.decision = f"Holding · {max(0, self.position['exit_timestamp'] - timestamp):.0f}s to planned exit"
        return self.snapshot()

    def snapshot(self):
        value = self.position["units"] * self.last_price * (1 - self.cost) if self.position else 0.0
        if self.position and self.execution:
            if not self.valuation_stale:
                try:
                    self.position_value = self.execution.sell(self.position["mint"], self.execution_market, self.position["units"], commit=False)["amount_sol"]
                except FillUnavailable:
                    self.valuation_stale = True
            value = self.position_value if self.position_value is not None else self.position["allocation_sol"]
        if self.position and not self.valuation_stale:
            change = (value / self.position["allocation_sol"] - 1) * 100
            self.position["max_favorable_pct"] = max(self.position["max_favorable_pct"], change)
            self.position["max_adverse_pct"] = min(self.position["max_adverse_pct"], change)
        frozen = sum(p["last_value_sol"] for p in self.unresolved)
        equity = self.cash + value + frozen
        return {"initial_sol": self.initial, "cash_sol": self.cash, "equity_sol": equity,
                "pnl_sol": equity - self.initial, "return_pct": (equity / self.initial - 1) * 100,
                "realized_sol": self.realized,
                "unrealized_sol": (value - self.position["allocation_sol"] if self.position else 0.0) +
                                  frozen - sum(p["allocation_sol"] for p in self.unresolved),
                "closed_trades": self.trades, "wins": self.wins,
                "execution_model": "constant-product shadow reserves" if self.execution else "reserve-price proxy",
                "costs_paid_sol": self.costs_paid if self.execution else None,
                "drawdown_pct": self.drawdown * 100, "halted": self.halted,
                "position": dict(self.position) if self.position else None,
                "pending": dict(self.pending) if self.pending else None,
                "prediction": self.prediction, "decision": self.decision,
                "position_stale": self.valuation_stale,
                "valuation_stale": self.valuation_stale or bool(self.unresolved),
                "unresolved": [dict(p) for p in self.unresolved], "orders": self.event_count}


class CoinModel:
    """Fit only the training prefix. Held-out observations never update the fit."""

    def __init__(self, reservoir, training_count):
        self.reservoir = reservoir
        self.reservoir.state.fill(0)
        self.training_count = training_count
        self.bars, self.training_states = [], []
        self.beta = None

    def observe(self, bar):
        if not np.isfinite(bar).all() or bar[1] <= 0 or bar[2] < 0:
            raise ValueError("Invalid price observation")
        if self.bars and bar[0] <= self.bars[-1][0]:
            raise ValueError("Observation timestamps must advance")
        t = len(self.bars)
        previous = self.bars[-1][1] if t else bar[1]
        base = self.bars[max(0, t - 5)][1] if t else bar[1]
        prior_volume = np.mean([b[2] for b in self.bars[max(0, t - 20):]]) if t else bar[2]
        features = np.clip([np.log(bar[1] / previous) * 20, np.log(bar[1] / base) * 10,
                            np.log1p(bar[2]) - np.log1p(prior_volume)], -3, 3)
        state = self.reservoir.step(features)
        self.bars.append(tuple(map(float, bar)))
        if t < self.training_count:
            self.training_states.append(state)
            if t + 1 == self.training_count:
                prices = np.asarray(self.bars)[:, 1]
                labels = np.log(prices[2:] / prices[1:-1])
                x = np.asarray(self.training_states)[20:-2]
                if not len(x):
                    raise ValueError("Not enough training observations after warmup and label purge")
                self.beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ labels[20:])
                self.training_states.clear()
            return None
        return float(state @ self.beta)


def pump_recording(path):
    """Require recorded Pump curve provenance, not an arbitrary Solana token CSV."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    with path.open() as stream:
        first = next(csv.DictReader(stream), {})
    mint = first.get("mint", "")
    if metadata.get("kind") != "reserve-price-snapshots" or not valid_mint(mint) or "volume_sol" not in first:
        raise ValueError("Quick trials require recorded native-SOL Pump.fun curve prices")
    if metadata.get("mint") != mint:
        raise ValueError("Recording mint and metadata do not match")
    return mint, read_bars(path)


class TrialRunner:
    def __init__(self, args, *, feed_factory=CurveFeed, discover_mint=None, sleeper=time.sleep):
        self.args = args
        args.mint = parse_coin(args.mint)
        self.continuous = getattr(args, "continuous", False)
        self.pin = getattr(args, "pin", False)
        self.profile = getattr(args, "profile", PROFILE)
        self.reward_mode = self.profile == REWARD_PROFILE
        self.enhanced = self.profile in (PROFILE, REWARD_PROFILE)
        self.reward_gain = getattr(args, "reward_gain", 1.5)
        self.exploration = getattr(args, "exploration", .35)
        self.min_cap, self.min_reserves = getattr(args, "min_cap_sol", 100), getattr(args, "min_reserves_sol", 10)
        if self.profile not in (PROFILE, REWARD_PROFILE, "legacy") or not all(math.isfinite(v) and 0 <= v <= 1_000_000 for v in (self.min_cap, self.min_reserves)):
            raise ValueError("Invalid model profile or market-size limits")
        if self.reward_mode and (not self.continuous or not math.isfinite(self.reward_gain) or not 0 <= self.reward_gain <= 3 or
                                 not math.isfinite(self.exploration) or not 0 <= self.exploration <= 1):
            raise ValueError("Reward training requires continuous live mode, reward gain 0–3 and exploration 0–1")
        if self.pin and (not self.continuous or args.source != "live" or not args.mint):
            raise ValueError("Pinning requires continuous live mode and a specific coin")
        if self.continuous and args.source != "live":
            raise ValueError("Continuous learning uses fresh live prices; saved trials remain separate")
        if args.source not in ("recordings", "live") or not 1 <= args.coins <= 5:
            raise ValueError("Choose recorded or live prices and one to five coins")
        if not 65 <= args.count <= 1000 or not math.isfinite(args.interval) or not 10 <= args.interval <= 3600:
            raise ValueError("Use 65–1,000 observations and a 10–3,600 second live interval")
        if not math.isfinite(args.watch_delay) or not 0 <= args.watch_delay <= 1:
            raise ValueError("Use visual pacing between zero and one second")
        if args.mint and (not valid_mint(args.mint) or (args.coins != 1 and not self.continuous)):
            raise ValueError("A specific mint requires a one-coin trial and a valid address")
        if args.source == "recordings" and not args.bars:
            raise ValueError("No ready Pump.fun recordings. Choose Live prices to collect and test a new coin")
        self.broker = PaperBroker(args.capital, args.cost_bps, impact_fills=self.reward_mode)
        self.path = Path(args.out)
        if self.path.exists():
            raise ValueError("A trial must use a new output path")
        self.feed_factory = feed_factory
        self.discover_mint = discover_mint or ExploreDiscovery(min_cap=self.min_cap if self.enhanced else 0,
                                                              min_reserves=self.min_reserves if self.enhanced else MIN_REAL_SOL,
                                                              include_migrated=self.enhanced)
        self.sleep = sleeper
        self.active = self.recorder = None
        self.learner = None
        self.persisted_events = 0
        self.pinned_model = self.pinned_feed = self.pinned_guard = self.pinned_activity = None
        self.current_market = {}
        self.session = {"version": 1, "mode": "paper-trial", "status": "running", "source": args.source, "pid": os.getpid(),
                        "started": utc_now(), "updated": utc_now(), "phase": "Loading brain model",
                        "edges": str(Path(args.edges).resolve()), "coins": [], "requested_coins": args.coins,
                        "planned_coins": None if self.continuous else args.coins, "continuous": self.continuous,
                        "coin_policy": "pin" if self.pin else "rotate", "model_profile": self.profile,
                        "risk_settings": {"minimum_market_cap_sol": self.min_cap, "minimum_reserves_sol": self.min_reserves,
                                          "max_allocation_fraction": .05, "max_reserve_fraction": .01},
                        "coin_count": 0, "observations": 0, "paper": self.broker.snapshot(),
                        "learning": {"enabled": False},
                        "entry_threshold_log_return": ENTRY_THRESHOLD, "diagnostics": new_diagnostics(),
                        "ledger_path": str(self.path.with_suffix(".orders.jsonl")),
                        "skipped_candidates": [],
                        "coin_selection": {"source": "Saved recordings" if args.source == "recordings" else
                                           "User-selected first coin" if args.mint else "Pump.fun Explore · last trade",
                                           "activity_screening": args.source == "live",
                                           "source_url": EXPLORE_URL, "minimum_age_seconds": MIN_AGE_SECONDS,
                                           "maximum_trade_age_seconds": MAX_TRADE_AGE_SECONDS, "minimum_real_sol": MIN_REAL_SOL,
                                           "probe_prices": 3, **activity_limits(args.interval),
                                           "minimum_changes": MIN_WINDOW_CHANGES},
                        "events": [], "cost_bps_per_side": args.cost_bps,
                        "interval_seconds": args.interval, "live_training_observations": 60,
                        "limitations": ["Paper only; never sends orders or accesses a wallet",
                                        "Native-SOL reserve-price fill proxy; assumed costs, no executable quotes",
                                        "Full unsigned tanh model; display samples cells, not computation",
                                        "Causal online readout in continuous mode; fixed holdout fit in finite trials",
                                        "At most 5% allocation; context profile adds real-reserve and missing-data limits",
                                        "No demonstrated profitability; fresh feeds can fail or migrate"]}
        if self.reward_mode:
            self.session.update(entry_threshold_log_return=.002, live_training_observations=WARMUP,
                                holding_periods_seconds=list(HORIZONS),
                                training_policy={"exploration": self.exploration, "reward_gain": self.reward_gain,
                                                 "exploration_allocation": .01, "learned_allocation": .03})
            self.session["decisions_path"] = str(self.path.with_suffix(".decisions.jsonl"))
            self.session["limitations"][3] = "Experimental PAM reward / PPL1 punishment mapping; artificial market inputs and action meanings"
            self.session["limitations"][1] = "Constant-product paper fills with shadow reserve offsets; assumes external flow unaffected; configured fees plus 0.000005 SOL per order"
            self.session["execution_model"] = "constant-product shadow reserves"
            self.session["limitations"].append("Learning uses completed paper trade net returns; unresolved positions provide no teaching reward")

    def publish(self, phase=None):
        if phase:
            self.session["phase"] = phase
        self.session["paper"] = self.broker.snapshot()
        if self.learner:
            self.session["learning"] = self.learner.snapshot()
        new_events = [e for e in self.broker.events if e["id"] > self.persisted_events]
        if new_events:
            path = Path(self.session["ledger_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                for event in new_events:
                    event["coin_index"] = self.active["index"] if self.active else None
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                    self.persisted_events = event["id"]
        # The complete append-only ledger is on disk; telemetry stays bounded.
        del self.broker.events[:-500]
        self.session["events"] = list(self.broker.events)
        self.session["skipped_candidates"] = self.session["skipped_candidates"][-20:]
        self.session["updated"] = utc_now()
        write_json(self.path, self.session)

    def retry(self, operation):
        # Retry reads only. Never synthesize prices or advance pending fills while disconnected.
        for attempt in range(4):
            try:
                result = operation()
                self.session["feed_status"] = "fresh"
                return result
            except UnsupportedCurve:
                raise
            except Exception as exc:
                self.broker.pause("Waiting for fresh data — pending buys cancelled")
                self.session["feed_status"] = "retrying"
                self.session["feed_message"] = f"{type(exc).__name__}: {exc}"
                self.publish(f"Data unavailable · retry {attempt + 1}/3" if attempt < 3 else "Data feed unavailable")
                if attempt == 3:
                    raise FeedUnavailable(f"Data feed unavailable after three retries: {exc}") from exc
                self.sleep(min(5 * 2 ** attempt, 20))

    def find_live_coin(self, seen):
        if self.pin and self.pinned_feed is not None:
            self.sleep(self.args.interval)
            bar = self.retry(self.pinned_feed.read)
            return self.args.mint, self.pinned_feed, [(bar, self.pinned_feed.slot, dict(getattr(self.pinned_feed, "snapshot", {})))]
        requested = self.args.mint if self.pin or not self.continuous or not getattr(self, "used_requested_mint", False) else ""
        self.used_requested_mint = True
        for candidate in range(1 if requested else 3):
            mint = requested or self.retry(lambda: self.discover_mint(seen))
            if not valid_mint(mint) or (mint in seen and not self.pin):
                raise ValueError("Discovery must provide a new valid Pump.fun coin")
            seen.add(mint)
            self.publish(f"Checking coin activity · {mint[:6]}…{mint[-4:]} · price 1/3")
            try:
                feed = self.retry(lambda: self.feed_factory(mint, self.args.rpc))
                if self.enhanced and hasattr(feed, "check_holders"):
                    self.publish("Checking token controls and large token accounts")
                    feed.check_holders()
                first_bar = self.retry(feed.read)
                previews = [(first_bar, feed.slot, dict(getattr(feed, "snapshot", {})))]
                for observation in range(2):
                    self.sleep(self.args.interval)
                    self.publish(f"Checking coin activity · {mint[:6]}…{mint[-4:]} · price {observation + 2}/3")
                    bar = self.retry(feed.read)
                    previews.append((bar, feed.slot, dict(getattr(feed, "snapshot", {}))))
                changes = sum(price_changed(a[0][1], b[0][1]) for a, b in zip(previews, previews[1:]))
                if changes < 2 and not self.pin:
                    raise QuietCandidate("Skipped quiet candidate: fewer than two price changes across three fresh quotes")
                if self.enhanced and not requested:
                    market = previews[-1][2]
                    if market.get("market_cap_sol", 0) < self.min_cap or market.get("real_sol", 0) < self.min_reserves:
                        raise QuietCandidate("Confirmed market cap or real reserves fell below the configured minimum")
                self.selected_candidate = ({"source": "User-selected coin"} if requested else
                                           dict(getattr(self.discover_mint, "last_candidate", None) or {"source": "Injected coin discovery"}))
                self.selected_candidate.update(probe_prices=3, probe_changes=changes)
                if self.pin:
                    self.pinned_feed = feed
                return mint, feed, previews
            except (UnsupportedCurve, RuntimeError, OSError, TimeoutError) as exc:
                self.session["skipped_candidates"].append({"mint": mint, "error": str(exc), "checked_at": utc_now()})
                self.publish("Skipped a quiet or unavailable coin; looking for an active Pump.fun curve")
                if requested or candidate == 2:
                    raise

    def live_bars(self, coin, feed, previews):
        path = Path(coin["bars"])
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"mint": coin["mint"], "kind": "reserve-price-snapshots", "quote_currency": "SOL per token",
                    "bonding_curve": feed.curve, "interval_seconds": self.args.interval,
                    "volume": "Unavailable; zero disables the volume input", "observations": 0,
                    "selection": coin.get("selection", {}), "activity_screening": True}
        market_path = path.with_suffix(".market.jsonl")
        coin["market_path"] = metadata["market_path"] = str(market_path)
        with path.open("x", newline="") as stream, market_path.open("x") as market_stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "mint", "close", "volume_sol"])
            stream.flush()
            try:
                for i in range(self.args.count):
                    started = time.monotonic()
                    bar, slot, market = previews[i] if i < len(previews) else (self.retry(feed.read), feed.slot, dict(getattr(feed, "snapshot", {})))
                    self.current_market = market
                    coin["market"] = market
                    writer.writerow([bar[0], coin["mint"], bar[1], bar[2]])
                    stream.flush()
                    market_stream.write(json.dumps({"bar": list(map(float, bar)), "mint": coin["mint"], "market": market}, allow_nan=False) + "\n")
                    market_stream.flush()
                    metadata["observations"] += 1
                    write_json(path.with_suffix(".metadata.json"), metadata)
                    coin["last_slot"] = slot
                    yield bar
                    if i + 1 < self.args.count and i + 1 >= len(previews):
                        self.sleep(max(0, self.args.interval - (time.monotonic() - started)))
            finally:
                write_json(path.with_suffix(".metadata.json"), metadata)

    def run_coin(self, coin, reservoir, graph, observations):
        history = coin.pop("_history", [])
        self.active = coin
        self.session["coins"].append(coin)
        self.session["coin_count"] = 1 if self.pin else coin["index"] + 1
        self.session["segment_count"] = coin["index"] + 1
        del self.session["coins"][:-50]
        self.session["active_coin"] = len(self.session["coins"]) - 1
        self.session["activity_path"] = coin["activity_path"]
        self.publish("Preparing neuron activity")
        if self.pin and self.pinned_model is not None:
            model = self.pinned_model
        elif self.reward_mode:
            model = RewardCoinModel(reservoir, self.learner, coin["mint"], self.args.interval, history)
        elif self.enhanced:
            model = ContextCoinModel(reservoir, self.learner, coin["mint"], self.args.interval, coin["training_count"], history)
        else:
            model = AdaptiveCoinModel(reservoir, self.learner, coin["mint"]) if self.learner else CoinModel(reservoir, coin["training_count"])
        guard = self.pinned_guard or (TrainingGuard if self.reward_mode else RiskGuard)(self.min_cap, self.min_reserves)
        if self.pin:
            self.pinned_model, self.pinned_guard = model, guard
        self.recorder = ActivityRecorder(Path(coin["activity_path"]).with_suffix(".activity.json"),
                                         self.args.edges, "data/malecns-annotations.feather", coin["total"],
                                         connectivity=graph, priority_ids=reservoir.display_ids if self.reward_mode else ())
        self.recorder.payload["recording"] = {"path": coin["bars"], "asset": coin["mint"], "price_unit": "SOL per token"}
        self.recorder.payload["trial_source"] = self.args.source
        write_json(self.recorder.path, self.recorder.payload)
        initial_equity, initial_trades = self.broker.snapshot()["equity_sol"], self.broker.trades
        previous_price = None
        activity = (self.pinned_activity or QuoteActivity(self.args.interval)) if self.args.source == "live" else None
        if self.pin:
            self.pinned_activity = activity
        quiet_reason = None
        try:
            for step, bar in enumerate(observations):
                if activity:
                    coin["activity"] = activity.observe(float(bar[0]), float(bar[1]))
                    quiet_reason = coin["activity"]["reason"] if self.pin else quiet_reason or coin["activity"]["reason"]
                    if quiet_reason:
                        self.broker.pause("Quiet coin — queued buys cancelled; preparing to move on")
                prediction = model.observe(bar, self.current_market) if self.enhanced else model.observe(bar)
                risk = None
                if self.enhanced:
                    coin["chart_context"] = dict(model.context.snapshot)
                    risk = guard.assess(coin["chart_context"], self.current_market, float(bar[0]), live=self.args.source == "live")
                    if activity and (quiet_reason or coin["activity"]["changes_in_window"] < 2):
                        risk["entry_allowed"] = False
                        risk["reasons"].append(quiet_reason or "Waiting for at least two meaningful price changes")
                        risk["label"] = "Entries paused"
                    coin["risk"] = risk
                options = {}
                if self.reward_mode:
                    plan = model.plan(float(bar[0]), risk, self.broker,
                                      math.inf if self.pin else (coin["total"] - step - 1) * self.args.interval)
                    exit_reason = None
                    position = self.broker.position
                    if position and float(bar[1]) * (1 - self.broker.cost) / position["entry_price"] <= .90:
                        exit_reason = "10% position loss limit on the fresh quote"
                    if risk.get("emergency_exit"):
                        exit_reason = "Collapse or reserve withdrawal on the fresh quote"
                    if quiet_reason:
                        exit_reason = "Coin became quiet; exit valued at its fresh quote"
                    if not self.pin and step + 1 == coin["total"]:
                        exit_reason = "End of this coin's training segment"
                    options = dict(policy_controlled=True, entry_plan=plan, exit_reason=exit_reason)
                before_events = self.broker.event_count
                paper = self.broker.observe(step, float(bar[0]), coin["mint"], float(bar[1]), prediction,
                                            allow_entry=not quiet_reason and (self.pin or step + 2 < coin["total"]),
                                            entry_block_reason="; ".join(risk["reasons"]) if risk and not risk["entry_allowed"] else None,
                                            allocation_fraction=risk["allocation_fraction"] if risk else .05,
                                            max_order_sol=risk["max_order_sol"] if risk else None,
                                            market=self.current_market, **options)
                if self.reward_mode:
                    model.settle([e for e in self.broker.events if e["id"] > before_events])
                    if not self.broker.pending and not self.broker.position and self.broker.event_count == before_events and risk["entry_allowed"] and not self.broker.halted:
                        self.broker.decision = model.decision
                        paper = self.broker.snapshot()
                    self.session["reward"] = coin["reward"] = {**model.snapshot(), "decision": paper["decision"]}
                    with Path(self.session["decisions_path"]).open("a") as stream:
                        stream.write(json.dumps({"timestamp": float(bar[0]), "mint": coin["mint"], "price": float(bar[1]),
                                                 "decision": paper["decision"], "risk": risk, "reward": coin["reward"],
                                                 "position": paper["position"], "prediction": prediction,
                                                 "new_orders": [e for e in self.broker.events if e["id"] > before_events],
                                                 "learning_updates": self.learner.meta["examples"],
                                                 "last_outcome": self.learner.meta["last_outcome"]}, allow_nan=False) + "\n")
                if quiet_reason:
                    self.broker.decision = "Pinned coin is quiet — observing and learning; paper buys paused" if self.pin else "Quiet coin — moving to another active coin" if self.continuous else "Quiet coin — ending this coin's trial"
                    paper = self.broker.snapshot()
                record_diagnostics(self.session["diagnostics"], float(bar[1]), previous_price, paper,
                                   self.session["entry_threshold_log_return"])
                previous_price = float(bar[1])
                phase = ((f"Warming neuron state {step + 1}/21" if self.learner.meta['examples'] >= 38 else
                          f"Initial training {step + 1}/60 · {self.learner.meta['examples']}/38 learned outcomes") if prediction is None
                         else "Paper trading + learning") if self.learner else (f"Training {step + 1}/{coin['training_count']}" if prediction is None else "Paper trading")
                if self.reward_mode:
                    phase = f"Warming reward circuit · {model.step}/{WARMUP}" if prediction is None else "Paper exploration + reward learning"
                coin.update(status="training" if prediction is None else "trading", observations=step + 1,
                            last_price=float(bar[1]), last_timestamp=float(bar[0]),
                            closed_trades=self.broker.trades - initial_trades,
                            pnl_sol=paper["equity_sol"] - initial_equity)
                self.session["observations"] += 1
                self.session["last_observation_at"] = utc_now()
                self.session["feed_status"] = "recorded" if self.args.source == "recordings" else "fresh"
                self.recorder(step, bar, reservoir.node_ids, reservoir.state, phase=phase, paper=paper, mint=coin["mint"],
                              learning=self.learner.snapshot() if self.learner else None, coin_index=coin["index"],
                              chart_context=coin.get("chart_context"), risk=risk, market=dict(self.current_market),
                              reward=coin.get("reward"))
                self.publish(phase)
                if step % 10 == 0 or len(self.broker.events) != getattr(self, "event_count", 0):
                    print(f"Coin {coin['index'] + 1}/{self.session['planned_coins'] or 'continuous'} · {step + 1}/{coin['total']} · {phase} · {paper['decision']}", flush=True)
                self.event_count = len(self.broker.events)
                if quiet_reason and not self.pin and not self.broker.position:
                    coin.update(status="quiet", rotation_reason=quiet_reason)
                    self.publish(f"Moving on: {quiet_reason}. Learning memory retained.")
                    break
                if self.args.source == "recordings" and self.args.watch_delay:
                    self.sleep(self.args.watch_delay)
            if coin.get("status") != "quiet":
                coin["status"] = "completed"
            self.recorder.finish("completed")
        except BaseException:
            coin["status"] = "interrupted"
            self.recorder.finish("interrupted")
            raise
        finally:
            close = getattr(observations, "close", None)
            if close:
                close()
            path = Path(coin["bars"])
            if path.exists():
                coin["fingerprint"] = fingerprint(path)
            write_json(Path(coin["activity_path"]).with_suffix(".coin.json"), {"mode": "paper-coin", **coin})
            self.recorder = None

    def run(self):
        self.publish()
        try:
            graph = load_graph(self.args.edges)
            reservoir = MushroomBody(graph) if self.reward_mode else Reservoir(graph)
            self.session.update(neurons=len(graph.node_ids), connections=graph.weights.nnz,
                                graph_fingerprint=fingerprint(self.args.edges),
                                model_fingerprints={name: fingerprint(Path(__file__).with_name(name))
                                                    for name in ("trial.py", "model.py", "connectome.py", "market.py", "activity.py", "learning.py", "selection.py", "context.py", "reward.py")})
            if self.continuous:
                if self.reward_mode:
                    self.learner = RewardMemory.for_brain(reservoir, self.session["graph_fingerprint"], self.args.interval,
                                                         self.args.cost_bps, gain=self.reward_gain, exploration=self.exploration)
                else:
                    factory = context_learner if self.enhanced else OnlineReadout.for_reservoir
                    self.learner = factory(reservoir, self.session["graph_fingerprint"], self.args.interval)
                self.session["learning_start"] = self.learner.snapshot()
                self.publish("Restored learning memory; finding fresh prices" if self.learner.resumed_examples else "Starting continuous learning from fresh prices")
            recordings, seen = [], set()
            if self.args.source == "recordings":
                for path in self.args.bars:
                    mint, bars = pump_recording(path)
                    if mint not in seen:
                        recordings.append((path, mint, bars))
                        seen.add(mint)
                recordings = recordings[:self.args.coins]
                self.session["planned_coins"] = len(recordings)
                seen.clear()
            index = 0
            while self.continuous or index < self.session["planned_coins"]:
                self.active = None
                coin = {"index": index, "status": "preparing", "observations": 0, "closed_trades": 0, "pnl_sol": 0.0,
                        "activity_path": str(self.path.with_name(f"{self.path.stem}-coin-{index + 1}.json"))}
                if self.pin:
                    coin["mint"] = self.args.mint
                try:
                    if self.args.source == "recordings":
                        path, mint, bars = recordings[index]
                        coin.update(mint=mint, bars=str(path), total=len(bars), training_count=int(len(bars) * .7))
                        coin["input_fingerprint"] = fingerprint(path)
                        observations = iter(bars)
                    else:
                        self.session["activity_path"] = None
                        self.publish("Following your pinned coin · runs until Stop" if self.pin else
                                     f"Finding active Pump.fun coin {index + 1} · runs until Stop" if self.continuous else f"Finding active Pump.fun coin {index + 1}/{self.session['planned_coins']}")
                        mint, feed, previews = self.find_live_coin(seen)
                        coin.update(mint=mint, bars=str(Path("data") / f"{self.path.stem}-coin-{index + 1}.csv"),
                                    total=self.args.count, training_count=60, selection=self.selected_candidate)
                        if self.enhanced:
                            history, sources = prior_history(mint, float(previews[0][0][0]))
                            context_path = Path(coin["bars"]).with_name(Path(coin["bars"]).stem + "-context.csv")
                            if history:
                                context_path.parent.mkdir(parents=True, exist_ok=True)
                                with context_path.open("w", newline="") as stream:
                                    writer = csv.writer(stream)
                                    writer.writerow(["timestamp", "mint", "close", "volume_sol"])
                                    writer.writerows((t, mint, p, 0) for t, p in history)
                                coin["history_path"] = str(context_path)
                            coin.update(_history=history, history={"points": len(history), "sources": sources,
                                        "scope": "Previously captured prices plus this session; full lifetime chart is not available",
                                        "oldest_timestamp": history[0][0] if history else float(previews[0][0][0])})
                        observations = self.live_bars(coin, feed, previews)
                    seen.add(mint)
                    self.run_coin(coin, reservoir, graph, observations)
                    if self.args.source == "recordings" and coin["fingerprint"] != coin["input_fingerprint"]:
                        raise ValueError("Input recording changed during the trial")
                except (UnsupportedCurve, FeedUnavailable) as exc:
                    coin.update(status="unavailable", error=str(exc))
                    if coin not in self.session["coins"]:
                        self.session["coins"].append(coin)
                    self.session["coin_count"] = 1 if self.pin else index + 1
                    self.session["segment_count"] = index + 1
                    del self.session["coins"][:-50]
                    self.session["active_coin"] = len(self.session["coins"]) - 1
                    write_json(Path(coin["activity_path"]).with_suffix(".coin.json"), {"mode": "paper-coin", **coin})
                    self.broker.pause("Coin unavailable — pending buys cancelled")
                    self.publish(str(exc))
                    if self.continuous:
                        self.broker.park_unresolved(str(exc))
                        if self.pin:
                            self.pinned_feed = None
                            if self.pinned_model is not None:
                                self.pinned_model.pending.clear()
                        if self.pin and isinstance(exc, UnsupportedCurve):
                            self.session["status"] = "failed"
                            self.session["feed_status"] = "unsupported"
                            self.session["feed_message"] = str(exc)
                            self.publish(str(exc))
                            return self.session
                        self.session["feed_status"] = "retrying"
                        self.session["feed_message"] = str(exc)
                        self.publish("Pinned coin unavailable; retrying the same coin in 15 seconds. No coin switching." if self.pin else
                                     "Feed unavailable; will find another coin in 15 seconds. Learning memory retained.")
                        self.sleep(15)
                    elif self.broker.position:
                        self.session["status"] = "needs_attention"
                        self.publish("Open paper position has no fresh quote; its last valuation is unresolved")
                        return self.session
                if self.continuous:
                    # Keep recent rejections out of the next shortlist as well as followed coins.
                    seen = set(c["mint"] for c in self.session["coins"] if c.get("mint"))
                    seen.update(c["mint"] for c in self.session["skipped_candidates"])
                index += 1
            self.broker.pending = None
            self.session["status"] = "completed" if all(c["status"] == "completed" for c in self.session["coins"]) else "partial"
            if not self.session["observations"]:
                self.session["status"] = "failed"
            self.publish("Trial complete" if self.session["status"] == "completed" else
                         "Trial ended early for quiet or unavailable coins — see coin details")
            return self.session
        except KeyboardInterrupt:
            self.broker.pause("Stopped by you; no new paper orders")
            self.session["status"] = "stopped"
            self.publish("Trial stopped; the open position keeps its last valuation and is unresolved" if self.broker.position
                         else "Trial stopped; saved prices, trades and neuron activity are available")
            raise
        except BaseException as exc:
            self.broker.pause("Trial interrupted; pending buys cancelled")
            self.session["status"] = "failed"
            self.session["error"] = str(exc)
            self.publish("Trial could not finish")
            raise
        finally:
            self.session["finished"] = utc_now()
            try:
                self.publish()
            finally:
                if self.learner:
                    self.learner.close()


def run_trial(args):
    session = TrialRunner(args).run()
    print(f"Trial {session['status']} · {session['observations']} observations · {session['paper']['closed_trades']} closed paper trades · {args.out}", flush=True)
    if session["status"] == "failed":
        raise ValueError("No valid Pump.fun data was received; see the saved trial for feed errors")
    return session
