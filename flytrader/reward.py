"""Experimental, dopamine-inspired paper learning on identified MaleCNS edges.

Market encoding, action values, and broad PAM/PPL1 teaching roles are engineering
choices, not a reconstruction of fly behaviour. Unknown transmitter signs remain
unsigned. Only completed, fee-adjusted paper trades provide teaching signals.
"""
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .context import ChartContext, RiskGuard
from .model import Reservoir
from .provenance import fingerprint

PROFILE = "mushroom-body-reward-v1"
HORIZONS = (30, 60, 180)
WARMUP = 6


class MushroomBody(Reservoir):
    """Full recurrent graph, KC market inputs, MBON outputs, plastic KC→MBON edges."""
    def __init__(self, connectivity, annotations="data/malecns-annotations.feather", *, types=None):
        super().__init__(connectivity)
        if types is None:
            import pyarrow.feather as feather
            rows = feather.read_table(annotations, columns=["bodyId", "type"]).to_pylist()
            types = {str(row["bodyId"]): row["type"] or "" for row in rows}
        self.groups = {name: np.asarray([i for i, n in enumerate(self.node_ids)
                                         if types.get(n, "").startswith(name)], dtype=int)
                       for name in ("KC", "PAM", "PPL1", "MBON")}
        if any(not len(indices) for indices in self.groups.values()):
            raise ValueError("Reward training needs annotated KC, PAM, PPL1 and MBON cells. Select the complete MaleCNS brain.")
        kc, mbon = self.groups["KC"], self.groups["MBON"]
        kc_lookup = np.full(len(self.node_ids), -1, dtype=int)
        kc_lookup[kc] = np.arange(len(kc))
        slots, pres, posts = [], [], []
        for j, row in enumerate(mbon):
            indices = np.arange(self.graph.indptr[row], self.graph.indptr[row + 1])
            indices = indices[kc_lookup[self.graph.indices[indices]] >= 0]
            slots.extend(indices)
            pres.extend(kc_lookup[self.graph.indices[indices]])
            posts.extend([j] * len(indices))
        self.slots, self.pre, self.post = (np.asarray(v, dtype=int) for v in (slots, pres, posts))
        if not len(self.slots):
            raise ValueError("This graph has no KC to MBON connections to train")
        self.base = self.graph.data[self.slots].copy()
        self.scales = np.ones(len(self.slots))
        self.gates = {}
        for group in ("PAM", "PPL1"):
            gate = np.asarray(connectivity.weights[mbon][:, self.groups[group]].sum(axis=1)).ravel()
            self.gates[group] = gate / max(float(gate.max()), 1)
        if not all(np.any(gate[self.post] > 0) for gate in self.gates.values()):
            raise ValueError("Reward training needs recorded dopamine to MBON connections")
        # All chart features enter through real Kenyon cells. The action head sees
        # MBON state only, so it cannot bypass the graph with raw price features.
        rng = np.random.default_rng(7)
        self.kc_inputs = np.zeros((len(kc), 21), dtype=self.graph.dtype)
        for row in self.kc_inputs:
            row[rng.choice(21, 6, replace=False)] = rng.normal(0, .5, 6)
        # Role assignments are an explicit abstraction using recorded DAN input
        # balance, not claims about the biological valence of an MBON type.
        balance = self.gates["PAM"] - self.gates["PPL1"]
        self.avoid = balance >= 0
        if self.avoid.all() or not self.avoid.any():
            self.avoid[np.argsort(balance)[:max(1, len(balance) // 2)]] = False
            self.avoid[np.argmax(balance)] = True
        self.approach = ~self.avoid
        if not self.approach.any() or not self.avoid.any():
            raise ValueError("Opponent training requires at least two MBON pathways")
        self.familiarity = np.zeros(len(kc))
        self.active_kc = np.zeros(len(kc))
        self.reference = np.zeros(len(mbon))
        self.bias = self.approach_activity = self.avoid_activity = 0.
        self.novelty = 1.
        self.regime = "unknown"
        self.pulse = {"PAM": 0.0, "PPL1": 0.0}

    @property
    def display_ids(self):
        return [self.node_ids[i] for group in ("PAM", "PPL1", "MBON") for i in self.groups[group]]

    def encode(self, drive, extra, market=None, *, remember=True):
        market = market or {}
        self.regime = market.get("regime", "unknown")
        regime_index = {"bonding_curve": 0, "pumpswap": 1}.get(self.regime, 2)
        current = self.graph @ self.state
        kc = self.groups["KC"]
        sensory = self.kc_inputs @ np.r_[drive, extra, np.eye(3)[regime_index]]
        permitted = np.arange(len(kc)) % 3 == regime_index if len(kc) >= 30 else np.ones(len(kc), dtype=bool)
        scores = sensory + .1 * current[kc]
        scores[~permitted] = -np.inf
        count = max(1, math.ceil(len(kc) * .05))
        selected = np.argpartition(scores, -count)[-count:]
        sparse = np.zeros(len(kc))
        sparse[selected] = .5 + .5 * np.tanh(np.maximum(0, scores[selected]))
        self.active_kc = sparse
        self.novelty = float(np.mean(1 / np.sqrt(1 + self.familiarity[selected])))
        if remember:
            self.familiarity[selected] += 1
        # A paired baseline drive removes pre-existing output imbalance; learning
        # changes the relative strength of approach and avoidance, initially neutral.
        reference = current[self.groups["MBON"]] + np.bincount(
            self.post, weights=(self.base - self.graph.data[self.slots]) * self.state[kc][self.pre],
            minlength=len(self.groups["MBON"]))
        for group, pulse in self.pulse.items():
            current[self.groups[group]] += pulse
            self.pulse[group] *= .6
        self.state = .5 * self.state + .5 * np.tanh(current)
        self.state[kc] = sparse
        self.reference = .5 * self.reference + .5 * np.tanh(reference)
        output = self.state[self.groups["MBON"]]
        relative = output / np.maximum(self.reference, 1e-9)
        self.approach_activity = float(np.mean(relative[self.approach]))
        self.avoid_activity = float(np.mean(relative[self.avoid]))
        self.bias = self.approach_activity - self.avoid_activity
        return np.r_[1.0, np.tanh(8 * self.state[self.groups["MBON"]])]

    def eligibility(self, head=None):
        return self.active_kc[self.pre].copy()

    def teach(self, eligibility, error, gain):
        group = "PAM" if error >= 0 else "PPL1"
        signal = float(np.tanh(abs(error) / .1) * (gain if error >= 0 else 1))
        pathway = self.avoid if group == "PAM" else self.approach
        delta = .08 * min(signal, 1) * eligibility * self.gates[group][self.post] * pathway[self.post]
        # Opponent depression: reward weakens avoidance; punishment weakens approach.
        self.scales = np.clip(self.scales - delta, .25, 1)
        self.graph.data[self.slots] = self.base * self.scales
        self.pulse[group] = min(1, signal)
        # Record the actual teaching pulse in the outcome frame immediately.
        cells = self.groups[group]
        self.state[cells] = np.clip(self.state[cells] + self.pulse[group], -1, 1)
        return int(np.count_nonzero(delta))

    def recover(self, seconds):
        """Slow recovery per observed elapsed time, not per rendered animation frame."""
        self.scales += -math.expm1(-max(0, min(seconds, 3600)) / 86400) * (1 - self.scales)
        self.graph.data[self.slots] = self.base * self.scales

    def snapshot(self):
        return {"neurons": {k: len(v) for k, v in self.groups.items()},
                "plastic_connections": len(self.slots),
                "changed_connections": int(np.count_nonzero(np.abs(self.scales - 1) > 1e-9)),
                "mean_weight_change_pct": float(np.mean(np.abs(self.scales - 1)) * 100),
                "active_kc_ids": [self.node_ids[self.groups["KC"][i]] for i in np.flatnonzero(self.active_kc)],
                "kc_active_fraction": float(np.count_nonzero(self.active_kc) / len(self.active_kc)),
                "kc_used_fraction": float(np.count_nonzero(self.familiarity) / len(self.familiarity)),
                "approach": self.approach_activity, "avoid": self.avoid_activity, "bias": self.bias,
                "novelty": self.novelty, "regime": self.regime,
                "approach_mbons": int(self.approach.sum()), "avoid_mbons": int(self.avoid.sum()),
                "saturated_fraction": float(np.mean(self.scales <= .250001)),
                "dopamine_activity": {k: float(np.mean(np.abs(self.state[self.groups[k]]))) for k in ("PAM", "PPL1")},
                "teaching_pulse": dict(self.pulse)}


class RewardMemory:
    """One atomic checkpoint holds both action values and real synapse multipliers."""
    def __init__(self, brain, path, identity, *, gain=1.5, exploration=.35, plasticity=True):
        if not math.isfinite(gain) or not 0 <= gain <= 3 or not math.isfinite(exploration) or not 0 <= exploration <= 1:
            raise ValueError("Reward gain must be 0–3 and exploration 0–1")
        self.brain, self.path, self.identity = brain, Path(path), identity
        self.gain, self.exploration, self.plasticity = gain, exploration, plasticity
        self.frozen = False
        self.rng = np.random.default_rng(19)
        # An unseen action has neutral expected net return. A random initial value
        # could incorrectly make the first losing trade look better than expected.
        self.weights = np.zeros((len(HORIZONS), 1 + len(brain.groups["MBON"])))
        self.meta = {"version": 2, "profile": PROFILE, "identity": identity, "examples": 0, "evaluated": 0,
                     "head_counts": [0] * len(HORIZONS), "absolute_error": 0., "baseline_error": 0.,
                     "last_timestamp": 0, "last_update": None, "last_outcome": None}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                with np.load(self.path, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata"]))
                    weights, scales, familiarity = saved["weights"], saved["scales"], saved["familiarity"]
                if (meta.get("version") != 2 or meta.get("identity") != identity or
                        weights.shape != self.weights.shape or scales.shape != brain.scales.shape or
                        not np.isfinite(weights).all() or not np.isfinite(scales).all() or
                        np.any(scales < .25) or np.any(scales > 1) or familiarity.shape != brain.familiarity.shape or
                        not np.isfinite(familiarity).all() or np.any(familiarity < 0)):
                    raise ValueError("Reward checkpoint does not match this circuit or policy")
                self.meta, self.weights, brain.scales = meta, weights, scales
                brain.familiarity = familiarity
                brain.graph.data[brain.slots] = brain.base * scales
                self.rng.bit_generator.state = meta["rng"]
        except BaseException:
            self.lock.close()
            raise
        self.resumed_examples = self.meta["examples"]

    @classmethod
    def for_brain(cls, brain, graph, interval, cost_bps, directory="learning", *, gain=1.5, exploration=.35):
        identity = {"profile": PROFILE, "graph": graph["sha256"],
                    "annotations": fingerprint("data/malecns-annotations.feather")["sha256"],
                    "code": {name: fingerprint(Path(__file__).with_name(name))["sha256"]
                             for name in ("reward.py", "model.py", "context.py", "trial.py", "execution.py", "market.py")},
                    "interval": float(interval), "cost_bps": float(cost_bps), "gain": gain,
                    "exploration": exploration, "horizons": list(HORIZONS), "seed": 19}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        return cls(brain, Path(directory) / f"pump-reward-{key}.npz", identity, gain=gain, exploration=exploration)

    def update(self, credit, event):
        if self.frozen or event["side"] != "SELL" or event["timestamp"] <= self.meta["last_timestamp"]:
            return False
        outcome = math.log(event["amount_sol"] / credit["allocation_sol"])
        action, features = credit["action"], credit["features"]
        prediction = credit["prediction"]
        error = outcome - prediction
        # Bounded value learning; losses remain unamplified, never ignored.
        self.weights[action] += .3 * np.clip(error, -.5, .5) * features / (1 + features @ features)
        eligibility = np.maximum(credit["eligibility"], .25 * credit.get("trace", credit["eligibility"]))
        changed = self.brain.teach(eligibility, error, self.gain) if self.plasticity else 0
        self.meta["examples"] += 1
        self.meta["evaluated"] += 1
        self.meta["head_counts"][action] += 1
        self.meta["absolute_error"] += abs(error)
        self.meta["baseline_error"] += abs(outcome)
        self.meta.update(last_timestamp=float(event["timestamp"]), last_update=datetime.now(timezone.utc).isoformat(),
                         last_outcome={"mint": event["mint"], "predicted": prediction, "observed": outcome,
                                       "pnl_sol": event["pnl_sol"], "reward_error": error,
                                       "entry_kind": credit["entry_kind"], "hold_seconds": HORIZONS[action],
                                       "entry_regime": credit.get("regime", "unknown"),
                                       "connections_updated": changed})
        self.save()
        return True

    def save(self):
        self.meta["rng"] = self.rng.bit_generator.state
        temporary = self.path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, weights=self.weights, scales=self.brain.scales,
                                familiarity=self.brain.familiarity,
                                metadata=json.dumps(self.meta, allow_nan=False))
        temporary.replace(self.path)

    def snapshot(self):
        n = self.meta["examples"]
        return {"enabled": True, "method": "Dopamine-inspired circuit + action-value learning", "profile": PROFILE,
                "target": "Closed paper trade net log return after both fees", "examples": n,
                "session_updates": n - self.resumed_examples, "resumed_examples": self.resumed_examples,
                "evaluated": n, "prediction_mae": self.meta["absolute_error"] / n if n else None,
                "baseline_mae": self.meta["baseline_error"] / n if n else None,
                "last_update": self.meta["last_update"], "last_outcome": self.meta["last_outcome"],
                "checkpoint": str(self.path), "reward_gain": self.gain, "exploration": self.exploration,
                "scope": "Opponent KC→MBON entry memories and MBON outcome estimates learn from closed paper trades; timed/risk exits are separate",
                "circuit": self.brain.snapshot(), "head_counts": list(self.meta["head_counts"])}

    def close(self):
        if not self.lock.closed:
            try:
                self.save()
            finally:
                self.lock.close()


class TrainingGuard(RiskGuard):
    """Ordinary volatility reduces size; verified hazards still block paper entries."""
    def __init__(self, min_cap=100, min_reserves=10):
        super().__init__(min_cap, min_reserves)
        self.previous_reserves = None

    def assess(self, context, market, timestamp, *, live=True):
        neutral = {**context, "last_change_pct": 0, "returns_pct": {"1m": 0, "5m": 0, "15m": 0},
                   "recent_runup_pct": 0, "recent_drawdown_pct": 0, "drawdown_pct": 0, "liquidity_drop_pct": 0}
        cooldown = self.cooldown_until
        self.cooldown_until = 0
        result = super().assess(neutral, market, timestamp, live=live)
        reserves = market.get("real_sol")
        withdrawal = reserves is not None and self.previous_reserves and reserves < self.previous_reserves * .65
        severe = context["last_change_pct"] <= -20 or bool(withdrawal)
        self.previous_reserves = reserves
        self.cooldown_until = max(cooldown, timestamp + 60 if severe else 0)
        reasons = result["reasons"]
        if severe:
            reasons.append("Abrupt price collapse or reserve withdrawal")
        if timestamp < self.cooldown_until:
            reasons.append("60-second pause after a new collapse or withdrawal")
        warnings = []
        returns = context["returns_pct"]
        pumping = (returns["1m"] or 0) >= 15 or (returns["5m"] or 0) >= 35
        dumping = context["last_change_pct"] <= -8 or (returns["1m"] or 0) <= -12
        if pumping: warnings.append("Rapid pump: smaller training position")
        if dumping: warnings.append("Price falling: smaller training position")
        if context["recent_drawdown_pct"] <= -8: warnings.append("Below the recent high: smaller training position")
        result.update(entry_allowed=not reasons, reasons=reasons, warnings=warnings,
                      trend="Dumping" if dumping else "Pumping" if pumping else "No rapid move detected",
                      label="Entries paused" if reasons else "Small exploratory trades allowed",
                      cooldown_seconds=max(0, self.cooldown_until - timestamp),
                      allocation_fraction=.01 if warnings or result["unknowns"] else .03,
                      emergency_exit=severe)
        return result


class RewardCoinModel:
    def __init__(self, reservoir, learner, mint, interval=10, history=()):
        self.reservoir, self.learner, self.mint, self.interval = reservoir, learner, mint, interval
        reservoir.state.fill(0)
        reservoir.reference.fill(0)
        reservoir.pulse = {"PAM": 0., "PPL1": 0.}
        self.context, self.pending = ChartContext(interval), {}
        self.step = self.serial = 0
        self.last_entry = self.last_exit = None
        self.eligible_wait = 0
        self.features = None
        self.last_time = None
        self.trace = np.zeros(len(reservoir.slots))
        self.values = np.zeros(len(HORIZONS))
        self.decision = "Collecting six fresh quotes before exploration"
        warmup = deque(maxlen=21)
        for timestamp, price in history:
            warmup.append(self.context.observe(timestamp, price))
        for drive, extra in warmup:
            reservoir.encode(drive, extra, remember=False)

    def observe(self, bar, market=None):
        drive, extra = self.context.observe(float(bar[0]), float(bar[1]), market)
        delta = float(bar[0]) - self.last_time if self.last_time is not None else self.interval
        if not self.learner.frozen:
            self.reservoir.recover(delta)
        self.features = self.reservoir.encode(drive, extra, market, remember=not self.learner.frozen)
        self.trace = np.maximum(self.trace * math.exp(-delta / 60), self.reservoir.eligibility())
        for credit in self.pending.values():
            if "allocation_sol" in credit:
                credit["trace"] = np.maximum(credit["trace"] * math.exp(-delta / 60), self.reservoir.eligibility())
        self.last_time = float(bar[0])
        self.values = self.learner.weights @ self.features
        self.step += 1
        return float(np.max(self.values)) if self.step >= WARMUP else None

    def plan(self, timestamp, risk, broker, remaining_seconds=math.inf):
        # Credits for cancelled orders are discarded, never converted into rewards.
        active = {p.get("signal_id") for p in (broker.pending, broker.position) if p}
        self.pending = {key: value for key, value in self.pending.items() if key in active}
        if self.step < WARMUP:
            self.decision = f"Warming circuit · {self.step}/{WARMUP} fresh quotes"
            return None
        if not risk["entry_allowed"] or broker.halted:
            self.eligible_wait = 0
            self.decision = "Waiting for risk checks to clear"
            return None
        if broker.position or broker.pending:
            self.decision = "Holding the paper position until its timed or risk exit" if broker.position else "Waiting for the queued fill"
            return None
        if self.last_exit is not None and timestamp - self.last_exit < max(20, self.interval):
            self.decision = "Brief pause after the last trade"
            return None
        available = [i for i, seconds in enumerate(HORIZONS) if seconds + self.interval <= remaining_seconds]
        if not available:
            self.decision = "Too little time remains on this coin for another training trade"
            return None
        self.eligible_wait += self.interval
        probability = self.learner.exploration * (.5 + .5 * self.reservoir.novelty)
        explore = self.learner.exploration > 0 and (self.last_entry is None or self.eligible_wait >= 180 or
                                                   self.learner.rng.random() < probability)
        trained = [i for i in available if self.learner.meta["head_counts"][i] >= 3]
        action = max(trained, key=lambda i: self.values[i]) if trained else None
        if explore:
            # Cover the least-tested holding periods before accumulating repeats.
            least = min(self.learner.meta["head_counts"][i] for i in available)
            candidates = [i for i in available if self.learner.meta["head_counts"][i] == least]
            action = candidates[0] if self.last_entry is None else int(self.learner.rng.choice(candidates))
        elif action is None or self.values[action] <= .002 or self.reservoir.bias <= .005:
            self.decision = "Avoiding a learned loss pattern" if self.reservoir.bias < -.005 else "Waiting for exploration or learned approach with positive net return"
            return None
        self.serial += 1
        signal = f"{self.mint}:{timestamp}:{self.serial}"
        kind = "exploration" if explore else "learned"
        self.pending[signal] = {"features": self.features.copy(), "action": action,
                                "prediction": float(self.values[action]), "entry_kind": kind,
                                "regime": self.reservoir.regime,
                                "eligibility": np.maximum(self.reservoir.eligibility(), .25 * self.trace),
                                "trace": self.reservoir.eligibility()}
        self.last_entry, self.eligible_wait = timestamp, 0
        self.decision = f"{kind.capitalize()} paper buy · {HORIZONS[action]}s planned hold"
        return {"signal_id": signal, "entry_kind": kind, "hold_seconds": HORIZONS[action],
                "expected_net_log_return": float(self.values[action]),
                "allocation_fraction": .01 if explore else .03, "reason": self.decision}

    def settle(self, events):
        for event in events:
            credit = self.pending.get(event.get("signal_id"))
            if credit is None:
                continue
            if event["side"] == "BUY":
                credit["allocation_sol"] = event["amount_sol"]
            elif event["side"] == "SELL" and "allocation_sol" in credit:
                self.learner.update(credit, event)
                self.pending.pop(event["signal_id"], None)
                self.last_exit = event["timestamp"]

    def snapshot(self):
        return {"decision": self.decision, "warmup_observations": min(self.step, WARMUP),
                "action_values": [{"hold_seconds": seconds, "expected_net_log_return": float(self.values[i]),
                                   "completed_examples": self.learner.meta["head_counts"][i]}
                                  for i, seconds in enumerate(HORIZONS)], **self.reservoir.snapshot()}
