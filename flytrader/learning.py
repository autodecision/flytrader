"""Causal online readout learning; the connectome itself remains fixed."""
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from .provenance import fingerprint


class OnlineReadout:
    """Exponentially weighted ridge regression, updated after labels mature.

    Sufficient statistics stay bounded; a small ridge term is retained forever.
    Prediction errors use the prediction made before its outcome was known.
    """

    def __init__(self, size, path, identity, forgetting=.995):
        self.path, self.identity = Path(path), identity
        self.size, self.forgetting = size, forgetting
        self.gram = np.zeros((size, size))
        self.target = np.zeros(size)
        self.beta = np.zeros(size)
        self.meta = {"version": 1, "identity": identity, "examples": 0, "evaluated": 0,
                     "absolute_error": 0.0, "baseline_error": 0.0, "last_timestamp": 0,
                     "last_update": None, "last_outcome": None, "forgetting": forgetting}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                with np.load(self.path, allow_pickle=False) as checkpoint:
                    meta = json.loads(str(checkpoint["metadata"]))
                    gram, target = checkpoint["gram"], checkpoint["target"]
                if (meta["identity"] != identity or meta["version"] != 1 or
                        meta["forgetting"] != forgetting or gram.shape != (size, size) or
                        target.shape != (size,) or not np.isfinite(gram).all() or not np.isfinite(target).all()):
                    raise ValueError("Learning checkpoint does not match this model")
                self.gram, self.target, self.meta = gram, target, meta
                self.beta = np.linalg.solve(gram + np.eye(size), target)
        except BaseException:
            self.lock.close()
            raise
        self.resumed_examples = self.meta["examples"]

    @classmethod
    def for_reservoir(cls, reservoir, graph_fingerprint, interval, directory="learning"):
        # The input interval changes the prediction horizon, so it has separate memory.
        identity = {"graph": graph_fingerprint["sha256"],
                    "dynamics": fingerprint(Path(__file__).with_name("model.py"))["sha256"],
                    "learner": fingerprint(__file__)["sha256"], "interval": float(interval), "seed": 7}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        size = 4 + len(reservoir.readout_nodes) + (64 if reservoir.pool_groups is not None else 0)
        return cls(size, Path(directory) / f"pump-{key}.npz", identity)

    def update(self, state, outcome, *, prediction, timestamp, mint, signal_step):
        # Live time only moves forward. Repeated/replayed observations cannot train twice.
        if timestamp <= self.meta["last_timestamp"]:
            return False
        state = np.asarray(state, dtype=float)
        if state.shape != (self.size,) or not np.isfinite(state).all() or not np.isfinite(outcome):
            raise ValueError("Learning requires finite states and an observed outcome")
        gram = self.forgetting * self.gram + np.outer(state, state)
        target = self.forgetting * self.target + state * outcome
        beta = np.linalg.solve(gram + np.eye(self.size), target)
        if not np.isfinite(beta).all():
            raise ValueError("Learning produced invalid parameters")
        self.gram, self.target, self.beta = gram, target, beta
        self.meta["examples"] += 1
        if prediction is not None:
            self.meta["evaluated"] += 1
            self.meta["absolute_error"] += abs(prediction - outcome)
            self.meta["baseline_error"] += abs(outcome)
        self.meta.update(last_timestamp=float(timestamp), last_update=datetime.now(timezone.utc).isoformat(),
                         last_outcome={"mint": mint, "signal_step": signal_step,
                                       "predicted": prediction, "observed": float(outcome)})
        self.save()
        return True

    def save(self):
        temporary = self.path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, gram=self.gram, target=self.target,
                                metadata=json.dumps(self.meta, allow_nan=False))
        temporary.replace(self.path)

    def snapshot(self):
        count = self.meta["evaluated"]
        return {"enabled": True, "method": "Online ridge readout", "examples": self.meta["examples"],
                "session_updates": self.meta["examples"] - self.resumed_examples,
                "resumed_examples": self.resumed_examples, "evaluated": count,
                "prediction_mae": self.meta["absolute_error"] / count if count else None,
                "baseline_mae": self.meta["baseline_error"] / count if count else None,
                "last_update": self.meta["last_update"], "last_outcome": self.meta["last_outcome"],
                "checkpoint": str(self.path), "forgetting": self.forgetting,
                "scope": "Trading readout adapts; neuron connections and dynamics stay fixed"}

    def close(self):
        if not self.lock.closed:
            try:
                self.save()
            finally:
                self.lock.close()


class AdaptiveCoinModel:
    """At t, learn the outcome of signal t-2, then predict using state t."""

    def __init__(self, reservoir, learner, mint):
        self.reservoir, self.learner, self.mint = reservoir, learner, mint
        reservoir.state.fill(0)
        self.bars, self.pending = deque(maxlen=21), deque()
        self.step = 0

    def observe(self, bar):
        if not np.isfinite(bar).all() or bar[1] <= 0 or bar[2] < 0:
            raise ValueError("Invalid price observation")
        if self.bars and bar[0] <= self.bars[-1][0]:
            raise ValueError("Observation timestamps must advance")
        t = self.step
        previous = self.bars[-1][1] if t else bar[1]
        base = self.bars[max(0, len(self.bars) - 5)][1] if t else bar[1]
        volume = np.mean([b[2] for b in list(self.bars)[-20:]]) if t else bar[2]
        features = np.clip([np.log(bar[1] / previous) * 20, np.log(bar[1] / base) * 10,
                            np.log1p(bar[2]) - np.log1p(volume)], -3, 3)
        state = self.reservoir.step(features)
        if self.pending and self.pending[0][0] == t - 2:
            signal_step, old_state, prediction = self.pending.popleft()
            self.learner.update(old_state, float(np.log(bar[1] / previous)), prediction=prediction,
                                timestamp=bar[0], mint=self.mint, signal_step=signal_step + 1)
        prediction = float(state @ self.learner.beta) if t >= 20 and self.learner.meta["examples"] >= 38 else None
        if t >= 20:
            self.pending.append((t, state, prediction))
        self.bars.append(tuple(map(float, bar)))
        self.step += 1
        return prediction
