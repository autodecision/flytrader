"""Compare frozen policies on later, unseen tokens with captured execution reserves.

Run with python -m flytrader.reward_eval --train ...market.jsonl --test ...market.jsonl.
Never opens a feed or writes to live learning memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .connectome import load_graph
from .provenance import fingerprint, write_json
from .reward import MushroomBody, RewardMemory, RewardCoinModel, TrainingGuard, HORIZONS
from .selection import QuoteActivity
from .trial import PaperBroker
from .evaluation import recording_rows


def load_split(paths):
    recordings = []
    for path in paths:
        recordings.append(recording_rows(path))
    recordings.sort(key=lambda rows: rows[0]["bar"][0])
    if any(b[0]["bar"][0] <= a[-1]["bar"][0] for a, b in zip(recordings, recordings[1:])):
        raise ValueError("Benchmark recordings must not overlap in time")
    return recordings


def validate_split(train, test):
    if not train or not test:
        raise ValueError("Both training and test recordings are required")
    if {r[0]["mint"] for r in train} & {r[0]["mint"] for r in test}:
        raise ValueError("Test tokens must be absent from the training set")
    if max(rows[-1]["bar"][0] for rows in train) >= min(rows[0]["bar"][0] for rows in test):
        raise ValueError("Every test observation must be later than the training set")


def parameter_hash(memory):
    return hashlib.sha256(memory.weights.tobytes() + memory.brain.scales.tobytes() +
                          memory.brain.familiarity.tobytes()).hexdigest()


def run_split(recordings, memory, policy, *, cost=150, interval=10, seed=19, min_cap=100, min_reserves=10):
    broker, rng = PaperBroker(10, cost, impact_fills=True), np.random.default_rng(seed)
    peak, max_drawdown, turnover = 10., 0., 0.
    rewards, impacts, signals = [], [], {"PAM": 0, "PPL1": 0}
    entries = {"learned": 0, "exploration": 0}
    for rows in recordings:
        mint = rows[0]["mint"]
        model = RewardCoinModel(memory.brain, memory, mint, interval)
        guard, activity = TrainingGuard(min_cap, min_reserves), QuoteActivity(interval)
        last_baseline_exit = None
        for step, row in enumerate(rows):
            bar, market = row["bar"], row["market"]
            timestamp, price, _ = bar
            prediction = model.observe(bar, market)
            risk = guard.assess(model.context.snapshot, market, timestamp)
            motion = activity.observe(timestamp, price)
            if motion["reason"] or motion["changes_in_window"] < 2:
                risk["entry_allowed"] = False
                risk["reasons"].append(motion["reason"] or "Waiting for price changes")
            remaining = rows[-1]["bar"][0] - timestamp
            plan = None
            if policy in ("random", "momentum"):
                plan = None
                choices = [seconds for seconds in HORIZONS if seconds + interval <= remaining]
                momentum = model.context.snapshot["returns_pct"]["1m"]
                trigger = rng.random() < .35 if policy == "random" else momentum is not None and momentum > 3
                if (prediction is not None and risk["entry_allowed"] and not broker.position and not broker.pending and
                        not broker.halted and choices and trigger and
                        (last_baseline_exit is None or timestamp - last_baseline_exit >= max(20, interval))):
                    plan = {"signal_id": f"{mint}:{timestamp}", "entry_kind": policy,
                            "hold_seconds": int(rng.choice(choices)) if policy == "random" else choices[0],
                            "allocation_fraction": .01, "reason": policy + " baseline"}
            else:
                bias = memory.brain.bias
                try:
                    # This ablation learns values with fixed synapses. A neutral
                    # (unlearnable) opponent gate would otherwise reject every
                    # learned entry, making it an exploration-only straw man.
                    if policy == "readout_only":
                        memory.brain.bias = 1.
                    plan = model.plan(timestamp, risk, broker, remaining)
                finally:
                    memory.brain.bias = bias
            exit_reason = "End of test coin" if step + 1 == len(rows) else None
            if risk.get("emergency_exit") or motion["reason"]:
                exit_reason = "Risk or quiet exit"
            if broker.position and price * (1 - broker.cost) / broker.position["entry_price"] <= .9:
                exit_reason = "Position loss limit"
            previous = broker.event_count
            paper = broker.observe(step, timestamp, mint, price, prediction, policy_controlled=True,
                                   entry_plan=plan, exit_reason=exit_reason, market=market,
                                   entry_block_reason="; ".join(risk["reasons"]) if not risk["entry_allowed"] else None,
                                   allocation_fraction=risk["allocation_fraction"], max_order_sol=risk["max_order_sol"])
            events = [e for e in broker.events if e["id"] > previous]
            model.settle(events)
            for event in events:
                turnover += event["amount_sol"]
                impacts.append(event["impact_pct"])
                if event["side"] == "SELL":
                    rewards.append(event["pnl_sol"])
                    last_baseline_exit = timestamp
                elif event.get("entry_kind") in entries:
                    entries[event["entry_kind"]] += 1
            for channel, amplitude in memory.brain.pulse.items():
                signals[channel] += int(amplitude > .05)
            peak = max(peak, paper["equity_sol"])
            max_drawdown = max(max_drawdown, 1 - paper["equity_sol"] / peak)
        if broker.position:
            broker.park_unresolved("No executable quote at end of benchmark coin")
        broker.pending = None
    paper = broker.snapshot()
    return {"net_pnl_sol": paper["pnl_sol"], "closed_trades": paper["closed_trades"],
            "win_fraction": paper["wins"] / paper["closed_trades"] if paper["closed_trades"] else None,
            "marked_drawdown_pct": 100 * max_drawdown, "costs_sol": paper["costs_paid_sol"],
            "turnover_sol": turnover, "mean_impact_pct": float(np.mean(impacts)) if impacts else None,
            "closed_trade_pnl_sol": rewards, "unresolved": len(paper["unresolved"]),
            "learned_entries": entries["learned"], "exploration_entries": entries["exploration"],
            "pulse_observations": signals, "circuit": memory.brain.snapshot()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--edges", default="data/malecns-full.npz")
    parser.add_argument("--cost-bps", type=float, default=150)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out", default="reports/reward-comparison.json")
    args = parser.parse_args()
    train, test = load_split(args.train), load_split(args.test)
    validate_split(train, test)
    if Path(args.out).exists():
        raise ValueError("Use a new benchmark output path")
    graph = load_graph(args.edges)
    result = {"mode": "reward-benchmark", "graph": fingerprint(args.edges),
              "train": [fingerprint(p) for p in args.train], "test": [fingerprint(p) for p in args.test],
              "seed": args.seed, "cost_bps": args.cost_bps, "policies": {},
              "limitations": "Frozen unseen-token evaluation. Single-seed results do not establish profitability. External flow is assumed unchanged by paper swaps. Exit policy is not learned."}
    with tempfile.TemporaryDirectory(prefix="flytrader-eval-") as temporary:
        for policy in ("opponent", "no_learning", "random", "momentum"):
            brain = MushroomBody(graph)
            memory = RewardMemory(brain, Path(temporary) / f"{policy}.npz", {"evaluation": policy})
            try:
                memory.rng = np.random.default_rng(args.seed)
                if policy == "opponent":
                    run_split(train, memory, policy, cost=args.cost_bps, interval=args.interval, seed=args.seed)
                memory.frozen = True
                memory.rng = np.random.default_rng(args.seed)
                before, examples = parameter_hash(memory), memory.meta["examples"]
                score = run_split(test, memory, policy, cost=args.cost_bps, interval=args.interval, seed=args.seed)
                if before != parameter_hash(memory) or examples != memory.meta["examples"]:
                    raise RuntimeError("Held-out evaluation changed learned parameters")
                result["policies"][policy] = {"training_trades": examples, "parameters_frozen": True, **score}
                print(f"{policy}: {score['closed_trades']} trades, {score['net_pnl_sol']:+.6f} SOL after costs", flush=True)
            finally:
                memory.close()
                del memory, brain
    write_json(args.out, result)


if __name__ == "__main__":
    main()
