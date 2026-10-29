"""Isolated learning checks using completed recordings, never live checkpoints."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import tempfile

from .provenance import fingerprint, write_json

SEEDS = (19, 43, 71)
POLICIES = ("opponent", "readout_only", "no_learning", "random", "momentum")


def recording_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) < 12 or len({r["mint"] for r in rows}) != 1:
        raise ValueError("Each benchmark file needs at least 12 observations of one coin")
    for i, row in enumerate(rows):
        bar, market = row["bar"], row["market"]
        reserves = [market.get(k) for k in ("quote_reserves_sol", "token_reserves", "real_tokens", "real_sol")]
        if (len(bar) != 3 or not all(math.isfinite(v) for v in bar) or bar[1] <= 0 or
                (i and bar[0] <= rows[i - 1]["bar"][0]) or
                market.get("regime") not in ("bonding_curve", "pumpswap") or
                any(v is None or not math.isfinite(v) or v < 0 for v in reserves) or min(reserves[:2]) <= 0):
            raise ValueError("Benchmark requires causal prices and their captured execution reserves")
        if not math.isclose(reserves[0] / reserves[1], bar[1], rel_tol=1e-6):
            raise ValueError("Recorded price and its contemporaneous reserves disagree")
    return rows


def plan_evaluation(root):
    """Split by first-seen coin order, without looking at trade returns.

    The first 60% of distinct coins are training candidates. Their observations
    must precede the first unseen coin. Later repeats of training coins and any
    recording crossing that boundary are excluded, not spliced into the test.
    """
    root = Path(root).resolve()
    paths, records, excluded = set(), [], []
    for report_path in sorted((root / "reports").glob("trial-*.json")):
        if not report_path.resolve().is_relative_to(root / "reports"):
            continue
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            continue
        if (report.get("mode") != "paper-trial" or report.get("status") == "running" or
                report.get("model_profile") != "mushroom-body-reward-v1"):
            continue
        for coin in report.get("coins", []):
            name = coin.get("market_path")
            if not name:
                continue
            path = (root / name).resolve()
            if path in paths or not path.is_relative_to(root / "data"):
                continue
            paths.add(path)
            try:
                rows = recording_rows(path)
                records.append({"path": str(path.relative_to(root)), "mint": rows[0]["mint"],
                                "start": rows[0]["bar"][0], "end": rows[-1]["bar"][0], "observations": len(rows)})
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                excluded.append({"path": str(path.relative_to(root)), "reason": "Incomplete or invalid reserve recording"})
    # One paper account follows a sequential stream. Do not replay a duplicated
    # or concurrent recording as if its earlier quotes arrived after later ones.
    sequential = []
    for record in sorted(records, key=lambda r: (r["start"], r["path"])):
        if sequential and record["start"] <= sequential[-1]["end"]:
            excluded.append({"path": record["path"], "reason": "Overlapping recording"})
        else:
            sequential.append(record)
    coins = list(dict.fromkeys(r["mint"] for r in sequential))
    if len(coins) < 2:
        return {"ready": False, "reason": "Record at least two different coins with execution reserves, then stop the session.",
                "excluded": excluded}
    training_coins = set(coins[:max(1, int(len(coins) * .6))])
    cutoff = next(r["start"] for r in sequential if r["mint"] not in training_coins)
    train, test = [], []
    for record in sequential:
        if record["end"] < cutoff:
            train.append(record)
        elif record["start"] >= cutoff and record["mint"] not in training_coins:
            test.append(record)
        else:
            excluded.append({"path": record["path"], "reason": "Training coin repeated later or recording crosses the split"})
    if not train or not test:
        return {"ready": False, "reason": "Need an earlier training recording and a later, different coin.", "excluded": excluded}
    return {"ready": True, "train": train, "test": test, "excluded": excluded,
            "train_coins": len({r["mint"] for r in train}), "test_coins": len({r["mint"] for r in test}),
            "train_observations": sum(r["observations"] for r in train),
            "test_observations": sum(r["observations"] for r in test), "cutoff": cutoff,
            "split_rule": "First 60% of distinct coins for training; later unseen coins for testing. No selection by profit."}


def summarize(runs, data):
    aggregates = {}
    for policy in (*POLICIES, "cash"):
        scores = [run["policies"][policy] for run in runs]
        aggregates[policy] = {key: statistics.mean(score[key] for score in scores) for key in
                              ("net_pnl_sol", "closed_trades", "costs_sol", "marked_drawdown_pct", "learned_entries", "exploration_entries")}
        aggregates[policy].update(
            min_pnl_sol=min(s["net_pnl_sol"] for s in scores), max_pnl_sol=max(s["net_pnl_sol"] for s in scores),
            unresolved=max(s["unresolved"] for s in scores))
    trained = aggregates["opponent"]
    delta = trained["net_pnl_sol"] - aggregates["no_learning"]["net_pnl_sol"]
    circuit_delta = trained["net_pnl_sol"] - aggregates["readout_only"]["net_pnl_sol"]
    if trained["unresolved"]:
        verdict = "Incomplete exits: this check cannot establish a profit result"
    elif delta <= 0:
        verdict = "Training did not beat the untrained fly in this check"
    elif trained["net_pnl_sol"] <= 0:
        verdict = "Training lost less than the untrained fly; holding cash still won"
    else:
        verdict = "Training finished ahead in this check; more unseen coins are needed"
    summary = {"verdict": verdict, "advantage_vs_untrained_sol": delta,
               "advantage_vs_readout_only_sol": circuit_delta, "evidence": "exploratory",
               "note": f"{data['test_coins']} unseen coin(s), {len(runs)} random seeds. Seeds reuse the same coins; they are not independent market samples. No trading edge established."}
    return aggregates, summary


def run_check(root, edges, output, *, cost=150, interval=10, gain=1.5, exploration=.35,
              min_cap=100, min_reserves=10, seeds=SEEDS):
    # Heavy model imports happen only in the separate evaluation worker.
    import numpy as np
    from .connectome import load_graph
    from .reward import MushroomBody, RewardMemory
    from .reward_eval import load_split, validate_split, parameter_hash, run_split

    if (not all(math.isfinite(v) for v in (cost, interval, gain, exploration, min_cap, min_reserves)) or
            not 0 <= cost < 10000 or not 10 <= interval <= 3600 or not 0 <= gain <= 3 or not 0 <= exploration <= 1 or
            not 0 <= min_cap <= 1_000_000 or not 0 <= min_reserves <= 1_000_000):
        raise ValueError("Invalid evaluation costs, interval or learning settings")
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise ValueError("Use a new benchmark output path")
    plan = plan_evaluation(root)
    if not plan["ready"]:
        raise ValueError(plan["reason"])
    train_paths, test_paths = ([root / r["path"] for r in plan[k]] for k in ("train", "test"))
    train, test = load_split(train_paths), load_split(test_paths)
    validate_split(train, test)
    graph = load_graph(edges)
    result = {"version": 2, "mode": "reward-benchmark", "created": datetime.now(timezone.utc).isoformat(),
              "data": plan, "seeds": list(seeds), "graph": fingerprint(edges),
              "train": [fingerprint(p) for p in train_paths], "test": [fingerprint(p) for p in test_paths],
              "settings": {"cost_bps": cost, "interval_seconds": interval, "reward_gain": gain, "exploration": exploration,
                           "initial_sol": 10, "min_cap_sol": min_cap, "min_reserves_sol": min_reserves}, "runs": [],
              "source": {name: fingerprint(Path(__file__).with_name(name)) for name in
                         ("evaluation.py", "reward_eval.py", "reward.py", "model.py", "context.py", "execution.py", "trial.py", "selection.py")},
              "limitations": ["Fresh training copies; the saved live brain is neither evaluated nor changed.",
                              "Same frozen test data across seeds, not independent market samples.",
                              "Hypothetical swaps assume external traders do not react. Fees and impact are estimates.",
                              "Entry policy is learned; timed and risk exits are not learned."]}
    print(f"Training: {plan['train_coins']} coins / {plan['train_observations']} quotes. Test: {plan['test_coins']} unseen coins / {plan['test_observations']} quotes.", flush=True)
    with tempfile.TemporaryDirectory(prefix="flytrader-check-") as temporary:
        for seed in seeds:
            run = {"seed": seed, "policies": {}}
            for policy in POLICIES:
                print(f"Seed {seed} · {policy}: {'training then testing' if policy in ('opponent', 'readout_only') else 'testing'}…", flush=True)
                brain = MushroomBody(graph)
                memory = RewardMemory(brain, Path(temporary) / f"{seed}-{policy}.npz", {"evaluation": policy},
                                      gain=gain, exploration=exploration, plasticity=policy != "readout_only")
                try:
                    memory.rng = np.random.default_rng(seed)
                    if policy in ("opponent", "readout_only"):
                        run_split(train, memory, policy, cost=cost, interval=interval, seed=seed,
                                  min_cap=min_cap, min_reserves=min_reserves)
                    memory.frozen = True
                    # Training consumes random numbers. Restart the test stream
                    # equally for every fly policy to avoid that accidental bias.
                    memory.rng = np.random.default_rng(seed)
                    before, examples = parameter_hash(memory), memory.meta["examples"]
                    score = run_split(test, memory, policy, cost=cost, interval=interval, seed=seed,
                                      min_cap=min_cap, min_reserves=min_reserves)
                    if before != parameter_hash(memory) or examples != memory.meta["examples"]:
                        raise RuntimeError("Held-out evaluation changed learned parameters")
                    run["policies"][policy] = {"training_trades": examples, "parameters_frozen": True, **score}
                    print(f"Seed {seed} · {policy}: {score['closed_trades']} trades, {score['net_pnl_sol']:+.6f} SOL after costs", flush=True)
                finally:
                    memory.close()
                    del memory, brain
            run["policies"]["cash"] = {"net_pnl_sol": 0., "closed_trades": 0, "costs_sol": 0., "marked_drawdown_pct": 0.,
                                        "learned_entries": 0, "exploration_entries": 0, "unresolved": 0, "parameters_frozen": True}
            result["runs"].append(run)
    result["aggregate"], result["summary"] = summarize(result["runs"], plan)
    write_json(output, result)
    print(result["summary"]["verdict"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", default="data/malecns-full.npz")
    parser.add_argument("--cost-bps", type=float, default=150)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--reward-gain", type=float, default=1.5)
    parser.add_argument("--exploration", type=float, default=.35)
    parser.add_argument("--min-cap-sol", type=float, default=100)
    parser.add_argument("--min-reserves-sol", type=float, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_check(Path.cwd(), args.edges, args.out, cost=args.cost_bps, interval=args.interval,
              gain=args.reward_gain, exploration=args.exploration, min_cap=args.min_cap_sol, min_reserves=args.min_reserves_sol)


if __name__ == "__main__":
    main()
