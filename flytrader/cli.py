import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from .model import evaluate, load_edges, read_bars
from .connectome import compile_full, load_graph
from .provenance import run_record, write_json


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def demo(directory):
    directory = Path(directory)
    rng = np.random.default_rng(7)
    write_csv(directory / "synthetic-edges.csv", ["bodyId_pre", "bodyId_post", "weight"],
              [(i, (i + j) % 64, int(rng.integers(1, 30))) for i in range(64) for j in (1, 7, 13)])
    prices = np.exp(np.cumsum(rng.normal(0, 0.04, 1000))) * 1e-7
    write_csv(directory / "synthetic-bars.csv", ["timestamp", "mint", "close", "volume_sol"],
              [(1700000000 + 60 * i, "SYNTHETIC_NOT_A_TOKEN", p, rng.uniform(0, 10))
               for i, p in enumerate(prices)])
    print(f"Synthetic fixtures written to {directory}; these are NOT fly or market data.")


def fetch_connectome(args):
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise ValueError("Set NEUPRINT_TOKEN locally with your neuPrint API token")
    from neuprint import Client, NeuronCriteria, fetch_adjacencies, fetch_neurons
    client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
    neurons, _ = fetch_neurons(NeuronCriteria(type=args.type_regex, regex=True), client=client)
    ids = sorted(neurons.bodyId.tolist())
    if not ids:
        raise ValueError("No matching neurons; choose a MaleCNS type regex")
    if len(ids) > args.max_neurons:
        raise ValueError(f"Matched {len(ids)} neurons; narrow --type-regex or increase --max-neurons")
    _, connections = fetch_adjacencies(ids, ids, client=client)
    totals = connections.groupby(["bodyId_pre", "bodyId_post"], as_index=False).weight.sum()
    if totals.empty:
        raise ValueError("No internal connections; choose a broader circuit")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    totals.to_csv(destination, index=False)
    provenance = {"dataset": "male-cns:v1.0", "type_regex": args.type_regex,
                  "matched_neurons": len(ids), "edges": len(totals), "license": "CC-BY",
                  "source": "https://male-cns.janelia.org/download/",
                  "attribution": "MaleCNS collaboration: HHMI Janelia, Cambridge, MRC LMB, Google Research",
                  "selection": "Induced subgraph of matching neurons; not the complete brain"}
    destination.with_suffix(".metadata.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


def prepare_connectome(args):
    """Filter the public bulk graph in record batches to bound peak memory."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather
    annotations = feather.read_table(args.annotations, columns=["bodyId", "type"])
    pattern = re.compile(args.type_regex)
    selected = [row["bodyId"] for row in annotations.to_pylist()
                if pattern.search(row["type"] or "")]
    if not selected or len(selected) > args.max_neurons:
        raise ValueError(f"Matched {len(selected)} neurons; need 1 through {args.max_neurons}")
    with pa.memory_map(args.weights, "r") as source:
        reader = pa.ipc.open_file(source)
        columns = reader.schema.names
        pre = "body_pre" if "body_pre" in columns else "bodyId_pre"
        post = "body_post" if "body_post" in columns else "bodyId_post"
        if not {pre, post, "weight"}.issubset(columns):
            raise ValueError(f"Unexpected connectivity schema: {columns}")
        edges = []
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            mask = pc.and_(pc.is_in(batch[pre], value_set=pa.array(selected, type=batch[pre].type)),
                           pc.is_in(batch[post], value_set=pa.array(selected, type=batch[post].type)))
            for row in batch.filter(mask).select([pre, post, "weight"]).to_pylist():
                edges.append((row[pre], row[post], row["weight"]))
    if not edges:
        raise ValueError("The selected neurons have no internal edges")
    write_csv(args.out, ["bodyId_pre", "bodyId_post", "weight"], edges)
    metadata = {"dataset": "male-cns:v1.0", "source": "https://male-cns.janelia.org/download/",
                "license": "CC-BY", "attribution": "MaleCNS collaboration: HHMI Janelia, Cambridge, MRC LMB, Google Research",
                "type_regex": args.type_regex, "matched_neurons": len(selected), "edges": len(edges),
                "selection": "Induced subgraph; experimental selection, not a validated trading circuit",
                "weights_file": str(Path(args.weights).resolve())}
    Path(args.out).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Flytrader research CLI (paper trading only)")
    commands = parser.add_subparsers(dest="command", required=True)
    demo_parser = commands.add_parser("demo", help="Generate explicitly synthetic smoke-test fixtures")
    demo_parser.add_argument("--out", default="data/demo")
    fetch = commands.add_parser("fetch-connectome", help="Import a real MaleCNS induced subgraph")
    fetch.add_argument("--type-regex", required=True)
    fetch.add_argument("--max-neurons", type=int, default=2000)
    fetch.add_argument("--out", default="data/malecns-edges.csv")
    prepare = commands.add_parser("prepare-connectome", help="Extract a circuit from public bulk Feather files")
    prepare.add_argument("--weights", default="data/malecns-weights.feather")
    prepare.add_argument("--annotations", default="data/malecns-annotations.feather")
    prepare.add_argument("--type-regex", default="^(MBON|APL|PAM|PPL)")
    prepare.add_argument("--max-neurons", type=int, default=2000)
    prepare.add_argument("--out", default="data/malecns-edges.csv")
    full = commands.add_parser("prepare-full", help="Compile the full retained MaleCNS neuronal graph into sparse arrays")
    full.add_argument("--weights", default="data/malecns-weights.feather")
    full.add_argument("--annotations", default="data/malecns-annotations.feather")
    full.add_argument("--out", default="data/malecns-full.npz")
    capture_parser = commands.add_parser("capture", help="Collect free read-only SOL curve price snapshots")
    capture_parser.add_argument("--mint", help="Default: first mint from the free creation stream")
    capture_parser.add_argument("--rpc", default="https://api.mainnet-beta.solana.com")
    capture_parser.add_argument("--count", type=int, default=120)
    capture_parser.add_argument("--interval", type=float, default=10)
    capture_parser.add_argument("--out", default="data/observations.csv")
    replay = commands.add_parser("replay", help="Train on earlier bars and evaluate on later bars")
    replay.add_argument("--edges", required=True)
    replay.add_argument("--bars", required=True)
    replay.add_argument("--cost-bps", type=float, default=150)
    replay.add_argument("--out", default="reports/replay.json")
    replay.add_argument("--activity-out", help="Record actual neuron states for the GUI")
    replay.add_argument("--watch-delay", type=float, default=0, help="Visual pacing per bar; capped at 20 seconds total")
    trial = commands.add_parser("trial", help="Run observable Pump.fun paper trials on saved or fresh prices")
    trial.add_argument("--edges", default="data/malecns-full.npz")
    trial.add_argument("--source", choices=["recordings", "live"], default="recordings")
    trial.add_argument("--continuous", action="store_true", help="Keep discovering live coins and learning until stopped; retain learning memory")
    trial.add_argument("--bars", action="append", default=[], help="Pump.fun recording; repeat for more coins")
    trial.add_argument("--coins", type=int, default=3)
    trial.add_argument("--mint", default="", help="Pump.fun coin URL or mint; use --coins 1 for a finite trial")
    trial.add_argument("--pin", action="store_true", help="Continuous mode: stay on the selected coin until Stop, including quiet periods")
    trial.add_argument("--profile", choices=["mushroom-body-reward-v1", "chart-context-v1", "legacy"], default="chart-context-v1")
    trial.add_argument("--reward-gain", type=float, default=1.5, help="Experimental positive dopamine teaching gain, 0–3")
    trial.add_argument("--exploration", type=float, default=.35, help="Probability of a small paper probe per eligible decision, 0–1")
    trial.add_argument("--min-cap-sol", type=float, default=100, help="Minimum price-times-supply value for new paper entries")
    trial.add_argument("--min-reserves-sol", type=float, default=10, help="Minimum real curve SOL reserves for new paper entries")
    trial.add_argument("--count", type=int, default=120)
    trial.add_argument("--interval", type=float, default=10)
    trial.add_argument("--rpc", default="https://api.mainnet-beta.solana.com")
    trial.add_argument("--capital", type=float, default=10)
    trial.add_argument("--cost-bps", type=float, default=150)
    trial.add_argument("--watch-delay", type=float, default=.12)
    trial.add_argument("--out", default="reports/trial.json")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            demo(args.out)
        elif args.command == "fetch-connectome":
            fetch_connectome(args)
        elif args.command == "prepare-connectome":
            prepare_connectome(args)
        elif args.command == "prepare-full":
            compile_full(args.weights, args.annotations, args.out)
        elif args.command == "capture":
            from .market import capture
            capture(args)
        elif args.command == "trial":
            from .trial import run_trial
            run_trial(args)
        else:
            provenance = run_record(args.edges, args.bars)
            bars = read_bars(args.bars)
            graph = load_graph(args.edges)
            recorder = None
            if args.activity_out:
                from .activity import ActivityRecorder
                recorder = ActivityRecorder(args.activity_out, args.edges, "data/malecns-annotations.feather",
                                            len(bars), args.watch_delay, bars_path=args.bars, connectivity=graph)
            try:
                report = evaluate(bars, graph, args.cost_bps, observer=recorder)
            except BaseException:
                if recorder:
                    recorder.finish("interrupted")
                raise
            if recorder:
                recorder.finish("completed")
                report["activity_file"] = str(Path(args.activity_out).resolve())
            if run_record(args.edges, args.bars) != provenance:
                raise ValueError("Inputs or model changed during replay; retry with stable files")
            report["provenance"] = provenance
            report["input_files"] = {"bars": str(Path(args.bars).resolve()),
                                     "edges": str(Path(args.edges).resolve())}
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, report)
            print(path.read_text())
    except KeyboardInterrupt:
        print("Operation cancelled. Any completed observations remain saved.")
    except (ValueError, OSError, TimeoutError, re.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
