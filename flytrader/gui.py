"""Loopback-only control panel for the existing paper research CLI."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.jobs = []
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.file_cache = {}
        self.graph_cache = {}
        self.summary_cache = {}
        self.evaluation_cache = None

    def file_info(self, path):
        meta = path.with_suffix(".metadata.json")
        key = (str(path), path.stat().st_mtime_ns, meta.stat().st_mtime_ns if meta.exists() else 0)
        if key in self.file_cache:
            return self.file_cache[key]
        if path.suffix == ".npz":
            first, kind, count = {}, "edges", 0
        else:
            with path.open() as stream:
                reader = csv.DictReader(stream)
                header = reader.fieldnames or []
                first = next(reader, {})
                kind = "bars" if "close" in header else "edges" if "bodyId_pre" in header else None
                count = (1 + sum(1 for _ in reader)) if first and kind == "bars" else 0
        metadata = {}
        if meta.exists() and meta.resolve().is_relative_to(self.root / "data"):
            try:
                metadata = json.loads(meta.read_text())
            except (ValueError, OSError):
                pass
        synthetic = "synthetic" in path.name
        product = metadata.get("product")
        mint = metadata.get("mint") or first.get("mint", "")
        label = (f"Imported prices · {product}" if product else "Synthetic demo" if synthetic else
                 f"Pump.fun · {mint[:6]}…{mint[-4:]}" if mint else path.stem)
        if kind == "bars":
            label += f" · {count:,} prices"
        else:
            if synthetic:
                label = "Synthetic graph"
            elif metadata.get("scope") == "full-retained":
                label = f"MaleCNS full retained · {metadata['matched_neurons']:,} neurons"
            elif metadata.get("dataset", "").startswith("male-cns"):
                label = f"MaleCNS · {metadata.get('matched_neurons', '?')} neurons"
                label += " · small circuit" if path.name == "malecns-edges.csv" else " · saved subset"
            else:
                label = f"Imported graph · {path.stem}"
        info = {"path": str(path.relative_to(self.root)), "kind": kind, "synthetic": synthetic,
                "metadata": metadata, "count": count, "label": label, "modified": path.stat().st_mtime,
                "ready": kind != "bars" or count >= 100,
                "pump_ready": kind == "bars" and count >= 100 and metadata.get("kind") == "reserve-price-snapshots"}
        self.file_cache = {k: v for k, v in self.file_cache.items() if k[0] != str(path)}
        self.file_cache[key] = info
        return info

    def brain(self, name):
        from .activity import graph_layout
        path = self.path(name)
        annotations = self.root / "data/malecns-annotations.feather"
        key = (str(path), path.stat().st_mtime_ns, annotations.stat().st_mtime_ns if annotations.exists() else 0)
        if key not in self.graph_cache:
            self.graph_cache = {key: graph_layout(path, annotations)}
        return self.graph_cache[key]

    def path(self, name, area="data"):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root / area):
            raise ValueError(f"Choose a file inside {area}/")
        return path

    def inventory(self):
        files = []
        for path in sorted([*(self.root / "data").rglob("*.csv"), *(self.root / "data").rglob("*.npz")]):
            if not path.resolve().is_relative_to(self.root / "data"):
                continue
            info = self.file_info(path)
            if info["kind"]:
                files.append(info)
        reports, trials, evaluations = [], [], []
        for path in sorted((self.root / "reports").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if ".activity." in path.name:
                continue
            if not path.resolve().is_relative_to(self.root / "reports"):
                continue
            try:
                report = json.loads(path.read_text())
                if report.get("mode") == "paper-replay":
                    reports.append({"path": str(path.relative_to(self.root)), "report": report})
                elif report.get("mode") == "reward-benchmark" and report.get("version") == 2:
                    evaluations.append({"path": str(path.relative_to(self.root)),
                                        **{key: report[key] for key in ("created", "data", "seeds", "settings", "aggregate", "summary")}})
                elif report.get("mode") == "paper-trial":
                    trial = self.trial(str(path.relative_to(self.root)), include_summary=False)
                    trials.append({"path": str(path.relative_to(self.root)),
                                   **{key: trial.get(key) for key in ("source", "status", "started", "updated", "neurons", "paper", "phase")},
                                   "coin_count": trial.get("coin_count", len(trial.get("coins", []))),
                                   "continuous": trial.get("continuous", False),
                                   "learning": {k: (trial.get("learning") or {}).get(k) for k in ("enabled", "session_updates", "examples")}})
            except (ValueError, OSError):
                pass
        with self.lock:
            jobs = [{k: v for k, v in j.items() if k != "process"} for j in self.jobs]
            for job in jobs:
                job["log"] = list(job["log"])
        files.sort(key=lambda f: f["modified"], reverse=True)
        default_graph = next((f["path"] for f in files if f["metadata"].get("scope") == "full-retained"),
                             next((f["path"] for f in files if f["kind"] == "edges" and not f["synthetic"]),
                                  next((f["path"] for f in files if f["kind"] == "edges"), "")))
        return {"files": files, "reports": reports[:50], "trials": trials[:30], "jobs": jobs, "default_graph": default_graph,
                "evaluations": evaluations[:5], "evaluation_plan": self.evaluation_plan(),
                "capabilities": ["chart-context-v1", "mushroom-body-reward-v1", "pin-coin", "pump-coin-url", "learning-check-v1"],
                "bulk_ready": (self.root / "data/malecns-weights.feather").exists(), "token": self.token}

    def evaluation_plan(self):
        from .evaluation import plan_evaluation
        paths = [*(self.root / "reports").glob("trial-*.json"), *(self.root / "data").glob("*.market.jsonl")]
        stamps = []
        for path in paths:
            if ".activity." not in path.name:
                try:
                    stat = path.stat()
                    stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
                except OSError:
                    pass
        key = tuple(sorted(stamps))
        with self.lock:
            if self.evaluation_cache is None or self.evaluation_cache[0] != key:
                self.evaluation_cache = (key, plan_evaluation(self.root))
            return self.evaluation_cache[1]

    def trial(self, name, include_summary=True):
        path = self.path(name, "reports")
        report = json.loads(path.read_text())
        if report.get("mode") != "paper-trial":
            raise ValueError("Choose a saved paper trial")
        if report.get("status") == "running":
            try:
                pid = int(report.get("pid", 0))
                if pid <= 0:
                    raise ProcessLookupError()
                os.kill(pid, 0)
            except (ProcessLookupError, ValueError):
                report["status"] = "interrupted"
                report["phase"] = "The trial process stopped. Saved activity and trades are still available."
                report["paper"]["pending"] = None
                report["paper"]["valuation_stale"] = bool(report["paper"].get("position"))
        if include_summary and report.get("status") != "running":
            from .results import session_summary, local_path
            dependencies = [path]
            for coin in report.get("coins", []):
                try:
                    dependencies.append(local_path(self.root, str(Path(coin["activity_path"]).with_suffix(".activity.json")), "reports"))
                except (ValueError, KeyError):
                    pass
            try:
                dependencies.append(local_path(self.root, report["learning"]["checkpoint"], "learning"))
            except (ValueError, KeyError, TypeError):
                pass
            stamps = []
            for dependency in dependencies:
                try:
                    stamps.append((str(dependency), dependency.stat().st_mtime_ns, dependency.stat().st_size))
                except OSError:
                    stamps.append((str(dependency), None, None))
            key = (str(path), report.get("status"), tuple(stamps))
            if key not in self.summary_cache:
                self.summary_cache = {k: v for k, v in self.summary_cache.items() if k[0] != str(path)}
                if len(self.summary_cache) >= 30:
                    self.summary_cache.pop(next(iter(self.summary_cache)))
                self.summary_cache[key] = session_summary(self.root, report)
            report["summary"] = self.summary_cache[key]
        return report

    def pump_recordings(self):
        files = self.inventory()["files"]
        seen, selected = set(), []
        for file in files:
            mint = file["metadata"].get("mint")
            if file["pump_ready"] and mint and mint not in seen:
                selected.append(file["path"])
                seen.add(mint)
        return selected

    def launch(self, body):
        action = body.get("action")
        public = body.get("public") is True
        if public and action not in ("trial", "replay"):
            raise ValueError("The public lab supports paper trials and replays only")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        command = [sys.executable, "-u", "-m", "flytrader.cli"]
        if action == "trial":
            edges = self.path(body.get("edges", "data/malecns-full.npz"))
            if not edges.is_file() or self.file_info(edges)["kind"] != "edges":
                raise ValueError("Choose an existing brain model")
            source = body.get("source", "recordings")
            continuous = body.get("continuous", False)
            if not isinstance(continuous, bool) or (continuous and source != "live"):
                raise ValueError("Continuous training requires live prices")
            profile = body.get("profile", "chart-context-v1")
            gain, exploration = float(body.get("reward_gain", 1.5)), float(body.get("exploration", .35))
            if profile not in ("chart-context-v1", "mushroom-body-reward-v1"):
                raise ValueError("Choose a supported training style")
            if not math.isfinite(gain) or not 0 <= gain <= 3 or not math.isfinite(exploration) or not 0 <= exploration <= 1:
                raise ValueError("Reward gain must be 0–3 and exploration 0–1")
            if profile == "mushroom-body-reward-v1" and not continuous:
                raise ValueError("Reward exploration requires continuous training")
            coins, count = int(body.get("coins", 3)), int(body.get("count", 120))
            interval, capital, cost = (float(body.get(k, v)) for k, v in (("interval", 10), ("capital", 10), ("cost", 150)))
            if source not in ("recordings", "live") or not 1 <= coins <= 5 or not 65 <= count <= 1000:
                raise ValueError("Choose recorded or live prices, one to five coins, and 65–1,000 observations")
            if not all(math.isfinite(v) for v in (interval, capital, cost)) or not 10 <= interval <= 3600 or not .01 <= capital <= 1000000 or not 0 <= cost < 10000:
                raise ValueError("Check the paper balance, interval and assumed costs")
            from .market import parse_coin
            mint = parse_coin(body.get("mint", "")) if source == "live" else ""
            pin = body.get("coin_policy") == "pin"
            if pin and (not continuous or not mint):
                raise ValueError("Stay on one coin requires continuous mode and a Pump.fun coin link or mint")
            min_cap, min_reserves = float(body.get("min_cap_sol", 100)), float(body.get("min_reserves_sol", 10))
            if not all(math.isfinite(v) and 0 <= v <= 1_000_000 for v in (min_cap, min_reserves)):
                raise ValueError("Market cap and reserve minimums must be finite SOL values from 0 to 1,000,000")
            if mint:
                from .market import valid_mint
                if not valid_mint(mint) or (coins != 1 and not continuous):
                    raise ValueError("For a specific mint, enter a valid address and select one coin")
            recordings = self.pump_recordings()[:coins] if source == "recordings" else []
            if source == "recordings" and not recordings:
                raise ValueError("No ready Pump.fun recordings. Select Live prices to collect and test a new coin")
            if public:
                if source not in ("recordings", "live"):
                    raise ValueError("Public trials must use recorded or live prices")
                if coins > 5 or count > 1000 or capital > 1000000 or cost >= 10000 or interval < 10:
                    raise ValueError("Public trials must use bounded parameters")
            output = f"reports/{'public-' if public else ''}trial-{stamp}.json"
            command += ["trial", "--edges", str(edges), "--source", source, "--coins", str(coins),
                        "--count", str(count), "--interval", str(interval), "--capital", str(capital),
                        "--cost-bps", str(cost), "--out", output]
            command += ["--min-cap-sol", str(min_cap), "--min-reserves-sol", str(min_reserves)]
            command += ["--profile", profile, "--reward-gain", str(gain), "--exploration", str(exploration)]
            if continuous:
                command += ["--continuous"]
            if pin:
                command += ["--pin"]
            if mint:
                command += ["--mint", mint]
            for path in recordings:
                command += ["--bars", str(self.path(path))]
        elif action == "learning-check":
            edges = self.path(body.get("edges", "data/malecns-full.npz"))
            if not edges.is_file() or self.file_info(edges)["metadata"].get("scope") != "full-retained":
                raise ValueError("Select the full retained MaleCNS model for a learning check")
            cost, interval, gain, exploration = (float(body.get(k, default)) for k, default in
                                                (("cost", 150), ("interval", 10), ("reward_gain", 1.5), ("exploration", .35)))
            min_cap, min_reserves = float(body.get("min_cap_sol", 100)), float(body.get("min_reserves_sol", 10))
            if (not all(math.isfinite(v) for v in (cost, interval, gain, exploration)) or
                    not 0 <= cost < 10000 or not 10 <= interval <= 3600 or not 0 <= gain <= 3 or not 0 <= exploration <= 1 or
                    not all(math.isfinite(v) and 0 <= v <= 1_000_000 for v in (min_cap, min_reserves))):
                raise ValueError("Check the evaluation costs, interval and learning settings")
            plan = self.evaluation_plan()
            if not plan["ready"]:
                raise ValueError(plan["reason"])
            output = f"reports/learning-check-{stamp}.json"
            command = [sys.executable, "-u", "-m", "flytrader.evaluation", "--edges", str(edges),
                       "--cost-bps", str(cost), "--interval", str(interval), "--reward-gain", str(gain),
                       "--exploration", str(exploration), "--min-cap-sol", str(min_cap), "--min-reserves-sol", str(min_reserves), "--out", output]
        elif action == "replay":
            edges = self.path(body["edges"])
            bars = self.path(body["bars"])
            if not edges.is_file() or not bars.is_file():
                raise ValueError("Select an existing graph and observations file")
            if not self.file_info(bars)["ready"]:
                raise ValueError("This recording needs at least 100 prices. Collect more data or choose another recording.")
            cost = float(body.get("cost", 150))
            if not math.isfinite(cost) or not 0 <= cost < 10000:
                raise ValueError("Cost must be between 0 and 9,999 basis points")
            if public and (cost > 1000 or body.get("watch", False)):
                raise ValueError("Public replays must use bounded costs and no visual pacing")
            output = f"reports/{'public-' if public else ''}replay-{stamp}.json"
            command += ["replay", "--edges", str(edges), "--bars", str(bars), "--cost-bps", str(cost), "--out", output]
            command += ["--activity-out", str(Path(output).with_suffix(".activity.json")),
                        "--watch-delay", ".06" if body.get("watch", True) else "0"]
        elif action == "capture":
            count, interval = int(body.get("count", 120)), float(body.get("interval", 10))
            if not 1 <= count <= 10000 or not math.isfinite(interval) or not 10 <= interval <= 3600:
                raise ValueError("Use 1–10,000 observations and a 10–3,600 second interval")
            output = f"data/capture-{stamp}.csv"
            command += ["capture", "--count", str(count), "--interval", str(interval), "--out", output]
            mint = str(body.get("mint", "")).strip()
            if mint:
                if not 32 <= len(mint) <= 44 or any(c not in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in mint):
                    raise ValueError("Enter a valid Solana mint address, or leave it blank")
                command += ["--mint", mint]
        elif action == "prepare-full":
            output = f"data/malecns-full-{stamp}.npz"
            command += ["prepare-full", "--out", output]
        elif action == "prepare":
            pattern = str(body.get("pattern", "^(MBON|APL|PAM|PPL)"))
            if not pattern or len(pattern) > 200:
                raise ValueError("Enter a cell-type expression (up to 200 characters)")
            limit = int(body.get("limit", 2000))
            if not 1 <= limit <= 2000:
                raise ValueError("Small circuits support up to 2,000 cells; use Build full model for the complete retained graph")
            output = f"data/connectome-{stamp}.csv"
            command += ["prepare-connectome", "--type-regex", pattern, "--max-neurons", str(limit), "--out", output]
        elif action == "demo":
            output = f"data/demo-{stamp}"
            command += ["demo", "--out", output]
        else:
            raise ValueError("Unknown action")
        with self.lock:
            if any(j["status"] in ("running", "stopping") for j in self.jobs):
                raise ValueError("Stop or finish the current operation first")
            environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, env=environment,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            job = {"id": stamp, "action": action, "status": "running", "output": output, "public": public,
                   "started": datetime.now(timezone.utc).isoformat(), "log": deque(maxlen=300), "process": process}
            self.jobs.insert(0, job)
            del self.jobs[30:]
            threading.Thread(target=self.watch, args=(job,), daemon=True).start()
        return {"id": stamp, "output": output}

    def watch(self, job):
        process = job["process"]
        for line in process.stdout:
            with self.lock:
                job["log"].append(line.rstrip())
        process.stdout.close()
        code = process.wait()
        with self.lock:
            job["status"] = "cancelled" if job["status"] == "stopping" else "completed" if code == 0 else "failed"
            job["exit_code"] = code
        if job.get("public"):
            self.prune_public_outputs()

    def prune_public_outputs(self, max_files=40, max_bytes=200 * 1024 * 1024):
        """Bound disk use from the anonymous public paper-trading lab."""
        reports = (self.root / "reports").resolve()
        paths = []
        for path in reports.glob("public-*"):
            try:
                resolved = path.resolve()
                if resolved.is_file() and resolved.is_relative_to(reports):
                    stat = resolved.stat()
                    paths.append((stat.st_mtime_ns, stat.st_size, resolved))
            except OSError:
                continue
        paths.sort(reverse=True)
        retained_bytes = 0
        for index, (_, size, path) in enumerate(paths):
            retained_bytes += size
            if index < max_files and retained_bytes <= max_bytes:
                continue
            try:
                path.unlink()
            except OSError:
                pass

    def stop(self):
        with self.lock:
            for job in self.jobs:
                if job["status"] == "running" and job["process"].poll() is None:
                    job["status"] = "stopping"
                    job["process"].send_signal(signal.SIGINT)
        return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, status, value, mime="application/json"):
        data = json.dumps(value, allow_nan=False).encode() if mime == "application/json" else value
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def trusted(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.trusted():
            return self.respond(403, {"error": "Local access only"})
        url = urlparse(self.path)
        try:
            if url.path == "/api/state":
                return self.respond(200, self.server.workspace.inventory())
            if url.path == "/api/brain":
                return self.respond(200, self.server.workspace.brain(parse_qs(url.query).get("edges", [""])[0]))
            if url.path == "/api/trial":
                return self.respond(200, self.server.workspace.trial(parse_qs(url.query).get("path", [""])[0]))
            if url.path == "/api/activity":
                query = parse_qs(url.query)
                path = self.server.workspace.path(query.get("path", [""])[0], "reports")
                path = path.with_suffix(".activity.live.json" if query.get("live") == ["1"] else ".activity.json")
                if not path.resolve().is_relative_to(self.server.workspace.root / "reports"):
                    raise ValueError("Choose a local replay")
                return self.respond(200, json.loads(path.read_text()) if path.exists() else {"available": False})
            if url.path == "/api/bars":
                path = self.server.workspace.path(parse_qs(url.query).get("path", [""])[0])
                rows = []
                with path.open() as stream:
                    for row in csv.DictReader(stream):
                        point = [float(row["timestamp"]), float(row["close"])]
                        if all(math.isfinite(v) for v in point):
                            rows.append(point)
                        if len(rows) >= 100000:
                            break
                budget = 2000 if parse_qs(url.query).get("full") == ["1"] else 400
                indices = list(range(0, len(rows), max(1, math.ceil(len(rows) / budget))))
                if rows and indices[-1] != len(rows) - 1:
                    indices.append(len(rows) - 1)
                return self.respond(200, {"count": len(rows), "points": [rows[i] for i in indices], "steps": [i + 1 for i in indices]})
            name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/brain.js": "brain.js", "/trial.js": "trial.js", "/chart.js": "chart.js", "/review.js": "review.js"}.get(url.path)
            if name:
                mime = {"html": "text/html; charset=utf-8", "css": "text/css", "js": "text/javascript"}[name.split(".")[-1]]
                return self.respond(200, (STATIC / name).read_bytes(), mime)
            self.respond(404, {"error": "Not found"})
        except (ValueError, OSError, KeyError) as exc:
            self.respond(400, {"error": str(exc)})

    def do_POST(self):
        if not self.trusted() or self.headers.get("X-Flytrader-Token") != self.server.workspace.token:
            return self.respond(403, {"error": "Refresh the local control panel"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 8192:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected an object")
            if self.path == "/api/jobs":
                result = self.server.workspace.launch(body)
            elif self.path == "/api/stop":
                result = self.server.workspace.stop()
            else:
                return self.respond(404, {"error": "Not found"})
            self.respond(200, result)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.respond(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Flytrader local GUI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.workspace = Workspace(args.workspace)
    print(f"Flytrader GUI: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.workspace.stop()
        server.server_close()


if __name__ == "__main__":
    main()
