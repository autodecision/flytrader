"""Whole-session results, with honest coverage for older sampled recordings."""
from datetime import datetime
import json
import math
from pathlib import Path


def new_diagnostics():
    return {"observations": 0, "training": 0, "forecasts": 0, "qualifying_signals": 0,
            "max_forecast": None, "price_intervals": 0, "changed_prices": 0, "decisions": {}}


def record_diagnostics(data, price, previous_price, paper, threshold):
    data["observations"] += 1
    prediction = paper.get("prediction")
    if prediction is None:
        data["training"] += 1
    else:
        data["forecasts"] += 1
        data["qualifying_signals"] += int(prediction > threshold)
        data["max_forecast"] = prediction if data["max_forecast"] is None else max(data["max_forecast"], prediction)
    decision = paper.get("decision", "Not recorded")
    data["decisions"][decision] = data["decisions"].get(decision, 0) + 1
    if previous_price is not None:
        data["price_intervals"] += 1
        data["changed_prices"] += int(price != previous_price)


def local_path(root, name, area):
    path = (root / name).resolve()
    if not path.is_relative_to(root / area):
        raise ValueError(f"Expected a file under {area}/")
    return path


def legacy_diagnostics(root, report, threshold):
    """Read old activity without rewriting reports or treating samples as all observations."""
    data = new_diagnostics()
    learning_start = None
    for coin in report.get("coins", []):
        try:
            path = local_path(root, str(Path(coin["activity_path"]).with_suffix(".activity.json")), "reports")
            payload = json.loads(path.read_text())
        except (OSError, ValueError, KeyError):
            continue
        previous, previous_step = None, None
        frames = payload.get("frames", [])
        if coin.get("index") == 0 and frames and frames[0].get("step") == 1:
            learning_start = frames[0].get("learning")
        for frame in frames:
            if "paper" not in frame or "price" not in frame:
                continue
            step = frame.get("step")
            # Only adjacent observations establish a one-observation price change.
            record_diagnostics(data, frame["price"], previous if previous_step is not None and step == previous_step + 1 else None,
                               frame["paper"], threshold)
            previous, previous_step = frame["price"], step
    return data, learning_start


def learning_comparison(end, start):
    """Subtract the starting cumulative errors so resumed sessions get their own score."""
    scope = "session" if start is not None else "all saved learning"
    count = end.get("evaluated", 0)
    model, baseline = end.get("prediction_mae"), end.get("baseline_mae")
    if model is None or baseline is None:
        return {"assessment": "insufficient", "count": 0, "scope": scope}
    model_total, baseline_total = model * count, baseline * count
    if start is not None:
        previous_count = start.get("evaluated", 0)
        count -= previous_count
        model_total -= (start.get("prediction_mae") or 0) * previous_count
        baseline_total -= (start.get("baseline_mae") or 0) * previous_count
    if count <= 0:
        return {"assessment": "insufficient", "count": 0, "scope": scope}
    model, baseline = max(0, model_total / count), max(0, baseline_total / count)
    assessment = "insufficient" if count < 20 else "same" if math.isclose(model, baseline, rel_tol=1e-8, abs_tol=1e-12) else "better" if model < baseline else "worse"
    return {"assessment": assessment, "count": count, "scope": scope,
            "model_error": model, "baseline_error": baseline,
            "error_ratio": model / baseline if baseline > 0 else None}


def checkpoint_status(root, report):
    learning = report.get("learning") or {}
    if not learning.get("enabled"):
        return "not_used"
    if not learning.get("examples"):
        return "no_updates"
    try:
        import numpy as np
        path = local_path(root, learning["checkpoint"], "learning")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata.get("version") == 2 and metadata.get("profile") == "mushroom-body-reward-v1":
                weights, scales = saved["weights"], saved["scales"]
                if (weights.ndim != 2 or weights.shape[0] != 3 or scales.ndim != 1 or
                        not np.isfinite(weights).all() or not np.isfinite(scales).all() or
                        np.any(scales < .25) or np.any(scales > 1) or
                        not np.isfinite(saved["familiarity"]).all()):
                    return "unverified"
            else:
                gram, target = saved["gram"], saved["target"]
                if (metadata.get("version") != 1 or target.ndim != 1 or gram.shape != (len(target), len(target)) or
                        not np.isfinite(gram).all() or not np.isfinite(target).all()):
                    return "unverified"
        graph = report.get("graph_fingerprint", {}).get("sha256")
        identity = metadata.get("identity", {})
        if metadata["examples"] < learning["examples"] or (graph and identity.get("graph") != graph):
            return "unverified"
        return "verified"
    except FileNotFoundError:
        return "missing"
    except (OSError, ValueError, KeyError, TypeError):
        return "unverified"


def session_summary(root, report):
    root = Path(root).resolve()
    paper = report.get("paper") or {}
    learning = report.get("learning") or {}
    threshold = report.get("entry_threshold_log_return", .04)
    data = report.get("diagnostics")
    start = report.get("learning_start")
    if data is None:
        data, start = legacy_diagnostics(root, report, threshold)
    observations = report.get("observations", 0)
    coverage = "complete" if data["observations"] == observations else "sampled" if data["observations"] else "unavailable"
    # Cumulative broker counters remain authoritative even when the recent ledger is capped.
    orders = paper.get("orders", len(report.get("events", [])))
    sells = paper.get("closed_trades", sum(e.get("side") == "SELL" for e in report.get("events", [])))
    buys = orders - sells
    elapsed = None
    try:
        end = report.get("finished") or report.get("updated")
        elapsed = max(0, (datetime.fromisoformat(end) - datetime.fromisoformat(report["started"])).total_seconds())
    except (ValueError, TypeError, KeyError):
        pass
    position_count = len(paper.get("unresolved", [])) + int(bool(paper.get("position")))
    reward_mode = report.get("model_profile") == "mushroom-body-reward-v1"
    if orders:
        reason = "traded"
    elif not observations:
        reason = "no_data"
    elif not data["observations"]:
        reason = "unknown"
    elif not data["forecasts"]:
        reason = "training" if coverage == "complete" else "unknown"
    elif reward_mode:
        reason = "reward_waiting"
    elif data["qualifying_signals"] == 0:
        reason = "below_threshold" if coverage == "complete" else "sampled_below_threshold"
    elif paper.get("halted"):
        reason = "risk_halt"
    elif any(key.startswith(("Paper entry paused", "Pinned coin is quiet")) for key in data["decisions"]):
        reason = "risk_checks"
    else:
        reason = "no_fills"
    intervals = data["price_intervals"]
    return {"version": 1, "duration_seconds": elapsed, "observations": observations,
            "coins": report.get("coin_count", len(report.get("coins", []))),
            "money": {**{k: paper.get(k, 0) for k in ("initial_sol", "cash_sol", "equity_sol", "pnl_sol", "return_pct", "realized_sol", "unrealized_sol", "costs_paid_sol")},
                      "orders": orders, "buys": buys, "sells": sells, "closed_trades": sells,
                      "open_positions": position_count, "valuation_stale": bool(paper.get("valuation_stale"))},
            "learning": {"enabled": bool(learning.get("enabled")), "new_updates": learning.get("session_updates", 0),
                         "profile": report.get("model_profile"),
                         "total_updates": learning.get("examples", 0), "restored_updates": learning.get("resumed_examples", 0),
                         "checkpoint_status": checkpoint_status(root, report),
                         "comparison": learning_comparison(learning, start)},
            "activity": {**data, "coverage": coverage, "entry_threshold": threshold,
                         "unchanged_price_pct": (1 - data["changed_prices"] / intervals) * 100 if intervals else None},
            "no_trade_reason": reason, "cost_bps_per_side": report.get("cost_bps_per_side"),
            "holding_period_seconds": report.get("interval_seconds") if report.get("source") == "live" and not reward_mode else None,
            "holding_periods_seconds": report.get("holding_periods_seconds"),
            "end_status": report.get("status")}
