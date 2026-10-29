"""Reproducible run records: exact input bytes, model sources and environment."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def run_record(edges, bars):
    source = Path(__file__).parent
    try:
        metadata = json.loads(Path(edges).with_suffix(".metadata.json").read_text())
    except (OSError, ValueError):
        metadata = {}
    return {"inputs": {"edges": fingerprint(edges), "bars": fingerprint(bars)},
            "model_source": fingerprint(source / "model.py"),
            "connectome_source": fingerprint(source / "connectome.py"),
            "activity_source": fingerprint(source / "activity.py"),
            "cli_source": fingerprint(source / "cli.py"),
            "python": platform.python_version(), "numpy": version("numpy"),
            "scipy": version("scipy"), "seed": 7,
            "graph_scope": metadata.get("scope", "selected subset"), "dynamics": "unsigned normalized tanh reservoir"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
