"""Run manifests: what produced an artifact, recorded at the time it was produced."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        return out  # "" is a valid answer (e.g. a clean `git status --porcelain`)
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD") or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": None if status is None else bool(status),
    }


def software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for mod in ("jax", "jaxlib", "flax", "optax", "numpy", "pandas", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import jax

        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        info["jax_backend"] = None
    return info


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "command": sys.argv,
        "git": git_state(),
        "software": software_versions(),
        "hardware": hardware(),
        **fields,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
