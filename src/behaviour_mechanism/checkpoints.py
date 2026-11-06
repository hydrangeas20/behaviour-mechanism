"""Save/load parameter trees as ``.npz`` (flattened ``a/b/c`` keys) with a sidecar manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


def _flatten(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) or hasattr(v, "items"):
            out.update(_flatten(v, key + "/"))
        else:
            out[key] = np.asarray(v)
    return out


def save_params(path: str | Path, params: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **_flatten(params))  # type: ignore[arg-type]


def load_params(path: str | Path) -> dict[str, Any]:
    flat = np.load(path)
    tree: dict[str, Any] = {}
    for key in flat.files:
        node = tree
        parts = key.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(flat[key])
    return tree
