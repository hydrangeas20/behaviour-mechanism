"""Corrected activation-patching measurement (metric version ``bm-patch-2``).

Why a new API
-------------
The historical statistic (``patching.legacy_layer_effects``, kept unchanged and reported
as ``legacy_patch_effect_size_v1``) has three properties that make it unsafe to read as
"mechanism strength":

1. **Denominator.** Pairs whose clean/corrupted gap is below the threshold are dropped
   from the numerator but kept in the denominator, so undefined pairs count as 0.
2. **Fixed site.** It only patches attention outputs. A mechanism whose distinguishing
   computation runs through the residual stream and an MLP produces *no* clean/corrupted
   difference at any attention output. Restoring those outputs then changes nothing:
   recovery is exactly 0 with every pair defined. ``bm stage1-controls``
   (``results/stage1/positive_controls/``) demonstrates this on the MLP-gated constructed copier. It is a site-coverage problem,
   not an arithmetic bug.
3. **Selection.** The headline is the maximum over layers.

What this module reports
------------------------
For every (site, direction) requested, a :class:`MeasurementResult` records:

- attempted, defined and undefined pair counts and the defined fraction;
- the conditional effect (mean of clipped per-pair effects, median, unclipped mean) over
  *defined* pairs only. It is ``None``, never 0, when no pair is defined;
- the mean clean-vs-corrupted activation difference at the site and read position, so
  "the site carries no difference" can be told apart from "the site carries a difference
  that does not matter";
- the metric version, site, direction and corruption, so serialised results are
  self-describing.

Directions
----------
``denoise``: run the corrupted input with the clean activation at ``site``.
``effect = (lp - lp_corrupt) / gap``, the fraction of the gap restored (sufficiency).

``noise``: run the clean input with the corrupted activation at ``site``.
``effect = (lp_clean - lp) / gap``, the fraction of the gap destroyed (necessity).

A detected effect shows that the site *mediates* the clean/corrupted difference under this
corruption. It does not show that the site is the unique or complete implementation of
a mechanism.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .config import ExperimentConfig
from .model import Cache, Patch, forward

METRIC_VERSION = "bm-patch-2"
LEGACY_NAME = "legacy_patch_effect_size_v1"
DEFAULT_GAP_THRESHOLD = 0.05
DEFAULT_CLIP = 5.0

Component = Literal["attn_out", "head_out", "mlp_out", "resid_post"]
Direction = Literal["denoise", "noise"]
COMPONENTS: tuple[str, ...] = ("attn_out", "head_out", "mlp_out", "resid_post")
DIRECTIONS: tuple[str, ...] = ("denoise", "noise")


@dataclass(frozen=True)
class Site:
    """An intervention target.

    ``attn_out``, ``head_out`` and ``mlp_out`` are replaced at **all** positions (the
    historical convention). ``resid_post`` is replaced at each pair's read position only.
    """

    component: str
    layer: int
    head: int | None = None

    def __post_init__(self) -> None:
        if self.component not in COMPONENTS:
            raise ValueError(f"unknown component {self.component!r}")
        if (self.component == "head_out") != (self.head is not None):
            raise ValueError("head must be given for head_out and only for head_out")

    @property
    def name(self) -> str:
        if self.component == "head_out":
            return f"head_out.L{self.layer}H{self.head}"
        return f"{self.component}.L{self.layer}"

    @property
    def positions(self) -> str:
        return "read_position" if self.component == "resid_post" else "all"

    def validate(self, cfg: ExperimentConfig) -> None:
        if not 0 <= self.layer < cfg.n_layers:
            raise ValueError(f"{self.name}: layer out of range for {cfg.n_layers} layers")
        if self.head is not None and not 0 <= self.head < cfg.n_heads:
            raise ValueError(f"{self.name}: head out of range for {cfg.n_heads} heads")


def standard_sites(cfg: ExperimentConfig, heads: bool = True) -> list[Site]:
    sites = [Site("attn_out", l) for l in range(cfg.n_layers)]
    sites += [Site("mlp_out", l) for l in range(cfg.n_layers)]
    sites += [Site("resid_post", l) for l in range(cfg.n_layers)]
    if heads:
        sites += [Site("head_out", l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    return sites


@dataclass(frozen=True)
class PairSet:
    """Clean/corrupted inputs. ``read_pos[i]`` is the position whose output predicts
    ``target[i]``."""

    clean: np.ndarray
    corrupt: np.ndarray
    target: np.ndarray
    read_pos: np.ndarray
    corruption: str

    def validate(self, cfg: ExperimentConfig) -> None:
        n = self.clean.shape[0]
        if n == 0:
            raise ValueError("empty pair set")
        if self.clean.shape != self.corrupt.shape or self.clean.ndim != 2:
            raise ValueError("clean and corrupt must both have shape (n, T)")
        if self.target.shape != (n,) or self.read_pos.shape != (n,):
            raise ValueError("target and read_pos must have shape (n,)")
        t = self.clean.shape[1]
        if t > cfg.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        if (self.read_pos < 0).any() or (self.read_pos >= t).any():
            raise ValueError("read_pos out of range")
        for name, arr in (
            ("clean", self.clean),
            ("corrupt", self.corrupt),
            ("target", self.target),
        ):
            if (arr < 0).any() or (arr >= cfg.vocab_size).any():
                raise ValueError(f"{name} contains token ids outside the vocabulary")


@dataclass
class MeasurementResult:
    metric_version: str
    site: str
    component: str
    layer: int
    head: int | None
    positions: str
    direction: str
    corruption: str
    n_attempted: int
    n_defined: int
    n_undefined: int
    defined_fraction: float
    conditional_effect_mean: float | None
    conditional_effect_median: float | None
    conditional_effect_unclipped_mean: float | None
    gap_median: float
    activation_delta_mean: float
    gap_threshold: float
    clip: float
    per_pair_effect: list[float | None] = field(default_factory=list)
    per_pair_gap: list[float] = field(default_factory=list)

    def to_dict(self, per_pair: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not per_pair:
            d.pop("per_pair_effect")
            d.pop("per_pair_gap")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeasurementResult:
        if d.get("metric_version") != METRIC_VERSION:
            raise ValueError(f"unsupported metric version {d.get('metric_version')!r}")
        return cls(**d)


def to_json(results: list[MeasurementResult], per_pair: bool = True) -> str:
    """Strict JSON (no NaN): undefined values are ``null``."""
    return json.dumps([r.to_dict(per_pair) for r in results], allow_nan=False, indent=1)


def from_json(text: str) -> list[MeasurementResult]:
    return [MeasurementResult.from_dict(d) for d in json.loads(text)]


def _patch(site: Site, src: Cache, read_mask: jnp.ndarray) -> Patch:
    if site.component == "attn_out":
        return Patch(layer_attn_out={site.layer: src.layer_attn_out[site.layer]})
    if site.component == "mlp_out":
        return Patch(mlp_out={site.layer: src.mlp_out[site.layer]})
    if site.component == "head_out":
        assert site.head is not None
        return Patch(head_out={(site.layer, site.head): src.head_out[site.layer][:, site.head]})
    return Patch(resid_post_masked={site.layer: (read_mask, src.resid_post[site.layer])})


def _site_activation(site: Site, c: Cache, read_pos: jnp.ndarray) -> jnp.ndarray:
    idx = jnp.arange(read_pos.shape[0])
    if site.component == "attn_out":
        a = c.layer_attn_out[site.layer]
    elif site.component == "mlp_out":
        a = c.mlp_out[site.layer]
    elif site.component == "resid_post":
        a = c.resid_post[site.layer]
    else:
        assert site.head is not None
        a = c.head_out[site.layer][:, site.head]
    return a[idx, read_pos]


@partial(jax.jit, static_argnums=(0, 1))
def _measure_lps(
    cfg: ExperimentConfig,
    sites: tuple[Site, ...],
    params: Any,
    clean: jnp.ndarray,
    corrupt: jnp.ndarray,
    target: jnp.ndarray,
    read_pos: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    idx = jnp.arange(target.shape[0])
    read_mask = jnp.arange(clean.shape[1])[None, :] == read_pos[:, None]

    def lp(logits: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.log_softmax(logits[idx, read_pos], axis=-1)[idx, target]

    cc, xc = Cache(), Cache()
    out = {
        "lp_clean": lp(forward(params, cfg, clean, cache=cc)),
        "lp_corrupt": lp(forward(params, cfg, corrupt, cache=xc)),
    }
    for s in sites:
        out[f"denoise|{s.name}"] = lp(forward(params, cfg, corrupt, patch=_patch(s, cc, read_mask)))
        out[f"noise|{s.name}"] = lp(forward(params, cfg, clean, patch=_patch(s, xc, read_mask)))
        diff = _site_activation(s, cc, read_pos) - _site_activation(s, xc, read_pos)
        out[f"delta|{s.name}"] = jnp.linalg.norm(diff, axis=-1)
    return out


def _clean_float(x: float) -> float | None:
    return None if (x is None or math.isnan(x)) else float(x)


def measure(
    params: Any,
    cfg: ExperimentConfig,
    pairs: PairSet,
    sites: list[Site],
    directions: tuple[str, ...] = DIRECTIONS,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    clip: float = DEFAULT_CLIP,
) -> list[MeasurementResult]:
    if gap_threshold <= 0 or clip <= 0:
        raise ValueError("gap_threshold and clip must be positive")
    if not sites:
        raise ValueError("no sites requested")
    for d in directions:
        if d not in DIRECTIONS:
            raise ValueError(f"unknown direction {d!r}")
    pairs.validate(cfg)
    for s in sites:
        s.validate(cfg)
    raw = _measure_lps(
        cfg,
        tuple(sites),
        params,
        jnp.asarray(pairs.clean),
        jnp.asarray(pairs.corrupt),
        jnp.asarray(pairs.target),
        jnp.asarray(pairs.read_pos),
    )
    res = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
    if not all(np.isfinite(v).all() for v in res.values()):
        raise FloatingPointError("non-finite log-probabilities or activations")
    lc, lx = res["lp_clean"], res["lp_corrupt"]
    gap = lc - lx
    defined = np.abs(gap) >= gap_threshold
    n = int(gap.shape[0])
    out: list[MeasurementResult] = []
    for s in sites:
        for d in directions:
            lp = res[f"{d}|{s.name}"]
            with np.errstate(divide="ignore", invalid="ignore"):
                eff = (lp - lx) / gap if d == "denoise" else (lc - lp) / gap
            eff_def = eff[defined]
            out.append(
                MeasurementResult(
                    metric_version=METRIC_VERSION,
                    site=s.name,
                    component=s.component,
                    layer=s.layer,
                    head=s.head,
                    positions=s.positions,
                    direction=d,
                    corruption=pairs.corruption,
                    n_attempted=n,
                    n_defined=int(defined.sum()),
                    n_undefined=int(n - defined.sum()),
                    defined_fraction=float(defined.mean()),
                    conditional_effect_mean=(
                        float(np.clip(eff_def, -clip, clip).mean()) if eff_def.size else None
                    ),
                    conditional_effect_median=float(np.median(eff_def)) if eff_def.size else None,
                    conditional_effect_unclipped_mean=float(eff_def.mean())
                    if eff_def.size
                    else None,
                    gap_median=float(np.median(gap)),
                    activation_delta_mean=float(res[f"delta|{s.name}"].mean()),
                    gap_threshold=gap_threshold,
                    clip=clip,
                    per_pair_effect=[
                        _clean_float(e) if ok else None
                        for e, ok in zip(eff.tolist(), defined, strict=True)
                    ],
                    per_pair_gap=[float(g) for g in gap],
                )
            )
    return out


def legacy_statistic(results: list[MeasurementResult], n_layers: int) -> dict[str, Any]:
    """Recompute ``legacy_patch_effect_size_v1`` from corrected per-pair outputs.

    Uses the historical definition exactly: attention-output denoising per layer, clipped,
    summed over defined pairs, divided by *all* attempted pairs, maximum over layers.
    """
    per_layer = []
    for l in range(n_layers):
        r = next(
            x
            for x in results
            if x.component == "attn_out" and x.layer == l and x.direction == "denoise"
        )
        vals = [min(max(e, -r.clip), r.clip) for e in r.per_pair_effect if e is not None]
        per_layer.append(sum(vals) / r.n_attempted)
    best = int(np.argmax(per_layer))
    return {
        "name": LEGACY_NAME,
        "value": float(per_layer[best]),
        "best_layer": best,
        "per_layer": [float(v) for v in per_layer],
    }
