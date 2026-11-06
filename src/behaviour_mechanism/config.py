"""Experiment configuration (faithful copy of the notebook's ``ExperimentConfig``).

The ``config_hash`` is computed exactly as in the notebook, so the committed
``configs/config.json`` hashes to ``57c45749eb6d`` -- the hash printed in the
notebook output that produced ``results/*.csv``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Historical, outcome-relevant constant from the notebook training objective.
INDUCTION_LOSS_WEIGHT = 3.0


@dataclass(frozen=True)
class ExperimentConfig:
    # model
    vocab_size: int = 100
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 64
    dropout_rate: float = 0.0  # recorded but unused: the model has no dropout layers
    # data
    seq_len: int = 64
    batch_size: int = 64
    # NOTE: induction_gap_min / induction_gap_max are recorded in the config (and in its
    # hash) but are NOT read by the data generator. Every induction sequence repeats its
    # first half at a fixed offset of seq_len // 2. See docs/RESEARCH_AUDIT.md (P0-1).
    induction_gap_min: int = 2
    induction_gap_max: int = 40
    frac_induction_sequences: float = 0.7
    # optimisation
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    total_steps: int = 8000
    # checkpointing
    dense_ckpt_every: int = 50
    dense_ckpt_until: int = 1500
    sparse_ckpt_every: int = 500
    # seeds
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    # eval
    eval_batches: int = 8
    eval_batch_size: int = 64

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.seq_len > self.max_seq_len:
            raise ValueError("seq_len must not exceed max_seq_len")
        if self.seq_len < 4:
            raise ValueError("seq_len must be at least 4 (the probe uses position half + 1)")
        if not 0.0 <= self.frac_induction_sequences <= 1.0:
            raise ValueError("frac_induction_sequences must be in [0, 1]")

    def checkpoint_steps(self) -> list[int]:
        dense = list(range(0, self.dense_ckpt_until + 1, self.dense_ckpt_every))
        sparse = list(range(self.dense_ckpt_until, self.total_steps + 1, self.sparse_ckpt_every))
        return sorted(set(dense + sparse + [self.total_steps]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def replace(self, **changes: Any) -> ExperimentConfig:
        return dataclasses.replace(self, **changes)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text())
    known = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "seeds" in raw:
        raw["seeds"] = tuple(int(s) for s in raw["seeds"])
    return ExperimentConfig(**raw)
