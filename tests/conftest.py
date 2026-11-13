from pathlib import Path

import pytest

from behaviour_mechanism.config import ExperimentConfig, load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def main_cfg() -> ExperimentConfig:
    return load_config(REPO / "configs" / "config.json")


@pytest.fixture(scope="session")
def tiny_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        vocab_size=20,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        max_seq_len=16,
        seq_len=16,
        batch_size=8,
        total_steps=20,
        dense_ckpt_every=10,
        dense_ckpt_until=20,
        sparse_ckpt_every=20,
        seeds=(0,),
        eval_batches=2,
        eval_batch_size=8,
    )
