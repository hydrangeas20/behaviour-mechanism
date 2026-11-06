import json

import numpy as np
import pytest
from jax import random

from behaviour_mechanism.config import ExperimentConfig, load_config
from behaviour_mechanism.data import (
    make_clean_corrupt_pair,
    make_corrupt_copy_batch,
    make_induction_batch,
    make_offset_probe_batch,
)


def test_committed_config_is_the_historical_run(main_cfg):
    # Hashes printed in the notebook outputs for the main run and the negative control.
    assert main_cfg.config_hash() == "57c45749eb6d"
    neg = main_cfg.replace(frac_induction_sequences=0.0, seeds=(0, 1))
    assert neg.config_hash() == "584a4b4b2a1c"
    assert len(main_cfg.checkpoint_steps()) == 44


def test_load_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"vocab_size": 10, "typo_field": 1}))
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)


def test_config_validation():
    with pytest.raises(ValueError):
        ExperimentConfig(d_model=10, n_heads=4)


def test_first_batch_matches_notebook_output(main_cfg):
    # Notebook cell 11 printed this sample for PRNGKey(0), batch size 4.
    tok, mask, tpos = make_induction_batch(random.PRNGKey(0), main_cfg, 4)
    assert list(np.asarray(mask)) == [False, False, True, True]
    assert list(np.asarray(tok[0])[:12]) == [22, 22, 96, 35, 51, 87, 12, 70, 35, 16, 82, 64]
    assert not np.asarray(tpos[0]).any()


def test_induction_batch_structure_and_determinism(main_cfg):
    a = make_induction_batch(random.PRNGKey(3), main_cfg, 32)
    b = make_induction_batch(random.PRNGKey(3), main_cfg, 32)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    tok, mask, tpos = (np.asarray(x) for x in a)
    half = main_cfg.seq_len // 2
    for i in range(32):
        if mask[i]:
            np.testing.assert_array_equal(tok[i, :half], tok[i, half:])
            assert tpos[i, half:].all() and not tpos[i, :half].any()
        else:
            assert not tpos[i].any()


def test_gap_config_fields_are_not_used(main_cfg):
    # Documents audit finding P0-1: the repeat offset is fixed whatever the gap fields say.
    other = main_cfg.replace(induction_gap_min=5, induction_gap_max=6)
    a = make_induction_batch(random.PRNGKey(1), main_cfg, 16)
    b = make_induction_batch(random.PRNGKey(1), other, 16)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_clean_corrupt_pair(main_cfg):
    clean, corrupt, pos, tok = make_clean_corrupt_pair(main_cfg, random.PRNGKey(5))
    half = main_cfg.seq_len // 2
    c, x = np.asarray(clean[0]), np.asarray(corrupt[0])
    np.testing.assert_array_equal(c[:half], x[:half])
    assert pos == half + 1 and tok == c[1] == c[half + 1]
    # the corruption also changes the query token read at pos - 1 (with prob 1 - 1/V)
    assert c[pos - 1] == c[0]


def test_offset_probe_batch(main_cfg):
    rng = np.random.default_rng(0)
    tok, scored = make_offset_probe_batch(rng, main_cfg, 16, 8)
    np.testing.assert_array_equal(tok[:, :16], tok[:, 16:32])
    assert scored[:, 17:32].all() and scored.sum() == 8 * 15
    with pytest.raises(ValueError):
        make_offset_probe_batch(rng, main_cfg, 40, 1)
    _tok2, sc2 = make_corrupt_copy_batch(rng, main_cfg, 4)
    assert sc2[:, 33:].all() and not sc2[:, :33].any()
