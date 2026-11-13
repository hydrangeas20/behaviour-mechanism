import math

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from behaviour_mechanism.model import Cache, Patch, build_model, forward, init_params
from behaviour_mechanism.patching import (
    PairRecord,
    layer_pair_records,
    legacy_head_effects,
    legacy_layer_effects,
    summarise,
)
from behaviour_mechanism.training import induction_eval_score


@pytest.fixture(scope="module")
def tiny_params(tiny_cfg):
    return init_params(tiny_cfg, random.PRNGKey(0))


def test_functional_forward_matches_flax(tiny_cfg, tiny_params):
    tok = random.randint(random.PRNGKey(1), (3, tiny_cfg.seq_len), 0, tiny_cfg.vocab_size)
    ref = build_model(tiny_cfg).apply({"params": tiny_params}, tok)
    np.testing.assert_allclose(forward(tiny_params, tiny_cfg, tok), ref, atol=2e-5)


def test_identity_patches_change_nothing(tiny_cfg, tiny_params):
    tok = random.randint(random.PRNGKey(2), (1, tiny_cfg.seq_len), 0, tiny_cfg.vocab_size)
    cache = Cache()
    base = forward(tiny_params, tiny_cfg, tok, cache=cache)
    for l in range(tiny_cfg.n_layers):
        p = Patch(layer_attn_out={l: cache.layer_attn_out[l]})
        np.testing.assert_allclose(forward(tiny_params, tiny_cfg, tok, patch=p), base, atol=1e-6)
        p = Patch(head_out={(l, 0): cache.head_out[l][:, 0]})
        np.testing.assert_allclose(forward(tiny_params, tiny_cfg, tok, patch=p), base, atol=1e-6)


def test_all_heads_patch_equals_layer_patch(tiny_cfg, tiny_params):
    k1, k2 = random.split(random.PRNGKey(3))
    a = random.randint(k1, (1, tiny_cfg.seq_len), 0, tiny_cfg.vocab_size)
    b = random.randint(k2, (1, tiny_cfg.seq_len), 0, tiny_cfg.vocab_size)
    ca = Cache()
    forward(tiny_params, tiny_cfg, a, cache=ca)
    layer = forward(tiny_params, tiny_cfg, b, patch=Patch(layer_attn_out={1: ca.layer_attn_out[1]}))
    heads = forward(
        tiny_params,
        tiny_cfg,
        b,
        patch=Patch(head_out={(1, h): ca.head_out[1][:, h] for h in range(tiny_cfg.n_heads)}),
    )
    np.testing.assert_allclose(layer, heads, atol=1e-5)


def test_patch_only_affects_downstream(tiny_cfg, tiny_params):
    # Patching the last layer's attention leaves earlier residual streams untouched.
    tok = random.randint(random.PRNGKey(4), (1, tiny_cfg.seq_len), 0, tiny_cfg.vocab_size)
    c0, c1 = Cache(), Cache()
    forward(tiny_params, tiny_cfg, tok, cache=c0)
    z = jnp.zeros_like(c0.layer_attn_out[1])
    forward(tiny_params, tiny_cfg, tok, patch=Patch(layer_attn_out={1: z}), cache=c1)
    np.testing.assert_allclose(c0.resid_post[0], c1.resid_post[0])
    assert not np.allclose(c0.resid_post[1], c1.resid_post[1])


def _rec(defined, rec):
    return PairRecord(1.0 if defined else 0.0, 0.0, 0.0, tuple(rec), defined)


def test_legacy_aggregate_counts_skipped_pairs_as_zero():
    recs = [_rec(True, [1.0, 0.0])] * 3 + [_rec(False, [math.nan, math.nan])] * 5
    np.testing.assert_allclose(legacy_layer_effects(recs, 2), [3 / 8, 0.0])
    s = summarise(recs, 2)
    assert s["patch_effect_size"] == pytest.approx(0.375)
    assert s["defined_fraction"] == pytest.approx(3 / 8)
    assert s["conditional_L0"] == pytest.approx(1.0)


def test_all_skipped_is_exact_zero_and_conditional_is_nan():
    recs = [_rec(False, [math.nan] * 3)] * 8
    assert legacy_layer_effects(recs, 3).tolist() == [0.0, 0.0, 0.0]
    s = summarise(recs, 3)
    assert s["patch_effect_size"] == 0.0 and s["patch_best_layer"] == 0
    assert math.isnan(s["conditional_best"])


def test_clipping_and_head_normalisation():
    recs = [_rec(True, [9.0, -9.0, 1.0, 0.0]), _rec(False, [math.nan] * 4)]
    np.testing.assert_allclose(legacy_layer_effects(recs, 4), [2.5, -2.5, 0.5, 0.0])
    np.testing.assert_allclose(legacy_head_effects(recs, 2, 2), [[5.0, -5.0], [1.0, 0.0]])


def test_reproduces_historical_step0_values(main_cfg):
    """Step 0 needs no training, so CPU must match the GPU run up to float error.

    results/behaviour.csv seed 0 step 0: 0.008916884816753927
    results/mechanistic.csv seed 0 step 0: 0.4146179190609896 (best layer 0)
    """
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    params = init_params(main_cfg, init_rng)
    rng, eval_rng = random.split(rng)
    score = induction_eval_score(build_model(main_cfg), params, main_cfg, eval_rng)
    assert score == pytest.approx(0.008916884816753927, abs=1e-12)
    _, patch_rng, _ = random.split(random.PRNGKey(0 + 10_000), 3)
    s = summarise(layer_pair_records(params, main_cfg, patch_rng, 8), main_cfg.n_layers)
    assert s["patch_effect_size"] == pytest.approx(0.4146179190609896, abs=1e-4)
    assert s["patch_best_layer"] == 0
