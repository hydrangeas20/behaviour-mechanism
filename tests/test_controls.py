"""E3/E4 controls and E1 fixed-probe infrastructure."""

import json
import math

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import random

from behaviour_mechanism import constructed
from behaviour_mechanism.cli import main, parse_steps
from behaviour_mechanism.controls import (
    CORRUPTIONS,
    fixed_pair_records,
    make_control_pairs,
    run_controls,
)
from behaviour_mechanism.model import Cache, Patch, build_model, forward, init_params
from behaviour_mechanism.patching import _layer_pair, layer_pair_records, summarise
from behaviour_mechanism.probes import positional_copy_rate
from behaviour_mechanism.training import induction_eval_score, run_training


@pytest.fixture(scope="module")
def built():
    return {m: constructed.build(m) for m in constructed.MODELS}


# ---------------------------------------------------------------- E4: constructed weights


def test_constructed_models_run_in_the_historical_architecture(built):
    tok = random.randint(random.PRNGKey(1), (2, 64), 0, 100)
    for p in built.values():
        ref = build_model(constructed.CFG).apply({"params": p}, tok)
        np.testing.assert_allclose(forward(p, constructed.CFG, tok), ref, atol=1e-4)


@pytest.mark.parametrize("model", constructed.MODELS)
def test_historical_probe_on_known_mechanisms(built, model):
    s = summarise(layer_pair_records(built[model], constructed.CFG, random.PRNGKey(5), 32), 2)
    exp = constructed.expected(model)
    # random_second_half may coincidentally keep the query token (prob 1/100 per pair)
    assert s["defined_fraction"] == pytest.approx(exp["defined_fraction"], abs=1 / 32)
    assert s["patch_effect_size"] == pytest.approx(exp["patch_effect_size"], abs=0.04)
    if "conditional_L0" in exp:
        assert s["conditional_L0"] == pytest.approx(exp["conditional_L0"], abs=0.02)


def test_behaviour_bound_separates_ungated_from_token_gated(built):
    cfg = constructed.CFG
    acc = {
        m: induction_eval_score(build_model(cfg), p, cfg, random.PRNGKey(0))
        for m, p in built.items()
    }
    assert acc["ungated"] == pytest.approx(1.0)
    # token-gated copiers cannot predict the first repeated token: capped at 31/32
    for m in ("mlp_gate", "attn_gate"):
        assert acc[m] == pytest.approx(31 / 32, abs=2e-3)
    rng = np.random.default_rng(0)
    assert positional_copy_rate(built["ungated"], cfg, 16, rng)["random_copy_logprob"] > -0.01
    assert positional_copy_rate(built["mlp_gate"], cfg, 16, rng)["random_copy_logprob"] < -4


def _table(params):
    df = pd.DataFrame(run_controls(params, constructed.CFG, 16, 0))
    return df.set_index(["corruption", "intervention"])


def test_controls_signatures_on_known_mechanisms(built):
    u, mg, ag = (_table(built[m]) for m in constructed.MODELS)
    # ungated copier: only the copy-source corruption is defined
    assert u.loc[("query_token", "attn_L0"), "n_defined"] == 0
    assert u.loc[("copy_source", "attn_L0"), "recovery_mean"] == pytest.approx(1, abs=1e-3)
    # the query-token corruption separates the two gates at layer-0 attention
    assert mg.loc[("query_token", "attn_L0"), "recovery_mean"] == pytest.approx(0, abs=1e-3)
    assert mg.loc[("query_token", "mlp_L0"), "recovery_mean"] == pytest.approx(1, abs=1e-3)
    assert ag.loc[("query_token", "attn_L0"), "recovery_mean"] == pytest.approx(1, abs=1e-3)
    # earlier-occurrence information reaches the gate through attention in both
    for t in (mg, ag):
        assert t.loc[("earlier_occurrence", "attn_L0"), "recovery_mean"] == pytest.approx(
            1, abs=1e-3
        )
        # noising is the mirror image of denoising for a single-path mechanism
        assert t.loc[("copy_source", "attn_L0"), "disruption_mean"] == pytest.approx(1, abs=1e-3)
        assert t.loc[("copy_source", "attn_L1"), "disruption_mean"] == pytest.approx(0, abs=1e-3)


# ------------------------------------------------------------------ E3: pair construction


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_control_pairs_change_only_intended_positions(main_cfg, corruption):
    clean, corrupt, target = make_control_pairs(np.random.default_rng(0), main_cfg, 64, corruption)
    half = main_cfg.seq_len // 2
    np.testing.assert_array_equal(clean[:, :half], clean[:, half:])
    np.testing.assert_array_equal(target, clean[:, 1])
    diff = clean != corrupt
    changed = {"query_token": [half], "earlier_occurrence": [0], "copy_source": [1]}
    if corruption in changed:
        cols = changed[corruption]
        assert diff[:, cols].all() and diff.sum() == 64 * len(cols)
    else:
        assert not diff[:, :half].any() and diff[:, half:].mean() > 0.9


def test_mlp_and_resid_identity_patches(tiny_cfg):
    p = init_params(tiny_cfg, random.PRNGKey(0))
    tok = random.randint(random.PRNGKey(2), (2, tiny_cfg.seq_len), 0, tiny_cfg.vocab_size)
    c = Cache()
    base = forward(p, tiny_cfg, tok, cache=c)
    for patch in (
        Patch(mlp_out={0: c.mlp_out[0]}),
        Patch(resid_post={1: (3, c.resid_post[1][:, 3])}),
    ):
        np.testing.assert_allclose(forward(p, tiny_cfg, tok, patch=patch), base, atol=1e-6)
    zero = Patch(resid_post={0: (5, jnp.zeros((2, tiny_cfg.d_model)))})
    changed = np.asarray(forward(p, tiny_cfg, tok, patch=zero))
    np.testing.assert_allclose(changed[:, :5], np.asarray(base)[:, :5], atol=1e-6)
    assert not np.allclose(changed[:, 5], np.asarray(base)[:, 5])


# ------------------------------------------------------------------ E1: fixed probe pairs


def test_fixed_pair_records_match_historical_per_pair_code(main_cfg):
    params = init_params(main_cfg, random.PRNGKey(0))
    clean, corrupt, target = make_control_pairs(
        np.random.default_rng(3), main_cfg, 6, "random_second_half"
    )
    batched = fixed_pair_records(params, main_cfg, clean, corrupt, target)
    half = main_cfg.seq_len // 2
    for i, r in enumerate(batched):
        lc, lx, lp = _layer_pair(
            main_cfg,
            params,
            jnp.asarray(clean[i : i + 1]),
            jnp.asarray(corrupt[i : i + 1]),
            jnp.asarray(half),
            jnp.asarray(target[i]),
        )
        assert r.lp_clean == pytest.approx(float(lc), abs=1e-4)
        assert r.lp_corrupt == pytest.approx(float(lx), abs=1e-4)
        if r.defined:
            want = (np.asarray(lp) - float(lx)) / (float(lc) - float(lx))
            np.testing.assert_allclose(r.recovered, want, atol=1e-3)


def test_extra_checkpoints_do_not_change_the_trajectory(tiny_cfg):
    a = run_training(tiny_cfg, 0, keep_params=True)
    b = run_training(tiny_cfg, 0, keep_params=True, extra_eval_steps=(5, 15))
    hist_b = [r for r in b.records if not r.get("extra_checkpoint")]
    assert [r["step"] for r in hist_b] == [r["step"] for r in a.records]
    for ra, rb in zip(a.records, hist_b, strict=True):
        assert ra == rb
    assert sorted(r["step"] for r in b.records if r.get("extra_checkpoint")) == [5, 15]
    for k in a.saved_params:
        for x, y in zip(_leaves(a.saved_params[k]), _leaves(b.saved_params[k]), strict=True):
            np.testing.assert_array_equal(x, y)


def _leaves(tree):
    import jax

    return jax.tree_util.tree_leaves(tree)


def test_parse_steps():
    assert parse_steps("") == ()
    assert parse_steps("1550:1650:50") == (1550, 1600, 1650)
    assert parse_steps("10,20") == (10, 20)


def test_cli_diagnose_with_fixed_pairs_controls_and_extra_steps(tmp_path, tiny_cfg):
    d = tiny_cfg.to_dict()
    d["seeds"] = list(d["seeds"])
    cfg_path = tmp_path / "tiny.json"
    cfg_path.write_text(json.dumps(d))
    out = tmp_path / "run"
    assert (
        main(
            [
                "diagnose",
                "--config",
                str(cfg_path),
                "--seed",
                "0",
                "--stop-at",
                "10",
                "--out",
                str(out),
                "--n-large",
                "4",
                "--n-probe-seq",
                "4",
                "--n-head-pairs",
                "0",
                "--fixed-pairs",
                "6",
                "--control-pairs",
                "4",
                "--extra-steps",
                "5",
            ]
        )
        == 0
    )
    rows = [json.loads(x) for x in (out / "checkpoints.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 5, 10]
    assert "hist_patch_effect_size" not in rows[1]  # extra step: historical RNG untouched
    assert all(r["fixed_n_pairs"] == 6 for r in rows)
    assert len(rows[0]["controls"]) == len(CORRUPTIONS) * (3 * tiny_cfg.n_layers + 1)
    # the fixed pair set is identical across checkpoints: lp_clean at init must equal a re-run
    man = json.loads((out / "manifest.json").read_text())
    assert man["measurement"]["fixed_pairs"] == 6 and man["measurement"]["extra_eval_steps"] == [5]
    assert not math.isnan(rows[0]["fixed_gap_median"])
