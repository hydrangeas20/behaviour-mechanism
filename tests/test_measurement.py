"""Stage 1: corrected measurement (bm-patch-2)."""

import json
import math

import numpy as np
import pytest
from jax import random

from behaviour_mechanism import constructed
from behaviour_mechanism.controls import make_control_pairs
from behaviour_mechanism.measurement import (
    LEGACY_NAME,
    METRIC_VERSION,
    PairSet,
    Site,
    from_json,
    legacy_statistic,
    measure,
    standard_sites,
    to_json,
)
from behaviour_mechanism.model import init_params
from behaviour_mechanism.patching import PairRecord, _layer_pair, legacy_layer_effects, summarise
from behaviour_mechanism.positive_controls import (
    SITES,
    predicted_defined_fraction,
    predicted_effect,
    run,
)

CFG = constructed.CFG


@pytest.fixture(scope="module")
def models():
    return {m: constructed.build(m) for m in constructed.MODELS}


def _pairs(corruption, n=16, seed=0, cfg=CFG):
    clean, corrupt, target = make_control_pairs(np.random.default_rng(seed), cfg, n, corruption)
    return PairSet(clean, corrupt, target, np.full(n, cfg.seq_len // 2), corruption)


def _get(results, site, direction):
    return next(r for r in results if r.site == site and r.direction == direction)


# ------------------------------------------------------------------ denominators / undefined


def test_undefined_pairs_are_reported_not_zero_filled(models):
    res = measure(models["ungated"], CFG, _pairs("query_token"), [Site("attn_out", 0)])
    r = _get(res, "attn_out.L0", "denoise")
    assert (r.n_attempted, r.n_defined, r.n_undefined) == (16, 0, 16)
    assert r.defined_fraction == 0.0
    assert r.conditional_effect_mean is None and r.conditional_effect_median is None
    assert r.per_pair_effect == [None] * 16


def test_fully_defined_pairs(models):
    res = measure(models["attn_gate"], CFG, _pairs("copy_source"), [Site("attn_out", 0)])
    r = _get(res, "attn_out.L0", "denoise")
    assert r.n_defined == r.n_attempted == 16 and r.n_undefined == 0
    assert r.conditional_effect_mean == pytest.approx(1.0, abs=1e-3)


def test_conditional_mean_uses_defined_pairs_only():
    """Mixed set: half the pairs undefined (identical clean/corrupt), half fully restored."""
    p = constructed.build("attn_gate")
    a = _pairs("copy_source", n=8)
    clean = a.clean.copy()
    corrupt = a.corrupt.copy()
    corrupt[4:] = clean[4:]  # pairs 4..7: no corruption -> gap 0 -> undefined
    pairs = PairSet(clean, corrupt, a.target, a.read_pos, "copy_source_half")
    r = _get(measure(p, CFG, pairs, [Site("attn_out", 0)]), "attn_out.L0", "denoise")
    assert (r.n_defined, r.n_undefined) == (4, 4)
    assert r.conditional_effect_mean == pytest.approx(1.0, abs=1e-3)  # not 0.5
    leg = legacy_statistic(
        measure(p, CFG, pairs, [Site("attn_out", 0), Site("attn_out", 1)]), CFG.n_layers
    )
    assert leg["name"] == LEGACY_NAME
    assert leg["value"] == pytest.approx(0.5, abs=1e-3)  # legacy zero-fills -> 4/8


def test_legacy_statistic_equals_historical_code(main_cfg):
    params = init_params(main_cfg, random.PRNGKey(0))
    pairs = _pairs("random_second_half", n=8, seed=4, cfg=main_cfg)
    res = measure(
        params, main_cfg, pairs, [Site("attn_out", l) for l in range(4)], directions=("denoise",)
    )
    recs = []
    for i in range(8):
        lc, lx, lp = _layer_pair(
            main_cfg,
            params,
            pairs.clean[i : i + 1],
            pairs.corrupt[i : i + 1],
            np.asarray(32),
            np.asarray(pairs.target[i]),
        )
        gap = float(lc) - float(lx)
        ok = abs(gap) >= 0.05
        recs.append(
            PairRecord(
                gap,
                float(lc),
                float(lx),
                tuple((np.asarray(lp) - float(lx)) / gap) if ok else (math.nan,) * 4,
                ok,
            )
        )
    want = summarise(recs, 4)["patch_effect_size"]
    assert legacy_statistic(res, 4)["value"] == pytest.approx(want, abs=1e-4)
    np.testing.assert_allclose(
        legacy_statistic(res, 4)["per_layer"], legacy_layer_effects(recs, 4), atol=1e-4
    )


# ------------------------------------------------------------- zero effect / direction


def test_zero_effect_intervention(models):
    """Layer-1 attention is identically zero in every constructed model."""
    res = measure(models["attn_gate"], CFG, _pairs("query_token"), [Site("attn_out", 1)])
    for d in ("denoise", "noise"):
        r = _get(res, "attn_out.L1", d)
        assert r.n_defined == 16
        assert r.conditional_effect_mean == pytest.approx(0.0, abs=1e-6)
        assert r.activation_delta_mean == pytest.approx(0.0, abs=1e-6)


def test_intervention_direction_is_respected(models):
    """MLP-gated copier, query-token corruption: the attention output carries no
    clean/corrupt difference (delta 0), so neither direction moves it; MLP-0 carries the
    match flag, so denoising restores and noising destroys."""
    res = measure(
        models["mlp_gate"], CFG, _pairs("query_token"), [Site("attn_out", 0), Site("mlp_out", 0)]
    )
    assert _get(res, "attn_out.L0", "denoise").activation_delta_mean == pytest.approx(0, abs=1e-6)
    assert _get(res, "attn_out.L0", "denoise").conditional_effect_mean == pytest.approx(0, abs=1e-3)
    assert _get(res, "attn_out.L0", "noise").conditional_effect_mean == pytest.approx(0, abs=1e-3)
    assert _get(res, "mlp_out.L0", "denoise").conditional_effect_mean == pytest.approx(1, abs=1e-3)
    assert _get(res, "mlp_out.L0", "noise").conditional_effect_mean == pytest.approx(1, abs=1e-3)
    assert _get(res, "mlp_out.L0", "denoise").activation_delta_mean > 0.5


def test_denoise_and_noise_formulas(models):
    """Swapping clean and corrupt flips the gap sign; per-pair effects stay in the same
    units (fraction of the gap)."""
    p = models["attn_gate"]
    a = _pairs("copy_source", n=4)
    swapped = PairSet(a.corrupt, a.clean, a.clean[:, 1], a.read_pos, "swapped")
    r1 = _get(measure(p, CFG, a, [Site("head_out", 0, 1)]), "head_out.L0H1", "denoise")
    r2 = _get(measure(p, CFG, swapped, [Site("head_out", 0, 1)]), "head_out.L0H1", "denoise")
    assert all(g > 0 for g in r1.per_pair_gap) and all(g < 0 for g in r2.per_pair_gap)
    assert r1.conditional_effect_mean == pytest.approx(1, abs=1e-3)
    assert r2.conditional_effect_mean == pytest.approx(1, abs=1e-3)


# --------------------------------------------------------------------- positive controls


@pytest.mark.parametrize("model", ["attn_gate", "mlp_gate", "ungated"])
def test_positive_controls_match_analytic_predictions(models, model):
    for corr in ("query_token", "earlier_occurrence", "copy_source"):
        res = measure(models[model], CFG, _pairs(corr), list(SITES))
        for r in res:
            assert r.defined_fraction == pytest.approx(
                predicted_defined_fraction(model, corr), abs=0.05
            ), (model, corr, r.site)
            want = predicted_effect(model, corr, r.site)
            if want is None:
                assert r.conditional_effect_mean is None
            else:
                assert r.conditional_effect_mean == pytest.approx(want, abs=0.02), (
                    model,
                    corr,
                    r.site,
                    r.direction,
                )


def test_legacy_metric_misses_the_mlp_gate_but_corrected_sites_detect_it(models):
    res = measure(models["mlp_gate"], CFG, _pairs("query_token"), standard_sites(CFG))
    assert legacy_statistic(res, CFG.n_layers)["value"] == pytest.approx(0, abs=1e-3)
    best = max(r.conditional_effect_mean for r in res if r.direction == "denoise")
    assert best == pytest.approx(1, abs=1e-3)


def test_full_positive_control_grid_passes():
    rows, legacy = run(n_pairs=16)
    assert len(rows) == 3 * 4 * len(SITES) * 2
    assert all(r["pass"] for r in rows), [r for r in rows if not r["pass"]][:3]
    assert len(legacy) == 12


# ------------------------------------------------------------ serialisation / reproducibility


def test_serialisation_round_trip_is_strict_json(models):
    res = measure(models["ungated"], CFG, _pairs("query_token", n=4), [Site("mlp_out", 1)])
    text = to_json(res)
    assert "NaN" not in text and "Infinity" not in text
    back = from_json(text)
    assert back[0].to_dict() == res[0].to_dict()
    d = json.loads(text)[0]
    assert d["metric_version"] == METRIC_VERSION and d["conditional_effect_mean"] is None
    bad = dict(d, metric_version="bm-patch-1")
    with pytest.raises(ValueError, match="version"):
        from_json(json.dumps([bad]))


def test_reproducible_for_same_inputs(models):
    a = measure(models["attn_gate"], CFG, _pairs("earlier_occurrence", seed=7), list(SITES))
    b = measure(models["attn_gate"], CFG, _pairs("earlier_occurrence", seed=7), list(SITES))
    assert [x.to_dict() for x in a] == [y.to_dict() for y in b]


# -------------------------------------------------------------------- failure handling


def test_failure_handling(models):
    p = models["attn_gate"]
    good = _pairs("copy_source", n=4)
    with pytest.raises(ValueError, match="layer"):
        measure(p, CFG, good, [Site("attn_out", 5)])
    with pytest.raises(ValueError, match="head"):
        Site("head_out", 0)
    with pytest.raises(ValueError, match="component"):
        Site("embedding", 0)
    with pytest.raises(ValueError, match="read_pos"):
        measure(
            p,
            CFG,
            PairSet(good.clean, good.corrupt, good.target, good.read_pos + 100, "x"),
            [Site("attn_out", 0)],
        )
    with pytest.raises(ValueError, match="vocabulary"):
        measure(
            p,
            CFG,
            PairSet(good.clean, good.corrupt, good.target + 1000, good.read_pos, "x"),
            [Site("attn_out", 0)],
        )
    with pytest.raises(ValueError, match="shape"):
        measure(
            p,
            CFG,
            PairSet(good.clean, good.corrupt[:, :-1], good.target, good.read_pos, "x"),
            [Site("attn_out", 0)],
        )
    with pytest.raises(ValueError, match="empty"):
        measure(
            p,
            CFG,
            PairSet(good.clean[:0], good.corrupt[:0], good.target[:0], good.read_pos[:0], "x"),
            [Site("attn_out", 0)],
        )
    with pytest.raises(ValueError, match="direction"):
        measure(p, CFG, good, [Site("attn_out", 0)], directions=("sideways",))
    with pytest.raises(ValueError, match="positive"):
        measure(p, CFG, good, [Site("attn_out", 0)], gap_threshold=0)
    with pytest.raises(ValueError, match="no sites"):
        measure(p, CFG, good, [])


def test_resid_patch_is_confined_to_each_pairs_read_position(main_cfg):
    """``resid_post`` replaces the residual only where ``read_pos`` says, per pair."""
    import jax.numpy as jnp

    from behaviour_mechanism.model import Cache, Patch, forward

    params = init_params(main_cfg, random.PRNGKey(1))
    a, b = (random.randint(random.PRNGKey(k), (2, 64), 0, 100) for k in (2, 3))
    ca = Cache()
    forward(params, main_cfg, a, cache=ca)
    mask = jnp.zeros((2, 64), dtype=bool).at[0, 40].set(True).at[1, 50].set(True)
    out = np.asarray(
        forward(params, main_cfg, b, patch=Patch(resid_post_masked={3: (mask, ca.resid_post[3])}))
    )
    base_b, base_a = (np.asarray(forward(params, main_cfg, x)) for x in (b, a))
    np.testing.assert_allclose(out[0, 40], base_a[0, 40], atol=1e-4)  # patched (last layer)
    np.testing.assert_allclose(out[1, 50], base_a[1, 50], atol=1e-4)
    np.testing.assert_allclose(out[0, 50], base_b[0, 50], atol=1e-5)  # untouched elsewhere
    np.testing.assert_allclose(out[1, 40], base_b[1, 40], atol=1e-5)
