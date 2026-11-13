"""Stage 2: E2 task, shortcut baselines, reference solutions, positive control."""

import json

import numpy as np
import pytest

from behaviour_mechanism import constructed_induction as ci
from behaviour_mechanism import e2 as E
from behaviour_mechanism.cli import main
from behaviour_mechanism.e2_run import eval_sets, evaluate_behaviour
from behaviour_mechanism.measurement import PairSet, legacy_statistic, measure, standard_sites

E2 = E.E2Config()


def test_generator_is_deterministic_and_versioned():
    a = E.make_batch(E.split_rng(E2, "train", 3), E2, 32)
    b = E.make_batch(E.split_rng(E2, "train", 3), E2, 32)
    np.testing.assert_array_equal(a.tokens, b.tokens)
    c = E.make_batch(E.split_rng(E2, "train", 4), E2, 32)
    assert not np.array_equal(a.tokens, c.tokens)
    assert E2.to_dict()["generator_version"] == "e2-v1"


def test_repeat_structure_and_scored_positions():
    b = E.make_batch(E.split_rng(E2, "val", 0), E2, 400)
    assert 0.6 < b.is_repeat.mean() < 0.8
    for i in range(400):
        s, l = int(b.start[i]), int(b.length[i])
        if not b.is_repeat[i]:
            assert (s, l) == (-1, 0) and not b.scored[i].any()
            continue
        assert l in E2.train_lengths and 0 <= s <= E2.seq_len - 2 * l
        np.testing.assert_array_equal(b.tokens[i, s : s + l], b.tokens[i, s + l : s + 2 * l])
        want = np.zeros(E2.seq_len, dtype=bool)
        want[s + l + 1 : s + 2 * l] = True  # first repeated token is not scored
        np.testing.assert_array_equal(b.scored[i], want)


def test_offsets_and_positions_vary():
    b = E.make_batch(E.split_rng(E2, "val", 0), E2, 2000, frac_repeat=1.0)
    assert set(b.length.tolist()) == set(E2.train_lengths)
    assert not set(b.length.tolist()) & set(E2.heldout_lengths)
    assert len(set((b.start + b.length).tolist())) > 30  # repeat start is spread out


def test_config_validation():
    with pytest.raises(ValueError):
        E.E2Config(heldout_lengths=(40,))
    with pytest.raises(ValueError):
        E.E2Config(extrapolation_lengths=(10,))
    with pytest.raises(ValueError):
        E.make_batch(np.random.default_rng(0), E2, 4, lengths=(30,), gap=8)


def test_splits_do_not_share_sequences_or_streams():
    tr = E.make_batch(E.split_rng(E2, "train", 0), E2, 500)
    va = E.make_batch(E.split_rng(E2, "val", 0), E2, 500)
    te = E.make_batch(E.split_rng(E2, "test", 0), E2, 500)
    h = [E.sequence_hashes(x.tokens) for x in (tr, va, te)]
    assert all(len(x) == 500 for x in h)
    assert not (h[0] & h[1]) and not (h[0] & h[2]) and not (h[1] & h[2])


@pytest.fixture(scope="module")
def baselines():
    batch = E.make_batch(E.split_rng(E2, "test", 0), E2, 600, frac_repeat=1.0)
    rows = E.baseline_table(E2, batch, E.fit_tables(E2, n_batches=40))
    return {r["label"]: r["accuracy"] for r in rows}


def test_shortcut_baselines_fail_and_reference_solves(baselines):
    acc = baselines
    for name in ("token_frequency", "position_only", "bigram", "previous_token"):
        assert acc[name] < 0.03, name
    assert acc["fixed_offset_historical"] < 0.03  # the original shortcut is at chance
    assert acc["fixed_offset_best"] < 0.2  # a constant offset only wins on its own length
    assert acc["reference_b"] > 0.98
    assert 0.75 < acc["reference_a"] < 0.95


def test_fixed_offset_predictor_alignment():
    tok = np.arange(12).reshape(1, 12)
    pred = E.predict_fixed_offset(tok, 3)
    assert pred[0, :2].tolist() == [-1, -1]
    np.testing.assert_array_equal(pred[0, 2:], tok[0, :9])  # target p gets token p - 3


def test_reference_solutions_on_a_hand_example():
    #        0  1  2  3  4  5  6  7
    tok = np.array([[5, 6, 7, 9, 5, 6, 7, 1]])
    a = E.predict_last_occurrence(tok)[0]
    assert a[4] == 6 and a[5] == 7 and a[6] == 9  # after 5 -> 6, after 6 -> 7, after 7 -> 9
    assert a[0] == -1
    # ambiguity: token 5 occurred twice with different successors
    tok2 = np.array([[5, 1, 2, 5, 3, 2, 5, 0]])
    assert E.predict_last_occurrence(tok2)[0][6] == 3  # most recent occurrence
    assert E.predict_longest_suffix(tok2)[0][6] == 3  # context "2 5" matches position 3


@pytest.mark.parametrize("corruption", E.E2_CORRUPTIONS)
@pytest.mark.parametrize("distinct", [False, True])
def test_e2_pairs_change_only_intended_positions(corruption, distinct):
    d = E.make_e2_pairs(np.random.default_rng(1), E2, 50, corruption, distinct=distinct)
    for i in range(50):
        c, x, q, l = d["clean"][i], d["corrupt"][i], int(d["read_pos"][i]), int(d["length"][i])
        assert c[q] == c[q - l] and d["target"][i] == c[q + 1] == c[q - l + 1]
        diff = np.flatnonzero(c != x)
        expect = {"earlier_occurrence": [q - l], "copy_source": [q - l + 1], "query_token": [q]}
        if corruption in expect:
            assert diff.tolist() == expect[corruption]
        else:
            assert diff.min() >= l  # nothing before the earliest possible repeat start
            assert x[q] != c[q] or not distinct
        if distinct:
            first = c[: q + 1]
            _, counts = np.unique(first, return_counts=True)
            assert counts.max() == 2  # only the repeat duplicates tokens


def test_unknown_corruption_rejected():
    with pytest.raises(ValueError):
        E.make_e2_pairs(np.random.default_rng(0), E2, 2, "nope")


# ----------------------------------------------------------- positive control (induction)


@pytest.fixture(scope="module")
def induction_model():
    return ci.build()


def test_constructed_induction_generalises_across_offsets(induction_model):
    r = evaluate_behaviour(induction_model, ci.CFG, E2, eval_sets(E2, "test", 64))
    # single-token matching: capped by duplicate tokens, like reference A
    for key in ("in_dist_acc", "heldout_acc", "extrapolation_acc", "gap4_acc", "gap8_acc"):
        assert 0.7 < r[key] < 0.95, key
    assert r["shuffled_acc"] < 0.15


@pytest.mark.parametrize("corruption", E.E2_CORRUPTIONS)
def test_corrected_metric_matches_analytic_predictions_on_induction_model(
    induction_model, corruption
):
    d = E.make_e2_pairs(np.random.default_rng([7, 0]), E2, 32, corruption, distinct=True)
    pairs = PairSet(d["clean"], d["corrupt"], d["target"], d["read_pos"], corruption)
    res = measure(induction_model, ci.CFG, pairs, standard_sites(ci.CFG))
    assert {r.site for r in res} == set(ci.SITES_2L)
    for r in res:
        assert r.n_defined == r.n_attempted == 32
        assert r.conditional_effect_mean == pytest.approx(
            ci.predicted_effect(corruption, r.site), abs=0.02
        ), (r.site, r.direction)
    leg = legacy_statistic(res, 2)
    assert leg["per_layer"][1] == pytest.approx(1.0, abs=0.02)
    want_l0 = 1.0 if corruption == "earlier_occurrence" else 0.0
    assert leg["per_layer"][0] == pytest.approx(want_l0, abs=0.02)


def test_cli_e2_train_smoke(tmp_path, tiny_cfg):
    d = tiny_cfg.to_dict()
    d.update(seeds=list(d["seeds"]), vocab_size=100, max_seq_len=64, seq_len=64)
    cfg_path = tmp_path / "tiny.json"
    cfg_path.write_text(json.dumps(d))
    out = tmp_path / "run"
    assert (
        main(
            [
                "e2-train",
                "--config",
                str(cfg_path),
                "--seed",
                "0",
                "--steps",
                "4",
                "--ckpt-every",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 2, 4]
    man = json.loads((out / "manifest.json").read_text())
    assert man["status"] == "complete" and man["generator_version"] == "e2-v1"
    assert set(man["checkpoint_sha256"]) == {"step_0.npz", "step_2.npz", "step_4.npz"}
    with pytest.raises(FileExistsError):
        main(["e2-train", "--config", str(cfg_path), "--seed", "0", "--out", str(out)])


def test_positional_mixture_shortcut_is_bounded_and_fails_on_unseen_lengths():
    tables = E.fit_tables(E2, n_batches=40)
    ind = E.make_batch(E.split_rng(E2, "test", 0), E2, 600, frac_repeat=1.0)
    held = E.make_batch(E.split_rng(E2, "test", 1), E2, 600, E2.heldout_lengths, 1.0)
    a_in = E.accuracy(E.predict_offset_by_position(ind.tokens, tables["offset_table"]), ind)
    a_held = E.accuracy(E.predict_offset_by_position(held.tokens, tables["offset_table"]), held)
    assert 0.08 < a_in < 0.25  # better than chance: a shortcut that E2 does not remove
    assert a_held < 0.03  # but it cannot transfer to chunk lengths it never saw


# ------------------------------------------------- recorded Stage 2 results (saved artifacts)


def _beh(repo, run):
    import pandas as pd

    rows = [json.loads(x) for x in (repo / f"results/stage2/{run}/behaviour.jsonl").open()]
    return pd.DataFrame(rows).set_index("step")


def test_recorded_stage2_results(repo):
    import pandas as pd

    cal = json.loads((repo / "results/stage2/calibration/summary.json").read_text())
    assert cal["distinct"]["passed"] == cal["distinct"]["cells"] == 80
    base = pd.read_csv(repo / "results/stage2/baselines/baselines.csv")
    base = base[base.test_set == "in_dist"].set_index("label").accuracy
    assert base["fixed_offset_historical"] < 0.02 and base["reference_b"] > 0.98
    leak = json.loads((repo / "results/stage2/baselines/leakage.json").read_text())
    assert leak["eval_sequences_also_in_train"] == leak["eval_chunks_also_in_train"] == 0
    assert leak["heldout_lengths_in_train"] == []

    learned, plateau = _beh(repo, "e2_lr1e-3_s0"), _beh(repo, "e2_main_lr3e-4_s0")
    # the run that learned the task generalises to chunk lengths it never saw
    assert learned.in_dist_acc.iloc[-1] > 0.98 and learned.heldout_acc.iloc[-1] > 0.98
    # the primary run stayed near the positional-mixture shortcut level
    assert plateau.in_dist_acc.iloc[-1] < 0.3 and plateau.heldout_acc.iloc[-1] < 0.1
    assert plateau.in_dist_acc.iloc[-1] > 3 * plateau.heldout_acc.iloc[-1]

    # the legacy statistic is near ceiling in a model that never learned the task
    leg = pd.read_csv(repo / "results/stage2/e2_main_lr3e-4_s0/legacy_statistic.csv")
    cs = leg[(leg.corruption == "copy_source") & (leg.step >= 750)]
    assert cs.legacy_patch_effect_size_v1.min() > 0.9

    # no result row silently zero-fills: undefined + defined = attempted everywhere
    for run in ("e2_lr1e-3_s0", "e2_main_lr3e-4_s0"):
        m = pd.read_csv(repo / f"results/stage2/{run}/mechanism.csv")
        assert (m.n_defined + m.n_undefined == m.n_attempted).all()
        assert (m.metric_version == "bm-patch-2").all()
        assert m[m.n_defined == 0].conditional_effect_mean.isna().all()


def test_settled_step_helper():
    from behaviour_mechanism.e2_figures import first_sustained, settled_step

    assert settled_step([0, 1, 2, 3], [0.0, 0.9, 0.5, 0.52]) == 2
    assert settled_step([0, 1], [0.5, 0.5]) == 0
    assert first_sustained([0, 1, 2, 3], [0.1, 0.9, 0.2, 0.95], 0.8) is None
    assert first_sustained([0, 1, 2, 3], [0.1, 0.9, 0.85, 0.95], 0.8) == 1
