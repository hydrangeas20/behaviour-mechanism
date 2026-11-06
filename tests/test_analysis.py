import json
import math

import numpy as np
import pandas as pd
import pytest
from jax import random

from behaviour_mechanism.artifacts import expected_raw_loss, quantisation, verify
from behaviour_mechanism.cli import main
from behaviour_mechanism.model import init_params
from behaviour_mechanism.probes import offset_accuracy, positional_copy_rate
from behaviour_mechanism.stability import (
    sustained_ceiling_step,
    threshold_sweep,
    variance_stable_step,
)


def test_variance_rule_accepts_floor_as_stable():
    # The flaw that led the preprint to reject this rule.
    assert variance_stable_step([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.5], 0.05) == 0


def test_sustained_ceiling_requires_every_seed_and_window():
    wide = pd.DataFrame(
        {0: [0.0, 0.9, 0.9, 0.9, 0.9], 1: [0.0, 0.9, 0.5, 0.9, 0.9]}, index=[0, 1, 2, 3, 4]
    )
    assert sustained_ceiling_step(wide, 0.8, window=3) is None
    assert sustained_ceiling_step(wide, 0.8, window=2) == 3
    long = (
        wide.stack()
        .rename("v")
        .reset_index()
        .rename(columns={"level_0": "step", "level_1": "seed"})
    )
    sw = threshold_sweep(long, {"T": "v"}, [0.4, 0.95])
    assert sw["T"].tolist()[0] == 1 and pd.isna(sw["T"].tolist()[1])


def test_quantisation_detects_k_over_8():
    q = quantisation(np.array([0.125, 0.25, 0.875, 1.0]))
    assert q["median_distance"] < 1e-9 and q["frac_within_0.01"] == 1.0


def test_ideal_losses_are_ordered():
    g = expected_raw_loss(100, 0.7, 64, gated=True)
    u = expected_raw_loss(100, 0.7, 64, gated=False)
    assert g < u < math.log(100)
    assert expected_raw_loss(100, 0.0, 64, gated=True) == pytest.approx(math.log(100), rel=0.01)


def test_historical_artifacts_are_consistent(repo):
    out = verify(repo)
    assert out["config_hash_matches_notebook_output"]
    assert out["merged_equals_join"] and out["variance_by_step_recomputes"]
    assert out["variance_rule"] == {"T_b": 0, "T_m": 2000}  # notebook cell 25 output
    sw = pd.DataFrame(out["sweep"]).set_index("theta")
    # Preprint section 4.1 claims; recomputed (T_b is 350 for theta 0.98-0.99, see audit)
    assert set(sw.loc[0.80:0.99, "T_m"]) == {2000, 2500}
    assert pd.isna(sw.loc[1.0, "T_m"])
    assert set(sw.loc[0.80:0.97, "T_b"]) == {300}
    assert out["exact_zero_patch"]["main_rows"] == 41


def test_probes_near_chance_at_init(main_cfg):
    p = init_params(main_cfg, random.PRNGKey(0))
    rng = np.random.default_rng(0)
    assert positional_copy_rate(p, main_cfg, 32, rng)["random_copy_rate"] < 0.05
    assert offset_accuracy(p, main_cfg, 16, 32, rng)["offset16_acc"] < 0.05


def test_cli_diagnose_smoke(tmp_path, tiny_cfg):
    cfg_path = tmp_path / "tiny.json"
    d = tiny_cfg.to_dict()
    d["seeds"] = list(d["seeds"])
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
                "2",
            ]
        )
        == 0
    )
    rows = [json.loads(x) for x in (out / "checkpoints.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 10]
    man = json.loads((out / "manifest.json").read_text())
    assert man["status"] == "complete" and man["config_hash"] == tiny_cfg.config_hash()
    with pytest.raises(FileExistsError):
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
            ]
        )


def test_git_state_reports_clean_tree_as_false(monkeypatch):
    from behaviour_mechanism import provenance

    answers = {"rev-parse": "abc123", "status": ""}
    monkeypatch.setattr(provenance, "_git", lambda *a: answers[a[0]])
    assert provenance.git_state()["dirty"] is False
    monkeypatch.setattr(provenance, "_git", lambda *a: None)
    assert provenance.git_state()["dirty"] is None
