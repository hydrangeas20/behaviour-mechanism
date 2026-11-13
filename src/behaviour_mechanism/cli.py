"""Command-line entry point: ``bm <command>``.

Commands
--------
verify-artifacts   check and analyse the historical CSVs (no training; seconds)
figures            regenerate figures from the historical CSVs into figures/derived/
diagnose           CPU/GPU re-run of one seed with extra measurements (see diagnose.py)
diagnostics-report summarise diagnostic runs into results/diagnostics/summary.csv + figures
env                print the provenance record for this environment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .provenance import build_manifest


def parse_steps(spec: str) -> tuple[int, ...]:
    """``"1550:2500:50"`` (inclusive range) or ``"1550,1600"``; empty -> ()."""
    if not spec:
        return ()
    if ":" in spec:
        lo, hi, step = (int(x) for x in spec.split(":"))
        return tuple(range(lo, hi + 1, step))
    return tuple(int(x) for x in spec.split(","))


def _repo(arg: str | None) -> Path:
    return Path(arg) if arg else Path.cwd()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify-artifacts")
    p.add_argument("--repo", default=None)
    p.add_argument("--strict", action="store_true", help="exit 1 if any consistency check fails")

    p = sub.add_parser("figures")
    p.add_argument("--repo", default=None)

    p = sub.add_parser("diagnose")
    p.add_argument("--config", default="configs/config.json")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stop-at", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set frac_induction_sequences=0",
    )
    p.add_argument("--label", default="")
    p.add_argument("--n-large", type=int, default=64)
    p.add_argument("--n-probe-seq", type=int, default=128)
    p.add_argument("--n-head-pairs", type=int, default=16)
    p.add_argument(
        "--fixed-pairs",
        type=int,
        default=0,
        help="E1: reuse one fixed set of N probe pairs at every checkpoint",
    )
    p.add_argument(
        "--control-pairs",
        type=int,
        default=0,
        help="E3: run corruption x intervention controls with N pairs per checkpoint",
    )
    p.add_argument(
        "--extra-steps",
        default="",
        help="E1: extra checkpoint steps, e.g. 1550:2500:50 or 1550,1600",
    )

    p = sub.add_parser("constructed", help="E4: probe the hand-set copiers")
    p.add_argument("--out", default="results/controls/e4_constructed")
    p.add_argument("--n-pairs", type=int, default=64)

    p = sub.add_parser("controls", help="E3: corruption x intervention controls on checkpoints")
    p.add_argument("--checkpoints", required=True, help="directory of step_<n>.npz files")
    p.add_argument("--config", default="configs/config.json")
    p.add_argument("--out", required=True)
    p.add_argument("--n-pairs", type=int, default=128)
    p.add_argument("--pair-seed", type=int, default=0)

    p = sub.add_parser("stage1-controls", help="Stage 1 positive controls (bm-patch-2)")
    p.add_argument("--out", default="results/stage1/positive_controls")
    p.add_argument("--n-pairs", type=int, default=64)
    p.add_argument("--pair-seed", type=int, default=0)

    p = sub.add_parser("controls-report")
    p.add_argument("--repo", default=None)

    p = sub.add_parser("diagnostics-report")
    p.add_argument("--repo", default=None)

    sub.add_parser("env")

    a = ap.parse_args(argv)

    if a.cmd == "verify-artifacts":
        from .artifacts import write_derived

        out = write_derived(_repo(a.repo))
        checks = {
            k: out[k]
            for k in (
                "config_hash_matches_notebook_output",
                "merged_equals_join",
                "variance_by_step_recomputes",
            )
        }
        print(json.dumps({"checks": checks, "variance_rule": out["variance_rule"]}, indent=2))
        print("wrote results/derived/artifact_verification.json and sustained_ceiling_sweep.csv")
        return 0 if (all(checks.values()) or not a.strict) else 1

    if a.cmd == "figures":
        from .figures import historical_figures

        for path in historical_figures(_repo(a.repo)):
            print("wrote", path)
        return 0

    if a.cmd == "diagnose":
        from .config import load_config
        from .diagnose import run_diagnostic

        cfg = load_config(a.config)
        overrides: dict[str, object] = {}
        for kv in a.set:
            key, _, val = kv.partition("=")
            cur = getattr(cfg, key)
            overrides[key] = (
                type(cur)(json.loads(val)) if not isinstance(cur, tuple) else tuple(json.loads(val))
            )
        cfg = cfg.replace(**overrides)
        path = run_diagnostic(
            cfg,
            a.seed,
            Path(a.out),
            a.stop_at,
            a.n_large,
            a.n_probe_seq,
            a.n_head_pairs,
            a.label,
            a.fixed_pairs,
            a.control_pairs,
            parse_steps(a.extra_steps),
        )
        print("wrote", path)
        return 0

    if a.cmd == "constructed":
        from .experiments import run_constructed

        print("wrote", run_constructed(Path(a.out), a.n_pairs))
        return 0

    if a.cmd == "controls":
        from .config import load_config
        from .experiments import run_checkpoint_controls

        path = run_checkpoint_controls(
            load_config(a.config), Path(a.checkpoints), Path(a.out), a.n_pairs, a.pair_seed
        )
        print("wrote", path)
        return 0

    if a.cmd == "stage1-controls":
        from .experiments import run_stage1_positive_controls

        s1_out = run_stage1_positive_controls(Path(a.out), a.n_pairs, a.pair_seed)
        summary = json.loads((s1_out / "summary.json").read_text())
        print(f"{summary['passed']}/{summary['cells']} cells match predictions; wrote {s1_out}")
        return 0 if summary["failed"] == 0 else 1

    if a.cmd == "controls-report":
        from .figures import controls_report

        for path in controls_report(_repo(a.repo)):
            print("wrote", path)
        return 0

    if a.cmd == "diagnostics-report":
        from .figures import diagnostics_report

        for path in diagnostics_report(_repo(a.repo)):
            print("wrote", path)
        return 0

    if a.cmd == "env":
        print(json.dumps(build_manifest("environment"), indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
