"""Re-train one seed with the historical code and save selected parameter checkpoints.

Checkpoints are written to artifacts/checkpoints/<tag>/step_<n>.npz (gitignored: large,
regenerable). Used as inputs to the E3 controls.

Usage: python scripts/train_and_save_checkpoints.py --seed 0 --stop-at 3000 \
           --steps 300,500,1000,1200,1300,1500,2000,3000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from behaviour_mechanism.checkpoints import save_params
from behaviour_mechanism.config import load_config
from behaviour_mechanism.provenance import build_manifest, write_manifest
from behaviour_mechanism.training import run_training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stop-at", type=int, required=True)
    ap.add_argument("--steps", required=True, help="comma-separated checkpoint steps to save")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    keep = {int(s) for s in a.steps.split(",")}
    missing = keep - set(cfg.checkpoint_steps())
    if missing:
        raise SystemExit(f"not historical checkpoint steps: {sorted(missing)}")
    out = Path(a.out or f"artifacts/checkpoints/main_s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    man = build_manifest(
        "checkpoints",
        config=cfg.to_dict(),
        config_hash=cfg.config_hash(),
        seed=a.seed,
        stop_at=a.stop_at,
        saved_steps=sorted(keep),
    )

    def on_ckpt(step: int, params: object) -> dict[str, object]:
        if step in keep:
            save_params(out / f"step_{step}.npz", params)
        return {}

    res = run_training(
        cfg,
        a.seed,
        on_checkpoint=on_ckpt,
        stop_at=a.stop_at,
        log=lambda m: print(m, flush=True),
        log_every=500,
    )
    man["records"] = res.records
    man["seconds"] = round(res.seconds, 1)
    write_manifest(out / "manifest.json", man)


if __name__ == "__main__":
    main()
