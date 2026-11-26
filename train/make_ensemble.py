"""Bundle several trained runs into one ensemble weights dir (rollouts are averaged at inference).

Usage: python -m train.make_ensemble --name ens3 runs/v1 runs/v1s1 runs/adv1
All members must share the same climatology/stats/static (same --val-start); this is checked.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    first = Path(a.runs[0])
    clim0 = np.load(first / "climatology.npy")
    members = []
    for k, r in enumerate(map(Path, a.runs)):
        cfg = json.loads((r / "config.json").read_text())
        assert np.array_equal(np.load(r / "climatology.npy"), clim0), f"{r}: different climatology"
        shutil.copy(r / "model.pt", out / f"model_{k}.pt")
        members.append({"file": f"model_{k}.pt", "width": cfg["width"], "blocks": cfg["blocks"],
                        "advect": cfg.get("advect", False), "flux": cfg.get("flux", False), "diag": cfg.get("diag", False), "from": r.name})
    for f in ("climatology.npy", "static.npy", "stats.json"):
        shutil.copy(first / f, out / f)
    (out / "config.json").write_text(json.dumps({"name": f"stencil-{a.name}", "members": members}, indent=1))
    print(f"{out}: {len(members)} members")


if __name__ == "__main__":
    main()
