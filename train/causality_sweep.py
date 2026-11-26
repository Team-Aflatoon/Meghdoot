"""Causality points across all seasons: both official shocks on every Nth day of the given years.

Usage: python -m train.causality_sweep --weights runs/final --years 2020 2021 --every 10
Prints mean/min causality points (of 15) and pass rates of each check per shock.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, doy_sincos

from model.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021])
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--prepared", default=str(ROOT.parent / "data" / "prepared.npz"))
    ap.add_argument("--stats", default=str(ROOT.parent / "data" / "stats.json"))
    a = ap.parse_args()
    z = np.load(a.prepared)
    data, ymd, years = z["data"], z["ymd"], z["years"]
    std = np.array(json.loads(Path(a.stats).read_text())["std"], "f4")
    sim = Simulator(Path(a.weights), device="cuda")
    pts, rates = [], {s: {k: 0 for k in ("nonzero", "localized", "propagates", "stable")} for s in ("warm_ocean", "monsoon_stall")}
    nulls = 0
    idx = [i for i in range(2, len(data) - 11, a.every) if years[i] in a.years]
    for i in idx:
        d = dt.date(int(ymd[i]) // 10000, int(ymd[i]) // 100 % 100, int(ymd[i]) % 100)
        sc = doy_sincos(d.timetuple().tm_yday)
        base = sim.simulate(data[i - 2:i + 1], sc, 10)
        nok = S.null_consistent(base, sim.simulate(data[i - 2:i + 1], sc, 10, np.zeros((6, 32, 36), "f4")), std)
        nulls += nok
        res = []
        for s, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
            c = S.causality_checks(base, sim.simulate(data[i - 2:i + 1], sc, 10, S.make_shock(s, data[i], std)), box, std)
            res.append(c)
            for k in rates[s]:
                rates[s][k] += bool(c[k])
        pts.append(S.causality_points(res, nok))
    n = len(idx)
    print(f"{sim.name}: windows={n}  causality mean={np.mean(pts):.2f}/15  min={np.min(pts):.2f}  null_ok={nulls}/{n}")
    for s, r in rates.items():
        print(f"  {s:13s} " + "  ".join(f"{k}={100 * v / n:.0f}%" for k, v in r.items()))


if __name__ == "__main__":
    main()
