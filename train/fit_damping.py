"""Fit lead-dependent anomaly damping lam[k, c] for an ensemble, OUT OF SAMPLE.

lam = argmin || T - lam * A ||^2 per lead k and channel c (latitude-weighted), where A = forecast anomaly and
T = observed anomaly w.r.t. the model's own climatology; clipped to [0, 1]. Long leads get lam < 1, which
pulls over-confident anomalies toward climatology (the classic optimal damping of anomaly persistence).

Usage: python -m train.fit_damping --weights runs/ens5 --years 2020 2021 [--keep-rain] --out runs/ens5/damping.npy
The years must NOT be training years of the weights (use the val_start split).
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from model.common import doy_index_from_sincos, doy_sincos, lat_weights
from model.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--keep-rain", action="store_true", help="do not damp rain (protects heavy-rain CSI)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepared", default=str(ROOT.parent / "data" / "prepared.npz"))
    a = ap.parse_args()
    z = np.load(a.prepared)
    data, ymd, years = z["data"], z["ymd"], z["years"]
    sim = Simulator(Path(a.weights), device="cuda")
    sim.damping = None
    clim = sim.clim.cpu().numpy()
    w = lat_weights()[:, None]
    num = np.zeros((a.steps, 6)); den = np.zeros((a.steps, 6))
    n = 0
    for i in range(2, len(data) - a.steps - 1, a.every):
        if years[i] not in a.years or years[i + a.steps] not in a.years:
            continue
        d = dt.date(int(ymd[i]) // 10000, int(ymd[i]) // 100 % 100, int(ymd[i]) % 100)
        sc = doy_sincos(d.timetuple().tm_yday)
        s0 = doy_index_from_sincos(*sc)
        pred = sim.simulate(data[i - 2:i + 1], sc, a.steps)
        c = np.stack([clim[(s0 + 1 + k) % clim.shape[0]] for k in range(a.steps)])
        A = pred - c; T = data[i + 1:i + 1 + a.steps] - c
        num += (A * T * w).sum((2, 3)); den += (A * A * w).sum((2, 3))
        n += 1
    lam = np.clip(num / np.maximum(den, 1e-12), 0.0, 1.0).astype("f4")
    if a.keep_rain:
        lam[:, 1] = 1.0
    np.save(a.out, lam)
    print(f"windows={n}  lam (rows = lead 1, 4, 8, 14; cols t2m tp u10 v10 msl tcwv):")
    print(np.round(lam[[0, 3, 7, a.steps - 1]], 2))


if __name__ == "__main__":
    main()
