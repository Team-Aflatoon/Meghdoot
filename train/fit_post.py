"""Fit statistical post-processing for an ensemble, OUT OF SAMPLE (years the members never trained on).

  --kind qm         per-lead rain quantile mapping  -> rain_qm.npz {pred_q, obs_q} [L, Q]
  --kind bias       per-lead, per-channel additive bias (domain mean)  -> bias.npy [L, 6]
  --kind bias_cell  per-lead, per-channel, per-cell additive bias (shrunk toward the domain mean) -> bias.npy [L, 6, H, W]

The forecasts are made with the weights dir as configured (so fit qm first, copy it in, then fit bias on top).
Rain bias is set to 0 (rain is handled by quantile mapping and must stay >= 0).

Usage: python -m train.fit_post --weights runs/X --years 2020 --kind qm --out runs/X/rain_qm.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from model.common import doy_sincos
from model.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--kind", choices=["qm", "bias", "bias_cell"], required=True)
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--shrink", type=float, default=30.0, help="bias_cell: windows-equivalent shrinkage to the domain mean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepared", default=str(ROOT.parent / "data" / "prepared.npz"))
    a = ap.parse_args()
    z = np.load(a.prepared)
    data, ymd, years = z["data"], z["ymd"], z["years"]
    sim = Simulator(Path(a.weights), device="cuda")
    P, T = [], []
    for i in range(2, len(data) - a.steps - 1, a.every):
        if years[i] not in a.years or years[i + a.steps] not in a.years:
            continue
        d = dt.date(int(ymd[i]) // 10000, int(ymd[i]) // 100 % 100, int(ymd[i]) % 100)
        P.append(sim.simulate(data[i - 2:i + 1], doy_sincos(d.timetuple().tm_yday), a.steps))
        T.append(data[i + 1:i + 1 + a.steps])
    P, T = np.stack(P), np.stack(T)                     # [N, L, 6, H, W]
    n = len(P)
    if a.kind == "qm":
        qs = np.concatenate([np.linspace(0, 0.9, 91), np.linspace(0.901, 0.999, 99)])
        pred_q = np.stack([np.quantile(P[:, k, 1], qs) for k in range(a.steps)]).astype("f4")
        obs_q = np.stack([np.quantile(T[:, k, 1], qs) for k in range(a.steps)]).astype("f4")
        pred_q = np.maximum.accumulate(pred_q + np.arange(len(qs))[None] * 1e-6, axis=1)   # strictly increasing
        np.savez(a.out, pred_q=pred_q, obs_q=obs_q)
        print(f"windows={n}  rain 99th pct pred->obs per lead 1/4/8: "
              + ", ".join(f"{pred_q[k, -10]:.1f}->{obs_q[k, -10]:.1f}" for k in (0, 3, 7)))
    else:
        err = (P - T).mean(0)                           # [L, 6, H, W]
        dom = err.mean((2, 3))                          # [L, 6]
        if a.kind == "bias":
            b = dom.copy(); b[:, 1] = 0.0
        else:
            w = n / (n + a.shrink)                      # shrink noisy per-cell bias toward the domain mean
            b = w * err + (1 - w) * dom[:, :, None, None]; b[:, 1] = 0.0
        np.save(a.out, b.astype("f4"))
        print(f"windows={n}  domain-mean bias day 1/4/8 (t2m K, u10, v10 m/s, msl Pa, tcwv):")
        for k in (0, 3, 7):
            print(f"  day {k + 1}: " + " ".join(f"{v:+.3f}" for v in dom[k, [0, 2, 3, 4, 5]]))


if __name__ == "__main__":
    main()
