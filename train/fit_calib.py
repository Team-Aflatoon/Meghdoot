"""Fit the calibration of the UNSHOCKED forecast (see model/simulator.py, calib.npz), out of sample.

  lam[k, c]  least-squares shrink of the forecast anomaly toward climatology per lead k and channel c (clipped to 0..1)
  tau10[k]   (only with --rain-event) threshold on the RAW ensemble rain that decides the > 10 mm/day event per lead,
             chosen to maximise the heavy-rain CSI skill against the better of persistence and climatology

Usage: python -m train.fit_calib --weights runs/X --years 2020 [--rain-event] [--no-rain-shrink] --out runs/X_cal/calib.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from monsoon_r1 import scoring as S

from model.common import doy_index_from_sincos, doy_sincos, lat_weights
from model.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent
STEPS = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--rain-event", action="store_true")
    ap.add_argument("--no-rain-shrink", action="store_true", help="leave rain unshrunk (lam=1 for rain)")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepared", default=str(ROOT.parent / "data" / "prepared.npz"))
    a = ap.parse_args()
    z = np.load(a.prepared)
    data, ymd, years = z["data"], z["ymd"], z["years"]
    sim = Simulator(Path(a.weights), device="cuda"); sim.calib = None
    clim = sim.clim.cpu().numpy(); w = np.sqrt(lat_weights())[:, None]
    X, T, P, C = [], [], [], []
    for i in range(2, len(data) - STEPS - 1, a.every):
        if years[i] not in a.years or years[i + STEPS] not in a.years:
            continue
        d = dt.date(int(ymd[i]) // 10000, int(ymd[i]) // 100 % 100, int(ymd[i]) % 100)
        sc = doy_sincos(d.timetuple().tm_yday); s0 = doy_index_from_sincos(*sc)
        raw, _ = sim._raw(data[i - 2:i + 1], sc, STEPS)
        X.append(raw); T.append(data[i + 1:i + 1 + STEPS]); P.append(np.repeat(data[i][None], STEPS, 0))
        C.append(np.stack([clim[(s0 + 1 + k) % clim.shape[0]] for k in range(STEPS)]))
    X, T, P, C = map(np.stack, (X, T, P, C))
    lam = np.ones((STEPS, 6), "f4")
    for k in range(STEPS):
        for c in range(6):
            A = ((X[:, k, c] - C[:, k, c]) * w).ravel(); b = ((T[:, k, c] - C[:, k, c]) * w).ravel()
            lam[k, c] = np.clip((A @ b) / max(A @ A, 1e-12), 0.0, 1.0)
    if a.no_rain_shrink:
        lam[:, 1] = 1.0
    res = {"lam": lam}
    if a.rain_event:
        tau = np.full(STEPS, 10.0, "f4"); grid = np.arange(3.0, 16.01, 0.5)
        for k in range(STEPS):
            refs = np.array([np.nanmax([S.csi(P[n, k, 1], T[n, k, 1]), S.csi(C[n, k, 1], T[n, k, 1])]) for n in range(len(X))])
            Tk = T[:, k, 1] > 10

            def val(g):
                Pk = X[:, k, 1] > g
                h = (Pk & Tk).sum((1, 2)); m = (~Pk & Tk).sum((1, 2)); f = (Pk & ~Tk).sum((1, 2))
                with np.errstate(invalid="ignore", divide="ignore"):
                    cs = np.where(h + m + f > 0, h / (h + m + f), np.nan)
                    return np.nanmean(np.where(refs < 1, (cs - refs) / (1 - refs), np.nan))
            tau[k] = max(grid, key=val)
        res["tau10"] = tau
    np.savez(a.out, **res)
    print(f"windows={len(X)}  lam day1/4/8:\n{np.round(lam[[0, 3, 7]], 2)}")
    if "tau10" in res:
        print("tau10:", np.round(res["tau10"], 1).tolist())


if __name__ == "__main__":
    main()
