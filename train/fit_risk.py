"""Calibrated ensemble probabilities for the dashboard's risk maps (not used by the judged /simulate).

p_raw  = share of ensemble members above a threshold (heavy rain > 10 mm/day, heat: daily-mean t2m >= 34 C)
p_cal  = w[k] * p_raw + (1 - w[k]) * base[month, cell]
base   = historical frequency of the event for that calendar month and cell (training years only)
w[k]   = least-squares weight per lead day k, clipped to [0, 1], fitted OUT OF SAMPLE.

Scored with the Brier skill score (BSS) vs the base rate: BSS > 0 means the probabilities beat "normal odds".

Usage:
  python -m train.fit_risk --weights runs/ens5 --fit-years 2020 --eval-years 2021          # evidence
  python -m train.fit_risk --weights runs/ens5 --fit-years 2020 2021 --out weights/risk.npz  # shipped
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from model.common import doy_sincos
from model.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent
EVENTS = (("rain", 1, 10.0), ("heat", 0, 307.15))
STEPS = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fit-years", type=int, nargs="+", required=True)
    ap.add_argument("--eval-years", type=int, nargs="*", default=[])
    ap.add_argument("--base-before", type=int, default=2020, help="base rates from years < this")
    ap.add_argument("--out")
    ap.add_argument("--prepared", default=str(ROOT.parent / "data" / "prepared.npz"))
    a = ap.parse_args()
    z = np.load(a.prepared)
    data, ymd, years = z["data"], z["ymd"], z["years"]
    dates = [dt.date(int(v) // 10000, int(v) // 100 % 100, int(v) % 100) for v in ymd]
    mon = np.array([d.month for d in dates])
    tr = years < a.base_before
    base = {n: np.stack([(data[tr & (mon == m), ch] >= thr).mean(0) for m in range(1, 13)]).astype("f4")
            for n, ch, thr in EVENTS}
    sim = Simulator(Path(a.weights), device="cuda")

    def collect(yrs):
        rows = {n: [] for n, _, _ in EVENTS}
        for i in range(2, len(data) - STEPS - 1, 3):
            if years[i] not in yrs or years[i + STEPS] not in yrs:
                continue
            M = sim.members(data[i - 2:i + 1], doy_sincos(dates[i].timetuple().tm_yday), STEPS)
            for n, ch, thr in EVENTS:
                p = (M[:, :, ch] >= thr).mean(0)                                   # [STEPS,H,W]
                o = (data[i + 1:i + 1 + STEPS, ch] >= thr).astype("f4")
                c = np.stack([base[n][mon[i + 1 + k] - 1] for k in range(STEPS)])
                rows[n].append((p, o, c))
        return {n: [np.stack(x) for x in zip(*v)] for n, v in rows.items()}

    def fit(d):
        w = {}
        for n, (p, o, c) in d.items():
            dp, do = p - c, o - c
            w[n] = np.clip((dp * do).sum((0, 2, 3)) / np.maximum((dp * dp).sum((0, 2, 3)), 1e-9), 0, 1).astype("f4")
        return w

    def bss(d, w):
        out = {}
        for n, (p, o, c) in d.items():
            pc = w[n][None, :, None, None] * p + (1 - w[n][None, :, None, None]) * c
            out[n] = {"raw": 1 - ((p - o) ** 2).mean((0, 2, 3)) / ((c - o) ** 2).mean((0, 2, 3)),
                      "cal": 1 - ((pc - o) ** 2).mean((0, 2, 3)) / ((c - o) ** 2).mean((0, 2, 3))}
        return out

    dfit = collect(set(a.fit_years))
    w = fit(dfit)
    for n in w:
        print(f"{n}: w per lead = {np.round(w[n], 2).tolist()}")
    if a.eval_years:
        r = bss(collect(set(a.eval_years)), w)
        for n, v in r.items():
            print(f"{n}: BSS eval {a.eval_years} day1/4/8  raw={np.round(v['raw'][[0, 3, 7]], 3).tolist()}  "
                  f"calibrated={np.round(v['cal'][[0, 3, 7]], 3).tolist()}  mean10 cal={v['cal'].mean():+.3f}")
    if a.out:
        np.savez(a.out, base_rain=base["rain"], base_heat=base["heat"], w_rain=w["rain"], w_heat=w["heat"],
                 thr_rain=np.float32(10.0), thr_heat=np.float32(307.15))
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
