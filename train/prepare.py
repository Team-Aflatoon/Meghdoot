"""Load india_climate_train.nc once, build day-of-year slots, climatology and static fields.

Usage: python -m train.prepare --nc ../data/india_climate_train.nc --stats ../data/stats.json
Writes data/prepared.npz (gitignored; the raw archive is never committed, the repo scan forbids it).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import xarray as xr

from model.common import N_DOY, doy_index_from_sincos, doy_sincos

ROOT = Path(__file__).resolve().parent.parent


def slot_for(d: dt.date) -> int:
    # The judge sends sin/cos of 2*pi*doy/365 for the last input day; we assume doy = tm_yday (1..366).
    # Going through sin/cos makes the training slot identical to what the server recovers at inference.
    return doy_index_from_sincos(*doy_sincos(d.timetuple().tm_yday))


def climatology(data: np.ndarray, slots: np.ndarray, smooth_days: int = 15) -> np.ndarray:
    """[N,6,H,W] + slot per day -> [365,6,H,W] mean per slot with a circular running mean (like the kit's)."""
    acc = np.zeros((N_DOY,) + data.shape[1:], "f8")
    cnt = np.zeros(N_DOY)
    for i, k in enumerate(slots):
        acc[k] += data[i]
        cnt[k] += 1
    acc /= np.maximum(cnt, 1)[:, None, None, None]
    h = smooth_days // 2
    out = np.zeros_like(acc)
    for k in range(N_DOY):
        out[k] = np.mean([acc[(k + o) % N_DOY] for o in range(-h, h + 1)], axis=0)
    return out.astype("f4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=str(ROOT.parent / "data" / "india_climate_train.nc"))
    ap.add_argument("--stats", default=str(ROOT.parent / "data" / "stats.json"))
    ap.add_argument("--out", default=str(ROOT.parent / "data" / "prepared.npz"))
    a = ap.parse_args()
    ds = xr.open_dataset(a.nc)
    data = ds["data"].values.astype("f4")
    static = ds["static"].values.astype("f4")
    dates = [dt.date.fromisoformat(str(t)[:10]) for t in ds["time"].values]
    assert all((dates[i + 1] - dates[i]).days == 1 for i in range(len(dates) - 1)), "archive must be contiguous daily"
    slots = np.array([slot_for(d) for d in dates], dtype="i4")
    years = np.array([d.year for d in dates], dtype="i4")
    ymd = np.array([d.year * 10000 + d.month * 100 + d.day for d in dates], dtype="i4")
    static_scaled = np.stack([static[0], static[1] / 1000.0]).astype("f4")   # land-sea mask, orography in km
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, data=data, slots=slots, years=years, ymd=ymd, static=static_scaled)
    st = json.loads(Path(a.stats).read_text())
    print(f"days={len(dates)} {dates[0]}..{dates[-1]} static={static.shape} mean={np.round(st['mean'], 2)}")
    print(f"nan={int(np.isnan(data).sum())} tp_min={data[:, 1].min():.3f} t2m_range=({data[:, 0].min():.1f},{data[:, 0].max():.1f})")


if __name__ == "__main__":
    main()
