"""Shared constants and helpers used by BOTH training and the inference server, so the server runs exactly the
code path that was evaluated. Inference-side: only numpy + torch (both available in the judge container)."""
from __future__ import annotations

import math

import numpy as np

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
NC, NLAT, NLON = 6, 32, 36
T_IN = 3
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0)
N_DOY = 365
TP = 1  # rain channel index

# Physical validity bounds enforced on every output (the judge's gate: rain >= 0, t2m within -40..+60 C).
T2M_MIN, T2M_MAX = 233.15 + 1.0, 333.15 - 1.0


def doy_index_from_sincos(s: float, c: float) -> int:
    """Day-of-year slot 0..364 recovered from [sin, cos] of 2*pi*doy/365. Same mapping in training and serving."""
    ang = math.atan2(s, c)
    return int(round(ang / (2 * math.pi) * N_DOY)) % N_DOY


def doy_sincos(idx: int) -> tuple[float, float]:
    a = 2 * math.pi * (idx % N_DOY) / N_DOY
    return math.sin(a), math.cos(a)


def lat_weights() -> np.ndarray:
    w = np.cos(np.deg2rad(LAT))
    return (w / w.mean()).astype("f4")
