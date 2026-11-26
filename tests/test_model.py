"""Guarantees we claim in the brief, each with a test. Run: pytest -q  (MONSOON_WEIGHTS=runs/<name> to test a run)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from model.simulator import Simulator

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.environ.get("MONSOON_WEIGHTS", ROOT / "weights"))
STD = np.array(json.loads((WEIGHTS / "stats.json").read_text())["std"], "f4")


@pytest.fixture(scope="module")
def sim():
    return Simulator(WEIGHTS, device="cpu", threads=4)


def realistic_init(sim, doy=200):
    clim = sim.clim.numpy()
    rng = np.random.default_rng(0)
    x = np.stack([clim[(doy - 2 + i) % 365] for i in range(3)]).astype("f4")
    x += rng.normal(0, 0.3, x.shape).astype("f4") * STD[None, :, None, None]
    x[:, 1] = np.maximum(x[:, 1], 0)
    return x


def sincos(doy):
    a = 2 * np.pi * doy / 365
    return [float(np.sin(a)), float(np.cos(a))]


def test_shape_and_validity_gate(sim):
    out = sim.simulate(realistic_init(sim), sincos(200), 10)
    assert out.shape == (10, 6, 32, 36)
    assert np.isfinite(out).all()
    assert out[:, 1].min() >= 0                                   # rain never negative
    assert 233.15 <= out[:, 0].min() and out[:, 0].max() <= 333.15  # t2m within -40..+60 C


def test_extreme_inputs_stay_valid(sim):
    x = realistic_init(sim)
    x[-1, 0] += 25.0           # absurd heat
    x[-1, 1] *= 20.0           # absurd rain
    x[-1, 2:4] *= 5.0          # absurd wind
    out = sim.simulate(x, sincos(140), 30)
    assert np.isfinite(out).all() and out[:, 1].min() >= 0
    assert 233.15 <= out[:, 0].min() and out[:, 0].max() <= 333.15


def test_null_shock_is_identical(sim):
    x = realistic_init(sim)
    a = sim.simulate(x, sincos(200), 10, None)
    b = sim.simulate(x, sincos(200), 10, np.zeros((6, 32, 36), "f4"))
    assert np.abs(a - b).max() == 0.0


def test_deterministic(sim):
    x = realistic_init(sim)
    assert np.array_equal(sim.simulate(x, sincos(10), 10), sim.simulate(x, sincos(10), 10))


def test_no_teleport_by_construction(sim):
    """A one-cell shock can move at most `radius` cells per day: everything beyond radius*(k+1) is EXACTLY unchanged."""
    x = realistic_init(sim)
    base = sim.simulate(x, sincos(200), 6)
    p = np.zeros((6, 32, 36), "f4"); p[0, 16, 18] = 3.0; p[5, 16, 18] = 5.0
    shock = sim.simulate(x, sincos(200), 6, p)
    r = sim.radius
    ii, jj = np.indices((32, 36))
    cheb = np.maximum(np.abs(ii - 16), np.abs(jj - 18))
    for k in range(6):
        far = cheb > r * (k + 1)
        if not far.any():
            break
        assert np.abs(shock[k][:, far] - base[k][:, far]).max() == 0.0, f"response beyond {r*(k+1)} cells on day {k+1}"
    assert np.abs(shock[0] - base[0]).max() > 0          # and it does respond


def test_judge_teleport_check_point_shock(sim):
    """The judge's own Euclidean teleport metric (energy beyond 3 + 8*(day) cells from the box) on a 1-cell box."""
    S = pytest.importorskip("monsoon_r1.scoring")
    x = realistic_init(sim)
    base = sim.simulate(x, sincos(200), 10)
    box = (15.5, 16.5, 82.5, 83.5)                       # covers the single cell at lat 16, lon 83 region
    p = np.zeros((6, 32, 36), "f4")
    from monsoon_r1.contract import box_mask
    p[0][box_mask(box)] = 3.0; p[5][box_mask(box)] = 5.0
    chk = S.causality_checks(base, sim.simulate(x, sincos(200), 10, p), box, STD)
    assert chk["no_teleport"], chk


def test_reach_within_judge_speed_limit(sim):
    assert sim.radius <= 8, "judge flags energy beyond 3 + 8 cells/day as teleportation"


def test_fast_enough(sim):
    x = realistic_init(sim)
    sim.simulate(x, sincos(1), 10)
    t0 = time.perf_counter()
    sim.simulate(x, sincos(1), 10)
    assert time.perf_counter() - t0 < 1.5                 # judge limit is 5.0 s including JSON transport


def test_nan_input_falls_back_to_climatology(sim):
    x = realistic_init(sim)
    x[-1, 0, 10, 10] = np.nan; x[0, 4, 3, 3] = np.inf
    out = sim.simulate(x, sincos(200), 10)
    assert np.isfinite(out).all() and out[:, 1].min() >= 0
    assert 233.15 <= out[:, 0].min() and out[:, 0].max() <= 333.15
