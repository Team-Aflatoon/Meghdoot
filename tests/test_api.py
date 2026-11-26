"""The HTTP contract, exercised in-process (no network): /health, /simulate shapes, units, errors, repo scan."""
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    import serve
    return TestClient(serve.app)


def test_health(client):
    r = client.post("/health")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_simulate_contract(client):
    import serve
    clim = serve.SIM.clim.numpy()
    init = np.stack([clim[100], clim[101], clim[102]]).tolist()
    r = client.post("/simulate", json={"init_states": init, "doy_sincos": [0.94, 0.34], "steps": 10, "perturbation": None})
    assert r.status_code == 200
    out = np.array(r.json()["rollout"], "f4")
    assert out.shape == (10, 6, 32, 36) and np.isfinite(out).all()
    assert isinstance(r.json()["inference_ms"], float)


def test_bad_input_is_422_not_500(client):
    r = client.post("/simulate", json={"init_states": [[1, 2]], "doy_sincos": [0, 1], "steps": 10})
    assert r.status_code == 422


def test_repo_scan_clean():
    scan = pytest.importorskip("monsoon_r1.repo_scan").scan
    res = scan(ROOT)
    crit = [f for f in res["flags"] if f["level"] == "critical"]
    assert not crit, crit
    assert res["anti_cheat_points"] == 8.0, res["flags"]
