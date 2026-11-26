"""Inference server for the judge.  python serve.py --port 8000

Offline, CPU only. Loads weights/ once at start-up; /health reports ready only after the model is loaded.
API contract: kits/yug-monsoon-kit docs/R1_CONTRACT.md (physical units, nested JSON lists).
"""
import argparse
import base64
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model.simulator import Simulator

MAX_STEPS = 30
SIM = Simulator(threads=4)           # loaded once, at import time
app = FastAPI()
# The dashboard (separate folder, static files) calls this API from the browser.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"], allow_headers=["*"])


def _arr(x, shape):
    if isinstance(x, dict):
        a = np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    else:
        a = np.asarray(x, dtype="f4")
    if a.shape != shape:
        raise ValueError(f"expected shape {shape}, got {a.shape}")
    return a


@app.post("/health")
@app.get("/health")
def health():
    return {"status": "ready", "model_name": SIM.name}


@app.post("/simulate")
async def simulate(request: Request):
    t0 = time.perf_counter()
    try:
        body = await request.json()
        init = _arr(body["init_states"], (3, 6, 32, 36))
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"], (6, 32, 36))
        steps = int(body.get("steps", 10))
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be 1..{MAX_STEPS}")
        sc = body["doy_sincos"]
        if len(sc) != 2:
            raise ValueError("doy_sincos must be [sin, cos]")
        out = SIM.simulate(init, sc, steps, pert)
    except (KeyError, ValueError, TypeError, OverflowError) as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return {"rollout": out.tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
