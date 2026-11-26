# Multiverse of Monsoon: a local-stencil climate digital twin of India

Team Aflatoon. Yugaantar DevForge 2026, Track 2.

`serve.py` runs a learned, physics-aware simulator of India's daily weather. It works on a 32 × 36 grid of 1° cells with 6 variables: 2 m temperature, rain, u and v wind, sea-level pressure and water vapour. It does two things:

- **Timeline Prime:** rolls the weather forward day by day from the last 3 days.
- **Multiverse:** applies a localised shock (for example "warm the Bay of Bengal by 3 K") and shows how its effect spreads.

## Core idea in one paragraph
The model is a residual CNN that steps the atmosphere forward one day at a time. It works in *anomaly space*, meaning the departure from the day-of-year climatology. Each step sees only a small neighbourhood of a few grid cells: 6 cells (about 650 km) for the plain members, and 8 cells for the wind-advection member, which carries anomalies up to 2 cells per day along the 10 m wind before its CNN runs. The network has no global layers (no pooling, attention or spatial normalisation), so a shock **cannot jump** across India. It starts where it was applied and spreads at no more than 8 cells per simulated day (Chebyshev distance), which is the judge's own speed limit. That is a property of the architecture, and `tests/test_model.py::test_no_teleport_by_construction` proves it bit-exactly. Training unrolls the model over 1, then 2, 4 and 6 days on its own predictions, so the rollouts stay stable and do not drift. The shipped model is an ensemble of 3 members, and their rollouts are averaged.

## Run (judge path)
```bash
pip install torch numpy fastapi uvicorn        # all present in the judge image
python serve.py --port 8000                    # offline, CPU, loads weights/ at start-up
```
- `POST /health` returns `{"status": "ready", ...}`.
- `POST /simulate` follows the contract in the kit's `docs/R1_CONTRACT.md`.

## Reproduce training
```bash
# data from https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0 into ../data/
python -m train.prepare                                   # -> ../data/prepared.npz (kept outside the repo)
python -m train.train --name v1                           # trains on 2005-2019, about 7 min on an RTX 4050
python -m train.evaluate --weights runs/v1 --device cuda  # local judge on held-out 2020-2021
cp runs/v1/{model.pt,config.json,stats.json,climatology.npy,static.npy} weights/
pytest -q
```

## Results
Every number, with how it was measured, is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Repository map
`serve.py` (API) · `model/` (inference: numpy + torch only) · `train/` (data, training, local judge) ·
`weights/` (shipped model) · `dashboard/` (separate static UI that calls the live `/simulate`) · `tests/` ·
`scripts/` (offline judge replica in Docker).
