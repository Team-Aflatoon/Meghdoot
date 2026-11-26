# Meghdoot: a local-stencil climate digital twin of India (Multiverse of Monsoon)

Team Aflatoon. Yugaantar DevForge 2026, Track 2. The 2-page project brief is [docs/BRIEF.pdf](docs/BRIEF.pdf).

`serve.py` runs a learned, physics-aware simulator of India's daily weather. It works on a 32 × 36 grid of 1° cells with 6 variables: 2 m temperature, rain, u and v wind, sea-level pressure and water vapour. It does two things:

- **Timeline Prime:** rolls the weather forward day by day from the last 3 days.
- **Multiverse:** applies a localised shock (for example "warm the Bay of Bengal by 3 K") and shows how it spreads.

## Core idea
- **Daily stepping in anomaly space.** Each member is a residual CNN that steps the atmosphere forward one day at a time. It works on departures from the day-of-year climatology and predicts tomorrow's *change* of that anomaly.
- **No global layers.** The network has no pooling, attention or spatial normalisation, so a shock **cannot jump** across India.
- **Bounded reach per day.** Each daily step can move information at most:
  - 6 cells for the 3 plain members;
  - 8 cells (Chebyshev distance) for the 2 wind-advection members. These first carry the anomaly up to 2 cells along the 10 m wind and then add a CNN residual.

  `tests/test_model.py::test_no_teleport_by_construction` proves this bound bit-exactly. `test_judge_teleport_check_point_shock` checks the judge's own Euclidean teleport metric.
- **Ensemble.** The shipped model (`meghdoot-ens5sk`) averages the rollouts of 5 members (3 plain, 2 advection, different seeds).
- **Metric-aligned loss.** Members are trained on the judge's own skill ratio instead of plain MSE: each sample's error is divided by the error of the better baseline (persistence or climatology) for the same lead and channel, then square-rooted (per-lead and pooled forms mixed 50/50). Plain MSE is dominated by late leads and stormy windows; this loss weights every lead and window as the judge does. Held-out 2020–21 (144 windows): pooled skill +0.154 (was +0.113 with MSE), per-lead skill +0.112 (was +0.058), heavy-rain CSI skill +0.105. The idea came from an independent research pass that had not seen our experiment history.
- **Calibrated risk maps (dashboard).** The 5 members also give probabilities of heavy rain and heat. They are blended with historical frequencies using weights fitted out of sample, and they beat "normal odds" at every lead on held-out 2021 (`train/fit_risk.py`, docs/EXPERIMENTS.md).
- **Training.** The model is trained on its own unrolled predictions over 1 → 2 → 4 → 8 days, so rollouts stay stable.

## Run (judge path)
```bash
python serve.py --port 8000        # offline, CPU; loads weights/ once at start-up
```
- `POST /health` returns `{"status": "ready", ...}`.
- `POST /simulate` follows the kit's `docs/R1_CONTRACT.md`.
- Dependencies: torch, numpy, fastapi, uvicorn (all in the judge image).

## Run the dashboard (separate from the server)
```bash
python serve.py --port 8000                                   # terminal 1, repo root
cd dashboard && python -m http.server 5173                    # terminal 2
# open http://localhost:5173  (every map comes from a live /simulate call)
```

## Reproduce the shipped model
Data: download `india_climate_train.nc` and `stats.json` from https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0 into `../data/`.

```bash
python -m train.prepare                                    # -> ../data/prepared.npz (kept outside the repo)
SK="--skill-loss 1 --schedule 1:10,2:4,4:6,8:6"            # metric-aligned loss, unrolled to 8 days
for s in 0 1 2; do python -m train.train --name sk_p$s --seed $s $SK; done                    # trains on 2005-2019
for s in 0 1;   do python -m train.train --name sk_a$s --seed $s --advect --blocks 4 $SK; done
python -m train.make_ensemble --name ens5sk runs/sk_p0 runs/sk_p1 runs/sk_p2 runs/sk_a0 runs/sk_a1
python -m train.evaluate --weights runs/ens5sk --device cuda        # local judge on held-out 2020-2021
python -m train.causality_sweep --weights runs/ens5sk               # both shocks, 72 windows, all seasons
python -m train.fit_risk --weights runs/ens5sk --fit-years 2020 2021 --out runs/ens5sk/risk.npz
cp runs/ens5sk/* weights/
pytest -q && bash scripts/judge_check.sh                   # tests + offline judge replica in Docker
```
The shipped members are trained on 2005–2019 only, so every held-out number in docs/EXPERIMENTS.md is a number of the
shipped model. We also tried damping long-lead anomalies toward climatology and retraining on all years. Neither is shipped; the evidence is in EXPERIMENTS.md.
The earlier MSE-loss recipe (`v1s0..2`, `adv1s0..1`, default `--schedule 1:12,2:6,4:6,6:6`) is kept in docs/EXPERIMENTS.md as history (`v1s0` is called `v1` and `adv1s0` is called `adv1` there).

## Results
Held-out 2020–21 summary for the shipped model: pooled skill +0.154, per-lead +0.112, heavy-rain CSI skill +0.105, anti-blur 10/10, causality sweep 13.39/15, 0.25 s per call in the judge container, tests 15/15. Every number, with how it was measured, is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). A team guide (architecture, glossary, Q&A) is in [docs/explainer.html](docs/explainer.html).

## Repository map
`serve.py` (API) · `model/` (inference: numpy + torch only) · `train/` (data, training, local judge, ensemble, damping) ·
`weights/` (shipped model) · `dashboard/` (static UI, calls the live `/simulate`) · `tests/` · `scripts/` (offline judge replica in Docker).
