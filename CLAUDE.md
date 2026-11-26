# Dev-Forge — Yugaantar DevForge 2026, Track 2 "Multiverse of Monsoon" (Team Aflatoon)

**Code freeze: Fri 27 Nov 2026 09:00 IST, absolute.** The last commit before that is the submission.

## What we build
A fast, physics-aware climate digital twin of India: `serve.py` (offline, CPU, 4 cores, 6 GB) answers
`/simulate` with a 10-day rollout from 3 input days, plus localised "multiverse" shocks. Wedge/USP:
**local-stencil neural stepper** — a residual CNN in climatology-anomaly space whose receptive field is 6
cells per simulated day (< 8 allowed), so shocks start local and spread at a bounded speed *by construction*.

## Where things are
- `serve.py` judge entrypoint (FastAPI). `model/` inference code (numpy + torch only; no kit imports, no dates).
- `train/` prepare, train, evaluate (local judge using `monsoon_r1.scoring`), make_scenarios.
- `weights/` the promoted run (model.pt, config.json, stats.json, climatology.npy [365 days], static.npy).
- `dashboard/` static HTML/JS, separate from the server; calls live `/simulate`.
- `tests/` guarantees (validity gate, null shock identical, no-teleport by construction, speed, repo scan).
- Data lives OUTSIDE the repo in `../data/` (repo scan flags multi-year arrays and .nc files).
- `docs/EXPERIMENTS.md` every run with numbers; `PROGRESS.md` state + next steps.

## Rules that cost points if broken (from R1_CONTRACT.md and repo_scan.py)
- Inference code (serve.py + everything it imports) must not reference weather APIs, ≥3 date literals, or
  `datetime(20..`. Never ship an array shaped (>400, …, 32, 36). No `.nc` in the repo.
- serve.py must not import streamlit/plotly/matplotlib. Dashboard must call our live `/simulate`.
- Null shock must equal the unperturbed call; no NaN/Inf; rain ≥ 0; t2m in [−40, +60] °C; 10 steps < 5 s.
- No spatial normalisation/pooling/attention in the net (breaks the locality guarantee; tested).

## How to work
- Score every candidate with `python -m train.evaluate --weights runs/<name> --device cuda` and log it in
  docs/EXPERIMENTS.md (trust the dense 144-window numbers; named = 10 windows is noisy).
- Promote a run by copying its 5 files into `weights/`, then `pytest -q` and the docker offline check.
- Python 3.12 venv at `.venv` (CUDA torch for training). Judge uses CPU torch.
- Playbook: ~/.claude/templates/hackathon/CLAUDE.md (evidence log, tests for every claim, adversarial
  review before submission, docs match shipped code).
