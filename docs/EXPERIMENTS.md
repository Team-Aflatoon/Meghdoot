# Experiment log

Every number below is from `python -m train.evaluate` (local judge, kit's own scoring functions) on held-out
years 2020–2021, unless marked otherwise. "named" = 10 windows matching the hidden case types
(winter WD 25 Jan, heatwave 15 May, onset 5 Jun, heavy rain 20 Jul, cyclone 20 Oct; × 2020, 2021).
"dense" = every 5th day of 2020–2021 (144 windows, low noise). Points are of 70 (skill 45 + anti-blur 10 +
causality 15); repo scan (15) and brief (15) are scored separately.

Evidence strength: 1 run each unless stated. Differences below ~1 point on "named" are within noise
(only 10 windows); trust "dense" for skill comparisons.

| id | what | named pts/70 | named meanSS | dense meanSS | CSI skill (named/dense) | causality/15 | notes |
|---|---|---:|---:|---:|---|---:|---|
| B0 | persistence (repeat last day) | 21.25 | −0.180 | −0.181 | −0.044 / −0.029 | 11.25 | blur 10/10; never propagates |
| S0 | smoke: 2 epochs (unroll 1 then 4) | 54.38 | +0.071 | +0.059 | +0.102 / +0.076 | 11.25–13.1 | proves pipeline; "propagates" mostly fails: shock energy decays before leaving box+3 cells |
| v1 | 96w×5 blocks, radius 6, schedule 1:12,2:6,4:6,6:6, rain×3, seed 0 | 67.50 | +0.123 | +0.076 | +0.142 / +0.085 | 12.50 avg (6 windows) | stab fails 3/12 shocks (rain spike >5σ); monsoon_stall rarely propagates |
| v1s1 | v1 recipe, seed 1 | 65.20 | +0.105 | +0.073 | +0.133 / +0.082 | – | seed spread of a single model ≈ ±0.02 named SS (named set is noisy) |
| adv1 | + semi-Lagrangian wind step (cap 2 cells), 4 blocks (reach 8) | 67.55 | +0.119 | +0.071 | +0.148 / +0.082 | 12.81 avg | bundled change (advect + fewer blocks); no clear win alone, 1 run |
| ens2 | mean of v1, v1s1 | 68.12 | +0.141 | +0.099 | +0.133 / +0.089 | 13.12 avg | ensemble +0.025 dense SS over best member |
| **ens3** | **mean of v1, v1s1, adv1 (promoted to weights/)** | **68.44** | **+0.151** | **+0.106** | +0.145 / +0.090 | **13.44 avg; stable 12/12** | diversity helps; rain spikes averaged out; stall propagation still fails 5/6 |

Container check (scripts/judge_check.sh, offline, 4 CPU, 6 GB, read-only): ens3 passes the kit schema validator,
0.18–0.21 s per 10-step call, ready in 5 s, 196 MB RAM.
