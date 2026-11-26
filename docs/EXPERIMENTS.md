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
| v1 | 96w×5 blocks, radius 6, schedule 1:12,2:6,4:6,6:6, rain loss weight 1+3 where >10 mm, seed 0 | 67.50 | +0.123 | +0.076 | +0.142 / +0.085 | 12.50 avg (6 windows) | stab fails 3/12 shocks (rain spike >5σ); monsoon_stall rarely propagates |
| v1s1 | v1 recipe, seed 1 | 65.20 | +0.105 | +0.073 | +0.133 / +0.082 | – | seed spread of a single model ≈ ±0.02 named SS (named set is noisy) |
| adv1 | + semi-Lagrangian wind step (cap 2 cells), 4 blocks (reach 8) | 67.55 | +0.119 | +0.071 | +0.148 / +0.082 | 12.81 avg | bundled change (advect + fewer blocks); no clear win alone, 1 run |
| ens2 | mean of v1, v1s1 | 68.12 | +0.141 | +0.099 | +0.133 / +0.089 | 13.12 avg | ensemble +0.025 dense SS over best member |
| **ens3** | **mean of v1, v1s1, adv1 (promoted to weights/)** | **68.44** | **+0.151** | **+0.106** | +0.145 / +0.090 | **13.44 avg; stable 12/12** | diversity helps; rain spikes averaged out; stall propagation still fails 5/6 |

Container check (scripts/judge_check.sh, offline, 4 CPU, 6 GB, read-only): ens3 passes the kit schema validator,
0.18–0.21 s per 10-step call, ready in 5 s, 196 MB RAM.

Gate (beat persistence at day 4, mean SS vs persistence): ens3 named +0.237, dense +0.224; v1 named +0.219, dense +0.198.
Teleport check (judge's Euclidean `far_frac_max`, must be < 0.05): 0.000 on all 12 ens3 shocks. Our bit-exact test bounds
reach in Chebyshev distance (8 cells/day for ens3). On day 1 a diagonal corner of that square lies 11.3 cells away, beyond
the judge's 3 + 8 = 11, so the Euclidean bound is not strict; tests/test_model.py::test_judge_teleport_check_point_shock
checks the judge's own metric on a point shock.

| id | what | named pts/70 | named meanSS | dense meanSS | CSI skill (named/dense) | causality/15 | notes |
|---|---|---:|---:|---:|---|---:|---|
| v1s2 | v1 recipe, seed 2 | – | – | – | – | – | member only |
| adv1s1 | adv1 recipe, seed 1 | 67.81 | +0.129 | +0.075 | +0.128 / +0.087 | – | single members cluster at dense +0.071–0.076 |
| **ens5** | **mean of v1, v1s1, v1s2, adv1, adv1s1 (promoted)** | 68.12 | **+0.163** | **+0.113** | +0.147 / +0.095 | 13.12 in 6/6 | dense ≈ 28.1 + 14.3 + 10 + 13.1 = 65.5/70; gate dense +0.231; warm_ocean full marks 6/6, monsoon_stall never propagates |

ens5 container check: validator all pass, 0.34–0.48 s per call (5 members), ready in 6 s, 201 MB.
