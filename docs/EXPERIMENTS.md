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

## Anomaly damping (post-processing) and aggregation ambiguity

The contract does not say how the judge averages skill over leads 1/4/8. Our local judge pools the three leads
into one RMSE per channel (as `rmse_lat` does over leading dims). An independent review flagged the alternative
"SS per lead, then mean", which is harsher because day 8 is weaker than climatology. `train/evaluate.py` now
prints both ("alt aggregation").

ens5 per lead (dense 144, 2020–21): SS vs best ref +0.217 (day 1), +0.024 (day 4), −0.067 (day 8).

Damping: out_k = clim_k + λ[k,c]·(out_k − clim_k), λ fitted by least squares on **2020**, scored on **2021 only**
(71 dense windows; out of sample for both the members and λ):

| variant | pooled SS | per-lead SS | CSI skill | causality (3 windows) | decision |
|---|---:|---:|---:|---|---|
| ens5, no damping | +0.120 | +0.064 | +0.100 | 13.12 ×3 | – |
| damp all channels | +0.134 | +0.080 (quick test) | +0.090 | 13.12, 11.25, 13.12 | rejected: CSI −1.6 pts, heavy-rain shock stops propagating |
| **damp all except rain** | **+0.121** | **+0.066** | **+0.100** | 13.12 ×3 | **adopted**: never worse under either reading |

Shipped λ: refit on 2020+2021 (238 windows) with the selection members (out of sample), rain λ = 1:
day 1 all 1.0; day 8 t2m 0.82, u10 0.60, v10 0.56, msl 0.89, tcwv 0.68; day 14 0.25–0.60.

Independent review (fresh agent): no gate or disqualification risk found; fresh clone passes the offline container on
py3.12/torch 2.15 and py3.10/torch 2.1.2. Seasonal causality over 72 windows (every 10th day of 2020–21, ens5):
warm_ocean propagates 88%, stable 96% (winter weakest); monsoon_stall propagates 8%; mean 12.97/15, min 11.25.
Anti-blur on a random draw of 5 windows: 24.5% chance of 8.33 instead of 10 (dry-day rain variance), expected 9.61.
Fixed: OverflowError → 422, NaN input → climatology fill (tests added).

## Calibrated ensemble probabilities (dashboard risk maps; not part of /simulate)

p_raw = share of the 5 members above the threshold; p_cal = w[k]·p_raw + (1 − w[k])·base[month, cell], where base =
historical frequency in training years (< 2020) and w fitted by least squares per lead. Brier skill score vs base.
Fit on 2020, scored on 2021 (selection ensemble ens5, every 3rd day):

| event | day 1 raw → cal | day 4 raw → cal | day 8 raw → cal | mean BSS days 1–10 (cal) |
|---|---|---|---|---:|
| heavy rain (> 10 mm/day) | +0.200 → **+0.328** | −0.618 → **+0.030** | −0.923 → **+0.004** | +0.059 |
| heat (daily-mean t2m ≥ 34 °C) | +0.557 → **+0.578** | +0.115 → **+0.195** | −0.017 → **+0.090** | +0.187 |

Raw member counts are over-confident after day 1 (5 similar members agree too often); calibration makes them beat
"normal odds" at every lead. Shipped calibration refit on 2020–2021 (w_rain 0.59 → 0.03, w_heat 0.83 → 0.33 over 10 days).

## Final decision: ship the selection ensemble (ens5), no damping, plus risk calibration

Seasonal causality sweep (`python -m train.causality_sweep`, both official shocks on every 10th day of 2020–21,
72 windows, null shock identical in 72/72 for all):

| variant | causality mean /15 | warm_ocean propagates / stable | monsoon_stall propagates / stable |
|---|---:|---|---|
| **ens5 (2005–2019 members), no damping — SHIPPED** | **13.10** | 89% / 94% | 15% / 100% |
| ens5 + damping (rain λ=1) | 12.94 | 85% / 94% | 11% / 100% |
| all-years members (2005–2021), no damping | 12.76 | 81% / 92% | 11% / 97% |
| all-years members + damping | 12.60 | 74% / 92% | 10% / 97% |

- Damping: +0.001 pooled / +0.002 per-lead SS on 2021 (EXPERIMENTS above) vs −0.16 causality → not shipped (code kept, optional).
- All-years retrain: forecast benefit unmeasurable (no held-out years left), −0.34 causality in this sweep → not shipped.
  Evidence strength: one sweep each, 72 correlated windows; differences of ~0.3 may be partly noise. We chose the
  variant whose every number is measured out of sample.
- Consequence: **every held-out number in this log is a number of the shipped model**, and the 2021 dashboard
  scenarios are genuinely unseen by it.
- Shipped weights: `weights/` = ens5 members (v1, v1s1, v1s2, adv1, adv1s1) + `risk.npz` (calibration fitted on
  2020–21 with these members). 15 tests pass; offline container: validator all pass, 0.26 s/call, 199 MB.

## Arabian Sea +2.5 °C heat shock (organisers' deck example)

The organisers' presentation (Track 2, slide "Shock the System. Watch It Ripple") gives this example: "Drop a +2.5°C
Arabian Sea heat anomaly … realistic atmospheric ripples must propagate". Shipped model (ens5). Shock = +2.5 K on t2m
over the official Arabian Sea box (lat 8–20, lon 65–75), on every 10th day of 2020–21 (72 windows), with the judge's
`causality_checks`:

| shock | nonzero | localised | propagates | stable | no teleport |
|---|---:|---:|---:|---:|---:|
| +2.5 K t2m | 100% | 100% | 29% | 100% | 100% |
| +2.5 K t2m and +15% tcwv | 100% | 100% | 33% | 100% | 100% |

Why propagation is seasonal (measured; box-mean Δt2m e-folds in about 4–5 days because there is no ocean model):

| start day | box wind (u, v) m/s | where the response goes on days 4–8 (energy centre, outside box) | share over land |
|---|---|---|---:|
| 2021-07-01 | +7.2, +0.3 (westerly monsoon flow) | 16.8 °N 72.4 °E (onto the Konkan coast) | 42% |
| 2021-07-20 | +8.8, +2.3 | 12.7 °N 75.9 °E (onto the peninsula) | 49% |
| 2021-01-20 | −1.3, −3.5 (north-easterly winter flow) | 8.3 °N 70.4 °E (out to sea, south-west) | 8% |
| 2020-01-25 | −1.6, −3.8 | 8.6 °N 71.1 °E (out to sea) | 8% |

The model carries the anomaly **downwind**. In monsoon months it reaches India and passes the propagation check more often.
In winter it drifts over the open sea and partly off the grid (the box touches the domain's western edge), so by the
judge's metric it "does not propagate". Demo it in a monsoon scenario (e.g. Konkan, July 2021).
