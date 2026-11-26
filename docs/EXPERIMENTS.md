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
