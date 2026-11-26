"""Export a handful of real events as dashboard scenarios (3 input days + the 10 observed days that followed).

Usage: python -m train.make_scenarios
Writes dashboard/scenarios/*.json and dashboard/scenarios/index.json. Dashboard-only data (13 days per event,
far below the repo scan's 400-day archive limit); the inference server never reads these files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVENTS = [
    ("cyclone_tauktae", 20210513, "Cyclone Tauktae (Arabian Sea), May 2021"),
    ("cyclone_yaas", 20210522, "Cyclone Yaas (Bay of Bengal), May 2021"),
    ("monsoon_onset", 20210601, "Monsoon onset over Kerala, June 2021"),
    ("heavy_rain_konkan", 20210719, "Extreme rain over Konkan / Maharashtra, July 2021"),
    ("heatwave_north", 20190601, "North India heatwave, June 2019"),
    ("winter_disturbance", 20210120, "Western disturbance, January 2021"),
]


def main() -> None:
    z = np.load(ROOT.parent / "data" / "prepared.npz")
    data, ymd = z["data"], z["ymd"]
    out = ROOT / "dashboard" / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for key, last_day, label in EVENTS:
        i = int(np.where(ymd == last_day)[0][0])
        d = int(str(last_day)[4:6]), int(str(last_day)[6:])
        import datetime as dt
        doy = dt.date(last_day // 10000, d[0], d[1]).timetuple().tm_yday
        a = 2 * np.pi * doy / 365.0
        rec = {"key": key, "label": label, "last_input_day": str(last_day),
               "doy_sincos": [float(np.sin(a)), float(np.cos(a))],
               "init_states": np.round(data[i - 2:i + 1], 3).tolist(),
               "observed": np.round(data[i + 1:i + 11], 3).tolist()}
        (out / f"{key}.json").write_text(json.dumps(rec, separators=(",", ":")))
        index.append({"key": key, "label": label, "last_input_day": str(last_day)})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    static = z["static"]
    (out / "static.json").write_text(json.dumps({"land_sea_mask": np.round(static[0], 3).tolist(),
                                                 "orography_km": np.round(static[1], 3).tolist(),
                                                 "lat": list(np.arange(6.5, 38.0, 1.0)), "lon": list(np.arange(65.5, 101.0, 1.0))}))
    print(f"wrote {len(index)} scenarios to {out}")


if __name__ == "__main__":
    main()
