"""Local judge: scores a model on held-out years with the kit's OWN scoring functions (monsoon_r1.scoring).

Usage:
  python -m train.evaluate --weights runs/<name>            # our model
  python -m train.evaluate --baseline persistence|damped|climatology

What is exact and what is our assumption (see docs/EXPERIMENTS.md):
  exact   : rmse_lat, skill_vs_best_ref, csi, csi_skill, forecast_points, variance_ratio, anti_blur_points,
            make_shock, causality_checks, causality_points, null_consistent, validity (all imported from the kit)
  assumed : the judge's per-window SS = mean over channels of SS computed on leads 1/4/8 stacked; the judge's
            climatology = kit climatology() over the training archive. We use training years only (honest).
  windows : 5 named windows per held-out year matching the hidden case types + every 5th day of the held-out years
            (the named set mirrors the hidden test, the dense set gives a low-noise mean).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, doy_sincos as kit_sincos
from monsoon_r1.stats import climatology as kit_climatology, clim_for

ROOT = Path(__file__).resolve().parent.parent
LEADS = [0, 3, 7]  # day 1, 4, 8
NAMED = {"winter_wd": (1, 25), "heatwave": (5, 15), "onset": (6, 5), "heavy_rain": (7, 20), "cyclone": (10, 20)}
SHOCK_WINDOWS = ("heavy_rain", "onset", "cyclone")


def load(prepared: Path):
    z = np.load(prepared)
    return z["data"], z["ymd"], z["years"]


def make_fn(kind: str, weights: str | None, device: str, data_train=None, slots_train=None):
    if kind == "model":
        from model.simulator import Simulator
        sim = Simulator(Path(weights), device=device, threads=4)
        return sim.simulate, sim.name
    if kind == "persistence":
        return (lambda x, sc, n, p=None: np.repeat((x[-1] + (0 if p is None else p))[None], n, 0)), "persistence"
    raise ValueError(kind)


def date_of(ymd: int) -> dt.date:
    return dt.date(ymd // 10000, ymd // 100 % 100, ymd % 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--baseline", choices=["persistence"])
    ap.add_argument("--prepared", default=str(ROOT.parent / "data" / "prepared.npz"))
    ap.add_argument("--stats", default=str(ROOT.parent / "data" / "stats.json"))
    ap.add_argument("--val-start", type=int, default=2020)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dense-every", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()

    data, ymd, years = load(Path(a.prepared))
    std = np.array(json.loads(Path(a.stats).read_text())["std"], "f4")
    tr = years < a.val_start
    clim366 = kit_climatology(data[tr], ymd[tr])
    fn, name = make_fn("model" if a.weights else a.baseline, a.weights, a.device)

    val_idx = np.where(years >= a.val_start)[0]
    lo, hi = val_idx[0] + 2, len(data) - 11
    named = []
    for y in sorted(set(years[val_idx])):
        for tag, (mo, da) in NAMED.items():
            i = int(np.where(ymd == y * 10000 + mo * 100 + da)[0][0])
            if lo <= i <= hi:
                named.append((f"{tag}_{y}", i))
    dense = [(f"d{i}", i) for i in range(lo, hi + 1, a.dense_every)]

    def run_window(i, pert=None):
        d = date_of(int(ymd[i]))
        sc = kit_sincos(d.timetuple().tm_yday)
        return np.asarray(fn(data[i - 2:i + 1], sc, 10, pert), "f4")

    def window_scores(i, pred):
        truth = data[i + 1:i + 11]
        pers = np.repeat(data[i][None], 10, 0)
        clim = np.stack([clim_for(clim366, date_of(int(ymd[i + 1 + k]))) for k in range(10)])
        L = LEADS
        sk = S.skill_vs_best_ref(pred[L], truth[L], pers[L], clim[L])
        c_m, c_p, c_c = S.csi(pred[L, 1], truth[L, 1]), S.csi(pers[L, 1], truth[L, 1]), S.csi(clim[L, 1], truth[L, 1])
        sk4 = S.skill_vs_best_ref(pred[[3]], truth[[3]], pers[[3]], clim[[3]])
        perlead = float(np.mean([np.nanmean(S.skill_vs_best_ref(pred[[l]], truth[[l]], pers[[l]], clim[[l]])["ss"]) for l in L]))
        return {"ss": float(np.nanmean(sk["ss"])), "ss_ch": sk["ss"].tolist(), "ss_perlead": perlead,
                "ss_persist_d4": float(np.nanmean(sk4["ss_persist"])),
                "csi_skill": S.csi_skill(c_m, c_p, c_c), "csi": c_m,
                "var_ratio": S.variance_ratio(pred[7], truth[7]).tolist(),
                "valid": S.validity(pred, 10)}

    res = {}
    for label, wins in (("named", named), ("dense", dense)):
        rows = []
        for tag, i in wins:
            r = window_scores(i, run_window(i)); r["tag"] = tag; rows.append(r)
        fp = S.forecast_points([r["ss"] for r in rows], [r["csi_skill"] for r in rows])
        ab = S.anti_blur_points(np.array([r["var_ratio"] for r in rows]))
        res[label] = {"n": len(rows), "forecast": fp, "anti_blur_pts": ab,
                      "ss_persist_d4_mean": float(np.mean([r["ss_persist_d4"] for r in rows])),
                      "ss_perlead_mean": float(np.mean([r["ss_perlead"] for r in rows])),
                      "ss_per_channel": np.nanmean(np.array([r["ss_ch"] for r in rows]), 0).round(3).tolist(),
                      "invalid": [r["tag"] for r in rows if r["valid"]],
                      "rows": [{k: r[k] for k in ("tag", "ss", "csi_skill", "ss_persist_d4")} for r in rows] if label == "named" else []}

    caus = []
    for tag, i in named:
        if not tag.startswith(SHOCK_WINDOWS):
            continue
        base = run_window(i)
        null = run_window(i, np.zeros((6, 32, 36), "f4"))
        shocks = []
        for sname, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
            pert = S.make_shock(sname, data[i], std)
            chk = S.causality_checks(base, run_window(i, pert), box, std)
            shocks.append(chk)
        nok = S.null_consistent(base, null, std)
        caus.append({"tag": tag, "points": S.causality_points(shocks, nok), "null_ok": nok,
                     "shocks": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for s in shocks]})

    n = res["named"]
    total = n["forecast"]["points"] + n["anti_blur_pts"] + float(np.mean([c["points"] for c in caus]))
    summary = {
        "model": name,
        "auto_points_of_70_named": round(total, 2),
        "named": {k: v for k, v in n.items()},
        "dense": {k: v for k, v in res["dense"].items() if k != "rows"},
        "causality": caus,
        "gate_beats_persistence_d4": n["ss_persist_d4_mean"] > 0,
    }
    txt = json.dumps(summary, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(txt)
    print(f"== {name}: auto points (of 70: skill 45 + blur 10 + causality 15) on named windows = {total:.2f}")
    for label in ("named", "dense"):
        r = res[label]
        print(f"  {label:5s} n={r['n']:3d}  meanSS={r['forecast']['mean_ss']:+.3f} rmse_pts={r['forecast']['rmse_pts']:.1f}/30  "
              f"csi_skill={r['forecast']['mean_csi_skill']:+.3f} csi_pts={r['forecast']['csi_pts']:.1f}/15  "
              f"blur={r['anti_blur_pts']:.1f}/10  ss_vs_persist_d4={r['ss_persist_d4_mean']:+.3f}  invalid={r['invalid']}")
        print(f"        alt aggregation (SS per lead, then mean) = {r['ss_perlead_mean']:+.3f} -> rmse_pts {30*min(1,max(0,r['ss_perlead_mean']/0.12)):.1f}/30")
        print(f"        ss per channel t2m,tp,u10,v10,msl,tcwv = {r['ss_per_channel']}")
    for c in caus:
        s = c["shocks"]
        print(f"  causality {c['tag']}: {c['points']:.2f}/15 null_ok={c['null_ok']} | " +
              " | ".join(f"nz={x['nonzero']} loc={x['localized']}({x.get('near_frac_day1', 0):.2f}) prop={x['propagates']} "
                         f"stab={x['stable']} far={x.get('far_frac_max', 0):.3f}" for x in s))


if __name__ == "__main__":
    main()
