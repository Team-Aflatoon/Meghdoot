"""Train the local-stencil stepper with a multi-step (unrolled) curriculum.

Usage:
  python -m train.train --name v1 --val-start 2020                 # honest run: trains on years < 2020
  python -m train.train --name final --val-start 2099               # final run: all years 2005-2021
Writes runs/<name>/{model.pt,config.json,stats.json,climatology.npy,static.npy,log.json}, the same layout as weights/.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from model.common import LAT, lat_weights
from model.net import StencilNet, rollout
from train.prepare import climatology

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--val-start", type=int, default=2020)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--schedule", default="1:12,2:6,4:6,6:6", help="unroll:epochs,... curriculum")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rain-weight", type=float, default=3.0, help="extra loss weight where true rain > 10 mm/day")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--advect", action="store_true", help="semi-Lagrangian wind step before the CNN")
    ap.add_argument("--flux", action="store_true", help="moisture-flux input features (u10*tcwv, v10*tcwv)")
    ap.add_argument("--skill-loss", type=float, default=0.0,
                    help="weight of the metric-aligned loss: error / error of the better baseline (persistence or climatology)")
    ap.add_argument("--prepared", default=str(ROOT.parent / "data" / "prepared.npz"))
    ap.add_argument("--stats", default=str(ROOT.parent / "data" / "stats.json"))
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(a.prepared)
    data, slots, years, static = z["data"], z["slots"], z["years"], z["static"]
    st = json.loads(Path(a.stats).read_text())
    tr_mask = years < a.val_start
    clim = climatology(data[tr_mask], slots[tr_mask])

    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "climatology.npy", clim); np.save(out / "static.npy", static)
    shutil.copy(a.stats, out / "stats.json")
    cfg = {"name": f"stencil-{a.name}", "width": a.width, "blocks": a.blocks, "val_start": a.val_start,
           "schedule": a.schedule, "rain_weight": a.rain_weight, "seed": a.seed, "advect": a.advect, "flux": a.flux, "skill_loss": a.skill_loss, "lr": a.lr, "batch": a.batch}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    T = lambda x: torch.as_tensor(x, device=dev)
    D, C, S_, M = T(data), T(clim), T(np.array(st["std"], "f4")), T(np.array(st["mean"], "f4"))
    STAT, LATt, SL = T(static), T(LAT.astype("f4")), T(slots.astype("i8"))
    latw = T(lat_weights())[None, None, None, :, None]
    net = StencilNet(a.width, a.blocks, advect=a.advect, flux=a.flux).to(dev)
    print(f"params={sum(p.numel() for p in net.parameters()):,} radius={net.radius} cells/step device={dev}")

    sched = [(int(k), int(e)) for k, e in (p.split(":") for p in a.schedule.split(","))]
    total_epochs = sum(e for _, e in sched)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    log, ep_global, t0 = [], 0, time.time()
    train_idx_all = np.where(tr_mask)[0]
    for unroll, epochs in sched:
        idx = train_idx_all[(train_idx_all >= 2) & (train_idx_all + unroll < len(data))]
        idx = idx[tr_mask[np.minimum(idx + unroll, len(data) - 1)]]   # targets must stay inside the training years
        for _ in range(epochs):
            net.train()
            perm = np.random.permutation(idx)
            tot, nb = 0.0, 0
            for b in range(0, len(perm) - a.batch + 1, a.batch):
                lr = a.lr * 0.5 * (1 + math.cos(math.pi * (ep_global + b / len(perm)) / total_epochs))
                for g in opt.param_groups: g["lr"] = lr
                ii = T(perm[b:b + a.batch].astype("i8"))
                win = torch.stack([D[ii + o] for o in range(-2, unroll + 1)], 1)     # [B, 3+unroll, 6, H, W]
                pred = rollout(net, win[:, :3], SL[ii], unroll, C, M, S_, STAT, LATt)
                true = win[:, 3:]
                err = ((pred - true) / S_[None, None, :, None, None]) ** 2
                w = torch.ones_like(err)
                w[:, :, 1] = 1 + a.rain_weight * (true[:, :, 1] > 10.0).float()
                if a.skill_loss > 0:
                    # Metric-aligned loss: the judge scores 1 - RMSE / min(RMSE_persistence, RMSE_climatology) per channel,
                    # pooled over leads (and possibly per lead). Train on that ratio directly so every lead and every window
                    # counts relative to how hard it is, instead of plain MSE, which is dominated by the late leads.
                    sn = S_[None, None, :, None, None]
                    num = (err * w * latw).mean((-2, -1))                                        # [B, K, 6]
                    pers = win[:, 2:3].expand(-1, unroll, -1, -1, -1)                           # repeat the last input day
                    cl = torch.stack([C[(SL[ii] + 1 + k) % C.shape[0]] for k in range(unroll)], 1)
                    e_p = ((((pers - true) / sn) ** 2) * latw).mean((-2, -1))
                    e_c = ((((cl - true) / sn) ** 2) * latw).mean((-2, -1))
                    per_lead = torch.sqrt(num / torch.minimum(e_p, e_c).clamp_min(1e-3) + 1e-8).mean()
                    pooled = torch.sqrt(num.mean(1) / torch.minimum(e_p.mean(1), e_c.mean(1)).clamp_min(1e-3) + 1e-8).mean()
                    loss = a.skill_loss * 0.5 * (per_lead + pooled) + (1 - a.skill_loss) * (err * w * latw).mean()
                else:
                    loss = (err * w * latw).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item(); nb += 1
            ep_global += 1
            log.append({"epoch": ep_global, "unroll": unroll, "loss": tot / max(nb, 1), "lr": lr, "t": round(time.time() - t0, 1)})
            print(f"ep {ep_global:3d}/{total_epochs} unroll={unroll} loss={tot / max(nb, 1):.5f} lr={lr:.2e} t={time.time() - t0:.0f}s", flush=True)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "model.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
