"""Inference wrapper used by serve.py AND by train/evaluate.py, so what we score locally is what the judge runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .common import LAT, NC, NLAT, NLON, T_IN, doy_index_from_sincos
from .net import StencilNet, rollout

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
# calendar month of each day-of-year slot (non-leap year), for the risk base rates
_MONTH_OF_SLOT = [m for m, n in enumerate((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), start=1) for _ in range(n)]


class Simulator:
    def __init__(self, weights_dir: Path = WEIGHTS, device: str = "cpu", threads: int | None = 4):
        if threads:
            torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.weights_dir = Path(weights_dir)
        cfg = json.loads((weights_dir / "config.json").read_text())
        # An ensemble is a list of members; each member rolls out independently and the rollouts are averaged.
        members = cfg.get("members") or [{"file": "model.pt", "width": cfg["width"], "blocks": cfg["blocks"],
                                          "advect": cfg.get("advect", False), "flux": cfg.get("flux", False)}]
        self.nets = []
        for mcfg in members:
            net = StencilNet(width=mcfg["width"], blocks=mcfg["blocks"], advect=mcfg.get("advect", False),
                             flux=mcfg.get("flux", False))
            net.load_state_dict(torch.load(weights_dir / mcfg["file"], map_location="cpu", weights_only=True))
            self.nets.append(net.eval().to(self.device))
        self.net = self.nets[0]
        self.radius = max(n.radius for n in self.nets)
        self.name = cfg.get("name", "stencil")
        st = json.loads((weights_dir / "stats.json").read_text())
        t = lambda a: torch.as_tensor(np.asarray(a, dtype="f4"), device=self.device)
        self.mean, self.std = t(st["mean"]), t(st["std"])
        self.clim = t(np.load(weights_dir / "climatology.npy"))
        # Optional lead-dependent anomaly damping (statistical post-processing, fitted out-of-sample by
        # train/fit_damping.py): out_k = clim_k + lam[k, c] * (out_k - clim_k), lam in [0, 1]. Leads beyond the
        # fitted range reuse the last row.
        dp = weights_dir / "damping.npy"
        self.damping = np.clip(np.load(dp).astype("f4"), 0.0, 1.0) if dp.exists() else None
        # Optional additive bias correction per lead (and channel, optionally per cell), fitted out of sample by
        # train/fit_bias.py: out_k -= bias[k]. A constant offset is identical with and without a shock, so it cancels
        # in every Multiverse - Prime difference (causality and the null shock are unaffected).
        bp = weights_dir / "bias.npy"
        self.bias = np.load(bp).astype("f4") if bp.exists() else None
        # Optional per-lead quantile mapping of rain (train/fit_rain_qm.py): a fixed monotone curve per lead maps the
        # ensemble's rain quantiles onto observed rain quantiles, restoring heavy rain that averaging smooths away.
        # Pointwise (each cell independently), so locality and the no-teleport guarantee are untouched.
        qp = weights_dir / "rain_qm.npz"
        self.rain_qm = dict(np.load(qp)) if qp.exists() else None
        self.static = t(np.load(weights_dir / "static.npy"))
        self.lat = t(LAT)

    def risk(self, init_states, doy_sincos, steps: int, perturbation=None) -> dict:
        """Calibrated event probabilities per day and cell (dashboard only; see train/fit_risk.py):
        p = w[k] * share_of_members_above_threshold + (1 - w[k]) * historical_frequency[month, cell]."""
        rp = self.weights_dir / "risk.npz"
        if not rp.exists():
            raise ValueError("risk calibration not available")
        r = np.load(rp)
        M = self.members(init_states, doy_sincos, steps, perturbation)
        doy = doy_index_from_sincos(float(doy_sincos[0]), float(doy_sincos[1]))
        out = {}
        for name, ch in (("rain", 1), ("heat", 0)):
            thr, w, base = float(r[f"thr_{name}"]), r[f"w_{name}"], r[f"base_{name}"]
            p = (M[:, :, ch] >= thr).mean(0)
            for k in range(p.shape[0]):
                month = _MONTH_OF_SLOT[(doy + 1 + k) % len(_MONTH_OF_SLOT)]
                wk = float(w[min(k, len(w) - 1)])
                p[k] = wk * p[k] + (1 - wk) * base[month - 1]
            out[f"p_{name}"] = np.clip(p, 0, 1)
        return out

    @torch.no_grad()
    def members(self, init_states: np.ndarray, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        """Raw rollouts of every ensemble member [M, steps, 6, H, W] (no damping). Dashboard risk maps only;
        the judged /simulate path is simulate()."""
        xt, d0, _ = self._prepare(init_states, doy_sincos, perturbation)
        outs = [rollout(n, xt, d0, int(steps), self.clim, self.mean, self.std, self.static, self.lat)[0] for n in self.nets]
        return np.nan_to_num(torch.stack(outs).cpu().numpy().astype("f4"))

    def _prepare(self, init_states, doy_sincos, perturbation):
        x = np.asarray(init_states, dtype="f4")
        if x.shape != (T_IN, NC, NLAT, NLON):
            raise ValueError(f"init_states shape {x.shape}")
        x = x.copy()
        if perturbation is not None:
            p = np.asarray(perturbation, dtype="f4")
            if p.shape != (NC, NLAT, NLON):
                raise ValueError(f"perturbation shape {p.shape}")
            x[-1] = x[-1] + p
        doy = doy_index_from_sincos(float(doy_sincos[0]), float(doy_sincos[1]))
        if not np.isfinite(x).all():               # contract forbids NaN/Inf; if one slips in, use that day's climatology
            clim = self.clim.cpu().numpy()
            for i in range(T_IN):
                c = clim[(doy - (T_IN - 1 - i)) % clim.shape[0]]
                bad = ~np.isfinite(x[i])
                x[i][bad] = c[bad]
        return (torch.as_tensor(x[None], device=self.device),
                torch.tensor([doy], device=self.device, dtype=torch.long), doy)

    @torch.no_grad()
    def simulate(self, init_states: np.ndarray, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        xt, d0, doy = self._prepare(init_states, doy_sincos, perturbation)
        rs = [rollout(net, xt, d0, int(steps), self.clim, self.mean, self.std, self.static, self.lat)[0] for net in self.nets]
        out = rs[0]
        for r in rs[1:]:
            out = out + r
        out = (out / len(self.nets)).cpu().numpy().astype("f4")
        if self.rain_qm is not None:
            P, O = self.rain_qm["pred_q"], self.rain_qm["obs_q"]          # [L, Q] each, increasing
            for k in range(out.shape[0]):
                j = min(k, P.shape[0] - 1)
                x = np.maximum(out[k, 1], 0.0)
                y = np.interp(x, P[j], O[j])
                top = x > P[j, -1]                                         # above the fitted range: keep the offset
                y[top] = x[top] + (O[j, -1] - P[j, -1])
                out[k, 1] = y
        return self._finish(out, doy)

    def _finish(self, out: np.ndarray, doy: int) -> np.ndarray:
        """Bias correction, damping toward climatology, then the validity guards."""
        if self.bias is not None:
            for k in range(out.shape[0]):
                b = self.bias[min(k, self.bias.shape[0] - 1)]
                out[k] -= b if b.ndim == 3 else b[:, None, None]
        if self.damping is not None:
            clim = self.clim.cpu().numpy()
            for k in range(out.shape[0]):
                c = clim[(doy + 1 + k) % clim.shape[0]]
                lam = self.damping[min(k, self.damping.shape[0] - 1)][:, None, None]
                out[k] = c + lam * (out[k] - c)
        # last-resort guard: never emit NaN/Inf or out-of-range values (the judge's validity gate). Non-finite
        # cells fall back to that day's climatology.
        if not np.isfinite(out).all():
            clim = self.clim.cpu().numpy()
            for k in range(out.shape[0]):
                c = clim[(doy + 1 + k) % clim.shape[0]]
                bad = ~np.isfinite(out[k])
                out[k][bad] = c[bad]
        out[:, 0] = np.clip(out[:, 0], 234.15, 332.15)
        out[:, 1] = np.maximum(out[:, 1], 0.0)
        return out

