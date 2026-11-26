// Multiverse of Monsoon — Climate Digital Twin dashboard.
// Vanilla JS + canvas. Every map shown is LIVE output of the /simulate endpoint (or the observed record);
// nothing is pre-rendered. If the API is unreachable the dashboard shows a red "server offline" state.
'use strict';

const APP_TITLE = 'Meghdoot — Multiverse of Monsoon Climate Twin';
const DEFAULT_BASE = 'http://localhost:8000';

// ---------------------------------------------------------------- grid & channels
const NR = 32, NC = 36, NCH = 6, NCELL = NR * NC, NOBS = 10;
const LAT0 = 6.5, LON0 = 65.5;             // centre of row 0 (south) / col 0 (west)
const LAT_N_EDGE = LAT0 + NR - 0.5;         // 38.0
const LON_W_EDGE = LON0 - 0.5;              // 65.0
const STD = [13.46, 7.55, 2.90, 2.43, 619.0, 20.49];
const ENERGY_TAU = 1e-3;                    // reach threshold on per-cell energy
const NEAR_CELLS = 2;                       // "near" = within this many cells of the box
const HEAT_K = 273.15 + 34;                 // daily-mean t2m >= 34 C
const DRY_MM = 2.5;                         // IMD rainy-day threshold

const CH = [
  { key: 't2m', name: 't2m — 2 m temperature', unit: '°C', dUnit: '°C', off: -273.15, scale: 1, addUnit: 'K', addFactor: 1, cmap: 'temp', signed: false, nonneg: false },
  { key: 'tp', name: 'tp — rain', unit: 'mm/day', dUnit: 'mm/day', off: 0, scale: 1, addUnit: 'mm/day', addFactor: 1, cmap: 'rain', signed: false, nonneg: true },
  { key: 'u10', name: 'u10 — wind, east-west', unit: 'm/s', dUnit: 'm/s', off: 0, scale: 1, addUnit: 'm/s', addFactor: 1, cmap: 'wind', signed: true, nonneg: false },
  { key: 'v10', name: 'v10 — wind, north-south', unit: 'm/s', dUnit: 'm/s', off: 0, scale: 1, addUnit: 'm/s', addFactor: 1, cmap: 'wind', signed: true, nonneg: false },
  { key: 'msl', name: 'msl — sea-level pressure', unit: 'hPa', dUnit: 'hPa', off: 0, scale: 0.01, addUnit: 'hPa', addFactor: 100, cmap: 'msl', signed: false, nonneg: false },
  { key: 'tcwv', name: 'tcwv — column water vapour', unit: 'kg/m²', dUnit: 'kg/m²', off: 0, scale: 1, addUnit: 'kg/m²', addFactor: 1, cmap: 'moist', signed: false, nonneg: true },
];
const disp = (c, v) => (v + CH[c].off) * CH[c].scale;     // physical -> display units
const dispD = (c, d) => d * CH[c].scale;                   // physical delta -> display delta

const REGIONS = [
  { name: 'NW India', s: 24, n: 32, w: 70, e: 78 },
  { name: 'Central India', s: 18, n: 24, w: 74, e: 84 },
  { name: 'Peninsular South', s: 8, n: 18, w: 74, e: 80 },
  { name: 'East & NE India', s: 20, n: 28, w: 84, e: 96 },
];

const PRESETS = {
  bob: { label: 'Warm Bay of Bengal', box: { s: 8, n: 20, w: 82, e: 94 },
    comps: [{ ch: 0, mode: 'add', val: 3 }, { ch: 5, mode: 'pct', val: 15 }] },
  stall: { label: 'Monsoon stall', box: { s: 8, n: 20, w: 65, e: 75 },
    comps: [{ ch: 2, mode: 'pct', val: -40 }, { ch: 3, mode: 'pct', val: -40 }] },
  zero: { label: 'Zero shock', box: { s: 8, n: 20, w: 82, e: 94 }, comps: [] },
};

// ---------------------------------------------------------------- state
const S = {
  base: localStorage.getItem('mom.base') || DEFAULT_BASE,
  online: null, modelName: '', lastMs: null, lastRtt: null,
  index: [], mask: null, land: null, coast: [],
  scen: null,
  steps: 10, day: 1, ch: 0,
  prime: null, primeSteps: 0,
  multi: null, multiSteps: 0, multiDesc: '', multiBox: null,
  shock: { box: null, comps: [], preset: null },
  showObs: false, arrows: false, smooth: false, regions: false,
  busy: false, busyMsg: '', timer: null, drag: null, hover: null,
  rangeCache: new Map(), version: 0,
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const sgn = (v, d = 0) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(d);
const at = (arr, day0, c, cell) => arr[(day0 * NCH + c) * NCELL + cell];   // day0 = 0-based

// ---------------------------------------------------------------- data helpers
function flatten(nested, shape) {
  const n = shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  let k = 0;
  const walk = (x, depth) => {
    if (depth === shape.length) { out[k++] = +x; return; }
    if (!Array.isArray(x) || x.length !== shape[depth]) throw new Error(`bad shape at axis ${depth}: expected ${shape[depth]}, got ${Array.isArray(x) ? x.length : typeof x}`);
    for (let i = 0; i < x.length; i++) walk(x[i], depth + 1);
  };
  walk(nested, 0);
  for (let i = 0; i < n; i++) if (!Number.isFinite(out[i])) throw new Error('non-finite value in response');
  return out;
}
function nestPert(flat) {
  const p = [];
  for (let c = 0; c < NCH; c++) {
    const rows = [];
    for (let r = 0; r < NR; r++) rows.push(Array.from(flat.subarray(c * NCELL + r * NC, c * NCELL + r * NC + NC)));
    p.push(rows);
  }
  return p;
}
function boxIdx(box) {           // cells whose centre lies inside the box (degrees, edges)
  if (!box) return null;
  const r0 = Math.ceil(box.s - LAT0 - 1e-9), r1 = Math.floor(box.n - LAT0 + 1e-9);
  const c0 = Math.ceil(box.w - LON0 - 1e-9), c1 = Math.floor(box.e - LON0 + 1e-9);
  return { r0: Math.max(0, r0), r1: Math.min(NR - 1, r1), c0: Math.max(0, c0), c1: Math.min(NC - 1, c1) };
}
function dateOf(yyyymmdd, plus) {
  const y = +yyyymmdd.slice(0, 4), m = +yyyymmdd.slice(4, 6), d = +yyyymmdd.slice(6, 8);
  const t = new Date(Date.UTC(y, m - 1, d + plus));
  return t.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
}

// marching squares on the land-sea mask (0.5 contour), cell centres as nodes
function coastline(mask) {
  const segs = [];
  const v = (r, c) => mask[r * NC + c];
  const lerp = (a, b) => (0.5 - a) / (b - a);
  for (let r = 0; r < NR - 1; r++) for (let c = 0; c < NC - 1; c++) {
    const a = v(r, c), b = v(r, c + 1), d = v(r + 1, c), e = v(r + 1, c + 1);   // a=SW b=SE d=NW e=NE
    const idx = (a >= .5 ? 1 : 0) | (b >= .5 ? 2 : 0) | (e >= .5 ? 4 : 0) | (d >= .5 ? 8 : 0);
    if (idx === 0 || idx === 15) continue;
    const S_ = [c + lerp(a, b), r], E_ = [c + 1, r + lerp(b, e)], N_ = [c + lerp(d, e), r + 1], W_ = [c, r + lerp(a, d)];
    const add = (p, q) => segs.push([p[0], p[1], q[0], q[1]]);
    switch (idx) {
      case 1: case 14: add(W_, S_); break;
      case 2: case 13: add(S_, E_); break;
      case 3: case 12: add(W_, E_); break;
      case 4: case 11: add(E_, N_); break;
      case 6: case 9: add(S_, N_); break;
      case 7: case 8: add(W_, N_); break;
      case 5: add(W_, N_); add(S_, E_); break;
      case 10: add(W_, S_); add(E_, N_); break;
    }
  }
  return segs;
}

// ---------------------------------------------------------------- API
async function apiHealth() {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), 3000);
  try {
    const r = await fetch(S.base + '/health', { method: 'POST', signal: ctl.signal });
    const j = await r.json();
    const ok = r.ok && j.status === 'ready';
    setOnline(ok, j.model_name || '');
  } catch (e) {
    setOnline(false, '');
  } finally { clearTimeout(t); }
}
function setOnline(ok, name) {
  const was = S.online;
  S.online = ok;
  if (name) S.modelName = name;
  if (ok && !was && S.scen && !S.busy && (!S.prime || S.primeSteps !== S.steps)) runPrime();
  renderStatus();
  renderAll();
}
async function simulate(pert) {
  const body = { init_states: S.scen.init_states, doy_sincos: S.scen.doy_sincos, steps: S.steps,
    perturbation: pert ? nestPert(pert) : null };
  const t0 = performance.now();
  let r;
  try {
    r = await fetch(S.base + '/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch (e) {
    setOnline(false, '');
    throw new Error('server offline — could not reach ' + S.base + '/simulate');
  }
  let j;
  try { j = await r.json(); } catch (e) { throw new Error('HTTP ' + r.status + ' (non-JSON response)'); }
  if (!r.ok) throw new Error('HTTP ' + r.status + ': ' + (j.error || j.detail || JSON.stringify(j)));
  const flat = flatten(j.rollout, [S.steps, NCH, NR, NC]);
  S.lastMs = +j.inference_ms; S.lastRtt = performance.now() - t0;
  if (!S.online) setOnline(true, '');
  return flat;
}

// ---------------------------------------------------------------- shock
function buildPerturbation() {
  const p = new Float32Array(NCH * NCELL);
  const bi = boxIdx(S.shock.box);
  if (!bi) return p;
  const last = S.scen.init;     // Float32Array [3][6][NCELL]
  for (const comp of S.shock.comps) {
    const c = comp.ch;
    for (let r = bi.r0; r <= bi.r1; r++) for (let col = bi.c0; col <= bi.c1; col++) {
      const cell = r * NC + col;
      const base = last[(2 * NCH + c) * NCELL + cell];
      let d = comp.mode === 'pct' ? base * comp.val / 100 : comp.val * CH[c].addFactor;
      const cur = base + p[c * NCELL + cell];
      if (CH[c].nonneg && cur + d < 0) d = -cur;   // keep rain / moisture physical (>= 0)
      p[c * NCELL + cell] += d;
    }
  }
  return p;
}
function compText(comp) {
  const c = CH[comp.ch];
  return comp.mode === 'pct' ? `${c.key} ${sgn(comp.val, 0)} %` : `${c.key} ${sgn(comp.val, comp.val % 1 ? 1 : 0)} ${c.addUnit}`;
}
function boxText(b) { return b ? `lat ${b.s}–${b.n}°N, lon ${b.w}–${b.e}°E` : 'none'; }

// ---------------------------------------------------------------- runs
async function runPrime() {
  if (!S.scen) return;
  S.busy = true; S.busyMsg = 'Running Timeline Prime…'; updateButtons(); renderAll();
  try {
    S.prime = await simulate(null);
    S.primeSteps = S.steps;
    setRunMsg(`Prime: ${S.steps} days in ${fmt(S.lastMs, 0)} ms (server), ${fmt(S.lastRtt, 0)} ms round trip`, 'ok');
  } catch (e) {
    S.prime = null;
    setRunMsg('Prime run failed: ' + e.message, 'err');
  } finally {
    S.busy = false; S.version++; updateButtons(); renderStatus(); renderAll();
  }
}
async function runMultiverse() {
  if (!S.scen || S.busy) return;
  if (!S.shock.box) { setRunMsg('Pick a preset or drag a box on a map first.', 'err'); return; }
  if (!S.prime || S.primeSteps !== S.steps) { await runPrime(); if (!S.prime) return; }
  const pert = buildPerturbation();
  S.busy = true; S.busyMsg = 'Running Multiverse…'; updateButtons(); renderAll();
  try {
    S.multi = await simulate(pert);
    S.multiSteps = S.steps;
    S.multiBox = { ...S.shock.box };
    S.multiDesc = (S.shock.comps.length ? S.shock.comps.map(compText).join(', ') : 'zero shock (all-zero perturbation)') + ' · ' + boxText(S.shock.box);
    let mx = 0; for (let i = 0; i < S.multi.length; i++) { const d = Math.abs(S.multi[i] - S.prime[i]); if (d > mx) mx = d; }
    S.multiMaxAll = mx;
    computeSpread();
    setRunMsg(`Multiverse: ${fmt(S.lastMs, 0)} ms (server). max |Δ| over all channels & days = ${mx === 0 ? '0 (exactly)' : mx.toPrecision(3) + ' (physical units)'}`, 'ok');
  } catch (e) {
    S.multi = null; S.spread = null;
    setRunMsg('Multiverse run failed: ' + e.message, 'err');
  } finally {
    S.busy = false; S.version++; updateButtons(); renderStatus(); renderAll();
  }
}
function invalidateRuns() { S.prime = null; S.multi = null; S.multiBox = null; S.spread = null; S.version++; }

// ---------------------------------------------------------------- colour ranges
function percentile(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))]; }
function valueRange(c) {
  const key = `${c}|${S.version}|${S.showObs}`;
  if (S.rangeCache.has(key)) return S.rangeCache.get(key);
  const vals = [];
  const push = (arr, days) => { if (!arr) return; for (let d = 0; d < days; d++) { const o = (d * NCH + c) * NCELL; for (let i = 0; i < NCELL; i++) vals.push(arr[o + i]); } };
  push(S.prime, S.primeSteps); push(S.multi, S.multiSteps);
  if (S.showObs && S.scen) push(S.scen.obs, NOBS);
  if (!vals.length && S.scen) { const o = (2 * NCH + c) * NCELL; for (let i = 0; i < NCELL; i++) vals.push(S.scen.init[o + i]); }
  const f = Float64Array.from(vals).sort();
  let lo = percentile(f, 0.01), hi = percentile(f, 0.99);
  if (CH[c].signed) { const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-6); lo = -m; hi = m; }
  if (c === 1) lo = 0;
  if (!(hi > lo)) hi = lo + 1;
  const res = { lo: disp(c, lo), hi: disp(c, hi) };
  if (c === 4) { res.lo = (lo) / 100; res.hi = (hi) / 100; }
  S.rangeCache.set(key, res);
  return res;
}
function deltaRange(c) {
  const key = `d${c}|${S.version}`;
  if (S.rangeCache.has(key)) return S.rangeCache.get(key);
  let m = 0;
  if (S.prime && S.multi) {
    for (let d = 0; d < S.multiSteps; d++) {
      const o = (d * NCH + c) * NCELL;
      for (let i = 0; i < NCELL; i++) { const v = Math.abs(S.multi[o + i] - S.prime[o + i]); if (v > m) m = v; }
    }
  }
  const res = { max: dispD(c, m) };
  S.rangeCache.set(key, res);
  return res;
}

// ---------------------------------------------------------------- map rendering
const panels = [...document.querySelectorAll('.panel')].map((el) => ({
  el, kind: el.dataset.kind, cv: el.querySelector('canvas.map'), lg: el.querySelector('canvas.legend'),
  ov: el.querySelector('.overlay'), geo: null,
}));
const off = document.createElement('canvas'); off.width = NC; off.height = NR;
const offCtx = off.getContext('2d');
const offImg = offCtx.createImageData(NC, NR);

function sizeCanvas(cv, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  cv.style.height = cssH + 'px';
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// field for a panel at the current day; returns Float32Array(NCELL) in display units, or null + message
function panelField(kind, c, day0) {
  if (kind === 'obs') {
    if (!S.scen) return { msg: 'Loading…' };
    if (day0 >= NOBS) return { msg: `No observation beyond day ${NOBS}` };
    return { f: extract(S.scen.obs, day0, c, false) };
  }
  if (kind === 'prime') {
    if (!S.prime) return { msg: S.busy ? S.busyMsg : 'Waiting for the live Prime run…' };
    return { f: extract(S.prime, day0, c, false) };
  }
  if (kind === 'multi') {
    if (!S.multi) return { msg: S.busy ? S.busyMsg : 'Pick a preset or drag a box, then press “Run multiverse”' };
    return { f: extract(S.multi, day0, c, false) };
  }
  if (kind === 'delta') {
    if (!S.multi || !S.prime) return { msg: 'Δ appears after a Multiverse run' };
    const f = new Float32Array(NCELL);
    const o = (day0 * NCH + c) * NCELL;
    for (let i = 0; i < NCELL; i++) f[i] = dispD(c, S.multi[o + i] - S.prime[o + i]);
    return { f };
  }
}
function extract(arr, day0, c) {
  const f = new Float32Array(NCELL);
  const o = (day0 * NCH + c) * NCELL;
  for (let i = 0; i < NCELL; i++) f[i] = c === 4 ? arr[o + i] / 100 : disp(c, arr[o + i]);
  return f;
}

function drawPanel(p) {
  const cssW = p.cv.parentElement.clientWidth;
  const padL = 26, padR = 4, padT = 4, padB = 16;
  const cs = (cssW - padL - padR) / NC;
  const cssH = Math.round(cs * NR + padT + padB);
  const ctx = sizeCanvas(p.cv, cssW, cssH);
  p.geo = { left: padL, top: padT, cs, w: cs * NC, h: cs * NR };
  const g = p.geo;
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(g.left, g.top, g.w, g.h);

  const c = S.ch, day0 = S.day - 1;
  let res, overlay = '', ovClass = '';
  if (S.online === false && p.kind !== 'obs') {          // never show model maps without a live server
    res = {}; overlay = 'SERVER OFFLINE<br>no live model output'; ovClass = 'offline';
  } else {
    res = panelField(p.kind, c, day0);
    if (res.msg) { overlay = res.msg; ovClass = S.busy && /Running/.test(res.msg) ? 'busy' : ''; }
  }
  p.ov.innerHTML = overlay; p.ov.className = 'overlay' + (overlay ? ' show ' + ovClass : '');

  let lo, hi, lut;
  if (p.kind === 'delta') { const m = deltaRange(c).max; lo = -m; hi = m; lut = CMAPS.delta; }
  else { const r = valueRange(c); lo = r.lo; hi = r.hi; lut = CMAPS[CH[c].cmap]; }
  p.scale = { lo, hi, lut };

  if (res.f) {
    const data = offImg.data;
    for (let r = 0; r < NR; r++) for (let col = 0; col < NC; col++) {
      const v = res.f[r * NC + col];
      const k = (p.kind === 'delta' && hi === 0) ? 128 : cmapIndex(v, lo, hi);
      const q = ((NR - 1 - r) * NC + col) * 4;
      data[q] = lut[k * 3]; data[q + 1] = lut[k * 3 + 1]; data[q + 2] = lut[k * 3 + 2]; data[q + 3] = 255;
    }
    offCtx.putImageData(offImg, 0, 0);
    ctx.imageSmoothingEnabled = S.smooth;
    ctx.drawImage(off, g.left, g.top, g.w, g.h);
  }
  p.field = res.f || null;

  // graticule
  ctx.font = '10px "Segoe UI", sans-serif'; ctx.fillStyle = '#8d99aa'; ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = 1;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const lat of [10, 20, 30]) {
    const y = g.top + (LAT_N_EDGE - lat) * cs;
    ctx.beginPath(); ctx.moveTo(g.left, y); ctx.lineTo(g.left + g.w, y); ctx.stroke();
    ctx.fillText(lat + '°N', g.left - 3, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const lon of [70, 80, 90, 100]) {
    const x = g.left + (lon - LON_W_EDGE) * cs;
    ctx.beginPath(); ctx.moveTo(x, g.top); ctx.lineTo(x, g.top + g.h); ctx.stroke();
    ctx.fillText(lon + '°E', x, g.top + g.h + 2);
  }

  // coastline
  const X = (fc) => g.left + (fc + 0.5) * cs, Y = (fr) => g.top + (NR - 0.5 - fr) * cs;
  const strokeCoast = (style, wdt) => {
    ctx.strokeStyle = style; ctx.lineWidth = wdt; ctx.lineCap = 'round'; ctx.beginPath();
    for (const s of S.coast) { ctx.moveTo(X(s[0]), Y(s[1])); ctx.lineTo(X(s[2]), Y(s[3])); }
    ctx.stroke();
  };
  strokeCoast('rgba(0,0,0,0.65)', 3); strokeCoast('rgba(255,255,255,0.9)', 1.2);

  // wind arrows
  if (S.arrows && p.kind !== 'delta') drawArrows(ctx, p, g, day0);

  // regions
  const rectDeg = (b) => [g.left + (b.w - LON_W_EDGE) * cs, g.top + (LAT_N_EDGE - b.n) * cs, (b.e - b.w) * cs, (b.n - b.s) * cs];
  if (S.regions) {
    ctx.setLineDash([2, 3]); ctx.strokeStyle = 'rgba(160,255,190,0.9)'; ctx.lineWidth = 1.2;
    ctx.fillStyle = 'rgba(160,255,190,0.95)'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '10px "Segoe UI", sans-serif';
    for (const R of REGIONS) { const [x, y, w, h] = rectDeg(R); ctx.strokeRect(x, y, w, h); ctx.fillText(R.name, x + 2, y + 2); }
    ctx.setLineDash([]);
  }

  // reach ring on Δ panel
  if (p.kind === 'delta' && S.spread && S.multiBox && res.f) {
    const reach = S.spread.reach[day0];
    if (reach != null && reach > 0) {
      const b = S.multiBox;
      const [x, y, w, h] = rectDeg({ s: b.s - reach, n: b.n + reach, w: b.w - reach, e: b.e + reach });
      ctx.save(); ctx.beginPath(); ctx.rect(g.left, g.top, g.w, g.h); ctx.clip();
      ctx.setLineDash([1.5, 3]); ctx.strokeStyle = '#111'; ctx.lineWidth = 1.6; ctx.strokeRect(x, y, w, h);
      ctx.restore(); ctx.setLineDash([]);
    }
  }

  // shock box (current painter box; dashed) — on all panels
  const box = S.drag ? S.drag.box : S.shock.box;
  if (box) {
    const [x, y, w, h] = rectDeg(box);
    ctx.setLineDash([5, 3]); ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(0,0,0,0.7)'; ctx.strokeRect(x, y, w, h);
    ctx.strokeStyle = '#f5b83d'; ctx.lineWidth = 1.4; ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }

  // hover marker
  if (S.hover) {
    const x = g.left + S.hover.col * cs, y = g.top + (NR - 1 - S.hover.r) * cs;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.strokeRect(x, y, cs, cs);
  }
  drawLegend(p);
}

function drawArrows(ctx, p, g, day0) {
  let src = null;
  if (p.kind === 'obs') src = day0 < NOBS && S.scen ? S.scen.obs : null;
  else if (p.kind === 'prime') src = S.prime; else if (p.kind === 'multi') src = S.multi;
  if (!src || S.online === false && p.kind !== 'obs') return;
  const step = 3, cs = g.cs, scale = cs * 2.4 / 10;     // 10 m/s = 2.4 cells
  ctx.lineWidth = 1.2;
  for (let r = 1; r < NR; r += step) for (let col = 1; col < NC; col += step) {
    const cell = r * NC + col;
    const u = at(src, day0, 2, cell), v = at(src, day0, 3, cell);
    const sp = Math.hypot(u, v); if (sp < 0.3) continue;
    const L = Math.min(sp * scale, cs * 3.2), ux = u / sp, uy = -v / sp;
    const x0 = g.left + (col + 0.5) * cs - ux * L / 2, y0 = g.top + (NR - 0.5 - r) * cs - uy * L / 2;
    const x1 = x0 + ux * L, y1 = y0 + uy * L, hl = Math.min(5, L * 0.45);
    for (const [st, w] of [['rgba(0,0,0,0.7)', 2.6], ['#ffffff', 1.1]]) {
      ctx.strokeStyle = st; ctx.lineWidth = w; ctx.beginPath();
      ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - hl * (ux * 0.85 - uy * 0.5), y1 - hl * (uy * 0.85 + ux * 0.5));
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - hl * (ux * 0.85 + uy * 0.5), y1 - hl * (uy * 0.85 - ux * 0.5));
      ctx.stroke();
    }
  }
}

function drawLegend(p) {
  const cssW = p.lg.clientWidth || p.cv.clientWidth;
  const ctx = sizeCanvas(p.lg, cssW, 34);
  ctx.clearRect(0, 0, cssW, 34);
  const { lo, hi, lut } = p.scale;
  const x0 = 26, w = cssW - 30, y0 = 2, h = 10;
  for (let i = 0; i < w; i++) {
    const k = Math.round(i / (w - 1) * 255);
    ctx.fillStyle = `rgb(${lut[k * 3]},${lut[k * 3 + 1]},${lut[k * 3 + 2]})`;
    ctx.fillRect(x0 + i, y0, 1.5, h);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.strokeRect(x0, y0, w, h);
  const c = S.ch, isD = p.kind === 'delta';
  const unit = isD ? CH[c].dUnit : CH[c].unit;
  const span = hi - lo;
  const dec = span === 0 ? 1 : span < 0.1 ? 3 : span < 1 ? 2 : span < 20 ? 1 : 0;
  ctx.fillStyle = '#c3ccd8'; ctx.font = '10.5px "Segoe UI", sans-serif'; ctx.textBaseline = 'top';
  const lbl = (v) => (isD && v > 0 ? '+' : '') + v.toFixed(dec);
  ctx.textAlign = 'left'; ctx.fillText(lbl(lo), x0, y0 + h + 3);
  ctx.textAlign = 'center'; ctx.fillText(isD ? `0  (${unit})` : `${lbl((lo + hi) / 2)}  ${unit}`, x0 + w / 2, y0 + h + 3);
  ctx.textAlign = 'right'; ctx.fillText(lbl(hi), x0 + w, y0 + h + 3);
}

// ---------------------------------------------------------------- spread analysis
function computeSpread() {
  S.spread = null;
  if (!S.prime || !S.multi || !S.multiBox) return;
  const bi = boxIdx(S.multiBox), N = S.multiSteps;
  const dist = new Int16Array(NCELL);
  for (let r = 0; r < NR; r++) for (let c = 0; c < NC; c++) {
    const dr = Math.max(bi.r0 - r, 0, r - bi.r1), dc = Math.max(bi.c0 - c, 0, c - bi.c1);
    dist[r * NC + c] = Math.max(dr, dc);
  }
  const near = [], far = [], reach = [];
  for (let d = 0; d < N; d++) {
    let sn = 0, nn = 0, sf = 0, nf = 0, rch = -1;
    for (let i = 0; i < NCELL; i++) {
      let e = 0;
      for (let c = 0; c < NCH; c++) { const o = (d * NCH + c) * NCELL + i; const z = (S.multi[o] - S.prime[o]) / STD[c]; e += z * z; }
      if (dist[i] <= NEAR_CELLS) { sn += e; nn++; } else { sf += e; nf++; }
      if (e > ENERGY_TAU && dist[i] > rch) rch = dist[i];
    }
    near.push(nn ? sn / nn : 0); far.push(nf ? sf / nf : 0); reach.push(rch < 0 ? null : rch);
  }
  S.spread = { near, far, reach, N };
}

function drawSpread() {
  const cv = $('spreadCv');
  const cssW = cv.clientWidth, cssH = 210;
  const ctx = sizeCanvas(cv, cssW, cssH);
  ctx.clearRect(0, 0, cssW, cssH);
  const L = 46, R = 40, T = 10, B = 26, w = cssW - L - R, h = cssH - T - B;
  ctx.font = '11px "Segoe UI", sans-serif';
  const sp = S.spread;
  if (!sp) {
    ctx.fillStyle = '#8d99aa'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(S.online === false ? 'Server offline' : 'Run a multiverse to see how the shock spreads', cssW / 2, cssH / 2);
    $('reachRow').innerHTML = '';
    return;
  }
  const N = sp.N;
  const all = sp.near.concat(sp.far).filter((v) => v > 0);
  const FLOOR = 1e-8;
  let top = all.length ? Math.ceil(Math.log10(Math.max(...all))) : 0;
  let bot = all.length ? Math.floor(Math.log10(Math.max(Math.min(...all), FLOOR))) : -1;
  if (top <= bot) top = bot + 1;
  const X = (d) => L + (N === 1 ? w / 2 : (d / (N - 1)) * w);
  const Yl = (v) => { const lv = Math.log10(Math.max(v, Math.pow(10, bot))); return T + h - ((lv - bot) / (top - bot)) * h; };
  const maxReach = Math.max(4, ...sp.reach.map((r) => r || 0));
  const Yr = (r) => T + h - (r / maxReach) * h;

  // axes & grid
  ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.fillStyle = '#8d99aa'; ctx.lineWidth = 1;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  const stepDec = Math.max(1, Math.ceil((top - bot) / 5));
  for (let e = bot; e <= top; e += stepDec) {
    const y = Yl(Math.pow(10, e)); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + w, y); ctx.stroke();
    ctx.fillText('1e' + e, L - 4, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let d = 0; d < N; d++) if (N <= 15 || d % 2 === 0) ctx.fillText('d' + (d + 1), X(d), T + h + 4);
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#a58bff';
  for (let r = 0; r <= maxReach; r += Math.max(1, Math.ceil(maxReach / 4))) ctx.fillText(r + ' c', L + w + 4, Yr(r));

  // reach bars
  const bw = Math.max(3, Math.min(14, w / N * 0.35));
  ctx.fillStyle = 'rgba(165,139,255,0.35)';
  sp.reach.forEach((r, d) => { if (r != null) ctx.fillRect(X(d) - bw / 2, Yr(r), bw, T + h - Yr(r)); });

  // current day
  const cd = S.day - 1;
  if (cd < N) { ctx.strokeStyle = 'rgba(245,184,61,0.6)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(cd), T); ctx.lineTo(X(cd), T + h); ctx.stroke(); ctx.setLineDash([]); }

  const line = (arr, col) => {
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
    arr.forEach((v, d) => (d ? ctx.lineTo(X(d), Yl(v)) : ctx.moveTo(X(d), Yl(v)))); ctx.stroke();
    ctx.fillStyle = col; arr.forEach((v, d) => { ctx.beginPath(); ctx.arc(X(d), Yl(v), 2.6, 0, 7); ctx.fill(); });
  };
  line(sp.near, '#f5b83d'); line(sp.far, '#4fb3ff');

  // legend
  ctx.font = '11px "Segoe UI", sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  const lg = [['#f5b83d', 'near box (≤2 cells)'], ['#4fb3ff', 'far from box'], ['rgba(165,139,255,0.8)', 'reach (cells, right axis)']];
  let lx = L + 6;
  for (const [col, t] of lg) { ctx.fillStyle = col; ctx.fillRect(lx, T + 6, 10, 4); ctx.fillStyle = '#c3ccd8'; ctx.fillText(t, lx + 14, T + 8); lx += ctx.measureText(t).width + 30; }

  const zero = all.length === 0;
  $('reachRow').innerHTML = zero
    ? '<b>Δ = 0 everywhere, every day:</b> Multiverse ≡ Prime (zero shock reproduces Prime exactly).'
    : 'Reach per day (cells): ' + sp.reach.map((r, d) => `d${d + 1} <b>${r == null ? '–' : r}</b>`).join(' · ');
}

// ---------------------------------------------------------------- impact co-pilot
function impactStats(roll, N) {
  const heat = new Float32Array(NCELL), rain = new Float32Array(NCELL), dry = new Float32Array(NCELL);
  for (let d = 0; d < N; d++) for (let i = 0; i < NCELL; i++) {
    if (at(roll, d, 0, i) >= HEAT_K) heat[i]++;
    const tp = Math.max(0, at(roll, d, 1, i));
    rain[i] += tp; if (tp < DRY_MM) dry[i]++;
  }
  return REGIONS.map((R) => {
    const bi = boxIdx(R); let n = 0, h = 0, rn = 0, dy = 0;
    for (let r = bi.r0; r <= bi.r1; r++) for (let c = bi.c0; c <= bi.c1; c++) {
      const i = r * NC + c; if (!S.land[i]) continue;
      n++; h += heat[i]; rn += rain[i]; dy += dry[i];
    }
    return { n, heat: n ? h / n : NaN, rain: n ? rn / n : NaN, dry: n ? dy / n : NaN };
  });
}
function waterRisk(P, M) {
  const pct = P.rain > 0 ? (M.rain - P.rain) / P.rain * 100 : 0;
  const dDry = M.dry - P.dry;
  if (P.rain < 5 && M.rain < 5) return { level: 'LOW', pct, note: 'baseline already dry (<5 mm) — little rain at stake' };
  if (pct <= -25 || dDry >= 3) return { level: 'HIGH', pct };
  if (pct <= -10 || dDry >= 1.5) return { level: 'MODERATE', pct };
  if (pct >= 30 && M.rain >= 50) return { level: 'WET', pct, note: 'excess rain — waterlogging / flash-flood watch' };
  return { level: 'LOW', pct };
}
function heatRisk(P, M) {
  const d = M.heat - P.heat;
  if (M.heat >= 3 && d >= 1) return 'HIGH';
  if (d >= 0.5) return 'MODERATE';
  return 'LOW';
}
function renderImpact() {
  const body = $('impactBody');
  const N = S.primeSteps;
  $('impactFoot').innerHTML =
    `Computed in the browser from the live rollouts over the next ${N || 'N'} days, averaged over land cells (land-sea mask ≥ 0.5) in each approximate region box. ` +
    `<b>Heat-stress day</b> = daily-mean t2m ≥ 34 °C (daily-mean proxy; IMD heatwave criteria use daily maximum temperature). ` +
    `<b>Dry day</b> = rain &lt; 2.5 mm/day (IMD rainy-day threshold). Negative model rain is treated as 0. ` +
    `<b>Water-stress risk</b>: HIGH if rain ≤ −25 % vs Prime or ≥ 3 extra dry days; MODERATE if ≤ −10 % or ≥ 1.5 extra dry days; ` +
    `WET (waterlogging watch) if ≥ +30 % and ≥ 50 mm; LOW otherwise or if both runs are below 5 mm. ` +
    `<b>Heat risk</b>: HIGH if ≥ 3 heat days and ≥ 1 more than Prime; MODERATE if ≥ 0.5 more. Regions: ` +
    REGIONS.map((R) => `${R.name} ${R.s}–${R.n}°N ${R.w}–${R.e}°E`).join('; ') + '.';
  if (S.online === false && !S.prime) { body.innerHTML = '<p class="muted">Server offline — no live rollout to analyse.</p>'; return; }
  if (!S.prime) { body.innerHTML = '<p class="muted">Waiting for the live Prime run…</p>'; return; }
  const P = impactStats(S.prime, N);
  const M = S.multi && S.multiSteps === N ? impactStats(S.multi, N) : null;
  const cmp = (p, m, dec, unit = '') => {
    if (!M) return `${fmt(p, dec)}${unit}`;
    const d = m - p, cls = Math.abs(d) < Math.pow(10, -dec) / 2 ? '' : d > 0 ? 'up' : 'down';
    return `${fmt(p, dec)} <span class="arrow">→</span> <b class="${cls}">${fmt(m, dec)}</b>${unit}`;
  };
  let html = `<table class="imp"><thead><tr><th>Region</th><th>Heat-stress days / cell</th><th>${N}-day rain (mm)</th><th>Δ rain</th><th>Dry days / cell</th><th>Water stress</th><th>Heat</th></tr></thead><tbody>`;
  const adv = [];
  REGIONS.forEach((R, k) => {
    const p = P[k], m = M ? M[k] : null;
    if (!p.n) { html += `<tr><td>${R.name}</td><td colspan="6" class="muted">no land cells</td></tr>`; return; }
    let wr = null, hr = null;
    if (m) { wr = waterRisk(p, m); hr = heatRisk(p, m); }
    html += `<tr><td>${R.name} <span class="muted">(${p.n})</span></td>` +
      `<td class="num">${cmp(p.heat, m && m.heat, 1)}</td>` +
      `<td class="num">${cmp(p.rain, m && m.rain, 0)}</td>` +
      `<td class="num">${m ? `<b class="${wr.pct > 0.5 ? 'up' : wr.pct < -0.5 ? 'down' : ''}">${sgn(wr.pct, 0)} %</b>` : '—'}</td>` +
      `<td class="num">${cmp(p.dry, m && m.dry, 1)}</td>` +
      `<td>${m ? `<span class="risk ${wr.level}">${wr.level}</span>` : '<span class="risk NA">run shock</span>'}</td>` +
      `<td>${m ? `<span class="risk ${hr}">${hr}</span>` : '<span class="risk NA">—</span>'}</td></tr>`;
    if (m) {
      const dryBase = p.rain < 5 && m.rain < 5;
      const rainTxt = dryBase
        ? `Rain ${fmt(p.rain, 1)} → ${fmt(m.rain, 1)} mm over ${R.name} in the next ${N} days → kharif sowing water-stress risk ${wr.level}`
        : `Rain ${sgn(wr.pct, 0)}% vs Prime over ${R.name} in the next ${N} days → kharif sowing water-stress risk ${wr.level === 'WET' ? 'LOW (excess rain)' : wr.level}`;
      const heatTxt = `heat-stress days ${fmt(p.heat, 1)} → ${fmt(m.heat, 1)} per land cell (heat risk ${hr})`;
      const worst = wr.level === 'HIGH' || hr === 'HIGH' ? 'HIGH' : wr.level === 'MODERATE' || hr === 'MODERATE' ? 'MODERATE' : wr.level === 'WET' ? 'WET' : 'LOW';
      adv.push(`<li class="${worst}" title="Rules in the footnote below">${rainTxt}${wr.note ? ' (' + wr.note + ')' : ''}; ${heatTxt}.</li>`);
    }
  });
  html += '</tbody></table>';
  if (adv.length) html += `<ul class="advis">${adv.join('')}</ul>`;
  else html += '<p class="muted" style="margin:8px 0 0">Prime baseline shown. Apply a shock and press “Run multiverse” for the advisory.</p>';
  body.innerHTML = html;
}

// ---------------------------------------------------------------- UI rendering
function renderStatus() {
  const dot = $('apiDot');
  dot.className = 'dot' + (S.online === true ? ' ok' : S.online === false ? ' bad' : '');
  $('apiText').textContent = S.online === true ? 'API ready' : S.online === false ? 'SERVER OFFLINE' : 'checking…';
  $('apiText').style.color = S.online === false ? 'var(--bad)' : '';
  $('modelName').textContent = S.modelName || '—';
  $('infMs').textContent = S.lastMs != null ? fmt(S.lastMs, 0) + ' ms' : '—';
  $('stepsTxt').textContent = S.primeSteps || S.steps;
  $('offlineBanner').hidden = S.online !== false;
  $('offlineUrl').textContent = S.base;
}
function updateButtons() {
  $('runBtn').disabled = S.busy || !S.scen || S.online === false;
  $('runBtn').textContent = S.busy ? (S.busyMsg || 'Running…') : 'Run multiverse ▶';
}
function setRunMsg(t, cls) { const el = $('runMsg'); el.textContent = t; el.className = 'runmsg ' + (cls || ''); }

function renderSidebar() {
  $('boxTxt').textContent = boxText(S.shock.box);
  const ul = $('compList');
  ul.innerHTML = '';
  if (!S.shock.comps.length) {
    const li = document.createElement('li'); li.className = 'empty';
    li.textContent = S.shock.box ? 'no components → all-zero perturbation' : 'no shock components'; ul.appendChild(li);
  }
  S.shock.comps.forEach((comp, k) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${compText(comp)}</span>`;
    const b = document.createElement('button'); b.textContent = '✕'; b.title = 'Remove';
    b.onclick = () => { S.shock.comps.splice(k, 1); S.shock.preset = null; renderSidebar(); };
    li.appendChild(b); ul.appendChild(li);
  });
  document.querySelectorAll('[data-preset]').forEach((b) => b.classList.toggle('active', b.dataset.preset === S.shock.preset));
  // perturbation stats
  const bi = boxIdx(S.shock.box);
  if (bi && S.scen) {
    const p = buildPerturbation();
    const n = (bi.r1 - bi.r0 + 1) * (bi.c1 - bi.c0 + 1);
    const parts = [];
    for (let c = 0; c < NCH; c++) {
      let mx = 0, sum = 0;
      for (let i = 0; i < NCELL; i++) { const v = p[c * NCELL + i]; sum += v; if (Math.abs(v) > Math.abs(mx)) mx = v; }
      if (mx !== 0) parts.push(`${CH[c].key} mean ${sgn(dispD(c, sum / n), 2)} ${CH[c].dUnit}`);
    }
    $('pertStats').textContent = `${n} cells in box. ` + (parts.length ? 'Δ on last input day: ' + parts.join(', ') : 'Perturbation is all zeros.');
  } else $('pertStats').textContent = '';
  $('pUnit').textContent = $('pModeSel').value === 'pct' ? '(%)' : `(${CH[+$('pChSel').value].addUnit})`;
}

function renderAll() {
  if (!S.scen) return;
  for (const p of panels) if (!p.el.hidden) drawPanel(p);
  drawSpread();
  renderImpact();
  const N = S.primeSteps || S.steps;
  $('dayTxt').textContent = S.day;
  $('dateTxt').textContent = `Forecast day ${S.day} of ${N} → valid ${dateOf(S.scen.last_input_day, S.day)} (start state ends ${dateOf(S.scen.last_input_day, 0)})`;
  const ch = CH[S.ch];
  $('subPrime').textContent = `unshocked rollout · ${ch.key} (${ch.unit})`;
  $('subObs').textContent = `what really happened · ${ch.key}`;
  $('subMulti').textContent = S.multi ? S.multiDesc : 'shocked rollout';
  $('subMulti').title = S.multi ? S.multiDesc : '';
  if (S.multi && S.prime) {
    const m = deltaRange(S.ch).max;
    $('subDelta').textContent = S.multiMaxAll === 0 ? 'max |Δ| = 0 exactly — Multiverse ≡ Prime' : `max |Δ| ${ch.key} (all days) = ${m.toPrecision(3)} ${ch.dUnit}`;
  } else $('subDelta').textContent = 'diverging, symmetric, white = 0';
  renderReadout();
}

function renderReadout() {
  const el = $('readout');
  if (!S.hover) { el.innerHTML = 'Hover a map to read values. Drag on a map to paint a shock box (snaps to 1° cells).'; return; }
  const { r, col } = S.hover, i = r * NC + col, ch = CH[S.ch];
  const parts = [`<b>${(LAT0 + r).toFixed(1)}°N ${(LON0 + col).toFixed(1)}°E</b> ${S.land[i] ? 'land' : 'sea'} · ${ch.key}`];
  for (const p of panels) {
    if (p.el.hidden || !p.field) continue;
    const name = { obs: 'Observed', prime: 'Prime', multi: 'Multiverse', delta: 'Δ' }[p.kind];
    const v = p.field[i];
    parts.push(`${name} <b>${p.kind === 'delta' && v > 0 ? '+' : ''}${v.toFixed(Math.abs(v) < 1 && v !== 0 ? 3 : 2)}</b>`);
  }
  el.innerHTML = parts.join(' &nbsp;·&nbsp; ') + ` ${ch.unit}`;
}

// ---------------------------------------------------------------- interaction
function cellFromEvent(p, ev) {
  if (!p.geo) return null;
  const rect = p.cv.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top, g = p.geo;
  const fc = (x - g.left) / g.cs, fr = (y - g.top) / g.cs;    // fr: rows from the top
  return { lon: LON_W_EDGE + fc, lat: LAT_N_EDGE - fr, inside: fc >= 0 && fc < NC && fr >= 0 && fr < NR,
    col: Math.floor(fc), r: NR - 1 - Math.floor(fr) };
}
function snapBox(a, b) {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  let w = Math.floor(Math.min(a.lon, b.lon)), e = Math.ceil(Math.max(a.lon, b.lon));
  let s = Math.floor(Math.min(a.lat, b.lat)), n = Math.ceil(Math.max(a.lat, b.lat));
  w = clamp(w, 65, 100); e = clamp(e, w + 1, 101); s = clamp(s, 6, 37); n = clamp(n, s + 1, 38);
  return { s, n, w, e };
}
for (const p of panels) {
  p.cv.addEventListener('mousedown', (ev) => {
    const c = cellFromEvent(p, ev); if (!c || !c.inside) return;
    ev.preventDefault();
    S.drag = { a: c, box: snapBox(c, c) };
    renderAll();
  });
  p.cv.addEventListener('mousemove', (ev) => {
    const c = cellFromEvent(p, ev);
    const h = c && c.inside ? { r: c.r, col: c.col } : null;
    const changed = JSON.stringify(h) !== JSON.stringify(S.hover);
    S.hover = h;
    if (S.drag && c) { S.drag.box = snapBox(S.drag.a, c); renderAll(); }
    else if (changed) { for (const q of panels) if (!q.el.hidden) drawPanel(q); renderReadout(); }
  });
  p.cv.addEventListener('mouseleave', () => { if (!S.drag) { S.hover = null; for (const q of panels) if (!q.el.hidden) drawPanel(q); renderReadout(); } });
}
window.addEventListener('mouseup', () => {
  if (!S.drag) return;
  S.shock.box = S.drag.box; S.shock.preset = null; S.drag = null;
  setRunMsg('Custom box set. Add components (or keep a preset’s) and press “Run multiverse”.', '');
  renderSidebar(); renderAll();
});

function applyPreset(k) {
  const pr = PRESETS[k];
  S.shock = { box: { ...pr.box }, comps: pr.comps.map((c) => ({ ...c })), preset: k };
  renderSidebar(); renderAll();
  setRunMsg(`Preset “${pr.label}” loaded. Press “Run multiverse”.`, '');
}

function setDay(d) {
  const N = S.primeSteps || S.steps;
  S.day = Math.max(1, Math.min(N, d));
  $('daySl').value = S.day;
  renderAll();
}
function togglePlay() {
  if (S.timer) { clearInterval(S.timer); S.timer = null; $('playBtn').textContent = '▶'; return; }
  const N = S.primeSteps || S.steps;
  if (S.day >= N) setDay(1);
  $('playBtn').textContent = '❚❚';
  S.timer = setInterval(() => {
    const NN = S.primeSteps || S.steps;
    if (S.day >= NN) { togglePlay(); return; }
    setDay(S.day + 1);
  }, 650);
}

async function loadScenario(key) {
  setRunMsg('Loading scenario…', '');
  const j = await (await fetch(`scenarios/${key}.json`)).json();
  S.scen = {
    key: j.key, label: j.label, last_input_day: j.last_input_day, doy_sincos: j.doy_sincos,
    init_states: j.init_states,                       // nested, sent verbatim to /simulate
    init: flatten(j.init_states, [3, NCH, NR, NC]),
    obs: flatten(j.observed, [NOBS, NCH, NR, NC]),
  };
  $('scenSub').textContent = `${j.label} · start state ends ${dateOf(j.last_input_day, 0)}`;
  invalidateRuns();
  setDay(1);
  renderSidebar();
  renderAll();
  if (S.online !== false) await runPrime();
}

function resize() { renderAll(); }

async function init() {
  document.title = APP_TITLE;
  $('appTitle').textContent = APP_TITLE;
  $('baseUrl').value = S.base;
  CH.forEach((c, k) => {
    $('chSel').add(new Option(`${c.name} (${c.unit})`, k));
    $('pChSel').add(new Option(c.key, k));
  });
  $('chSel').onchange = () => { S.ch = +$('chSel').value; renderAll(); };
  $('pChSel').onchange = $('pModeSel').onchange = () => renderSidebar();
  $('stepsIn').onchange = () => {
    let n = Math.round(+$('stepsIn').value); if (!(n >= 1)) n = 10; n = Math.min(30, n);
    $('stepsIn').value = n;
    if (n === S.steps) return;
    S.steps = n; $('daySl').max = n; invalidateRuns(); setDay(Math.min(S.day, n)); renderStatus();
    if (S.online !== false) runPrime();
  };
  $('obsTog').onchange = () => { S.showObs = $('obsTog').checked; $('panelObs').hidden = !S.showObs; renderAll(); };
  $('arrowTog').onchange = () => { S.arrows = $('arrowTog').checked; renderAll(); };
  $('smoothTog').onchange = () => { S.smooth = $('smoothTog').checked; renderAll(); };
  $('regionTog').onchange = () => { S.regions = $('regionTog').checked; renderAll(); };
  $('daySl').oninput = () => setDay(+$('daySl').value);
  $('playBtn').onclick = togglePlay;
  $('runBtn').onclick = runMultiverse;
  $('addComp').onclick = () => {
    const v = +$('pVal').value;
    if (!Number.isFinite(v) || v === 0) { setRunMsg('Enter a non-zero magnitude.', 'err'); return; }
    if (!S.shock.box) { setRunMsg('Drag a box on a map (or pick a preset) first.', 'err'); return; }
    S.shock.comps.push({ ch: +$('pChSel').value, mode: $('pModeSel').value, val: v });
    S.shock.preset = null; renderSidebar(); renderAll();
  };
  document.querySelectorAll('[data-preset]').forEach((b) => (b.onclick = () => applyPreset(b.dataset.preset)));
  $('baseUrl').onchange = () => {
    S.base = $('baseUrl').value.trim().replace(/\/+$/, '') || DEFAULT_BASE;
    $('baseUrl').value = S.base; localStorage.setItem('mom.base', S.base);
    S.online = null; renderStatus(); apiHealth();
  };
  $('scenSel').onchange = () => loadScenario($('scenSel').value).catch((e) => setRunMsg('Scenario load failed: ' + e.message, 'err'));
  document.addEventListener('keydown', (ev) => {
    if (/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) return;
    if (ev.key === 'ArrowRight') { setDay(S.day + 1); ev.preventDefault(); }
    else if (ev.key === 'ArrowLeft') { setDay(S.day - 1); ev.preventDefault(); }
    else if (ev.key === ' ') { togglePlay(); ev.preventDefault(); }
  });
  window.addEventListener('resize', resize);
  renderStatus(); updateButtons();

  try {
    const [idx, st] = await Promise.all([fetch('scenarios/index.json').then((r) => r.json()), fetch('scenarios/static.json').then((r) => r.json())]);
    S.index = idx;
    S.mask = flatten(st.land_sea_mask, [NR, NC]);
    S.land = Uint8Array.from(S.mask, (v) => (v >= 0.5 ? 1 : 0));
    S.coast = coastline(S.mask);
    idx.forEach((s) => $('scenSel').add(new Option(s.label, s.key)));
  } catch (e) {
    $('scenSub').textContent = 'Could not load scenarios/ — serve this folder with: python -m http.server 5173';
    setRunMsg('Failed to load scenario files: ' + e.message, 'err');
    return;
  }
  await apiHealth();
  setInterval(apiHealth, 5000);
  await loadScenario(S.index[0].key);
  updateButtons();
}

init();
