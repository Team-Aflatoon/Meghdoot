// Colour maps as 256-entry RGB lookup tables. No external libraries.
'use strict';

// Zero of the diverging Δ map: the ground (--color-paper-3, dark or light by theme). This literal is only the
// fallback until app.js reads the token and calls setDivergingZero() with its sRGB value.
const DELTA_ZERO = '#08090c';
const CMAP_STOPS = {
  // temperature: perceptual sequential ramp ("plasma"), cool blue-violet -> hot yellow
  temp: ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
  // rain: dark = dry, bright cyan/white = heavy
  rain: ['#151b27', '#17324f', '#1d5d8f', '#2b8fd1', '#49c1ea', '#a5ecff', '#ffffff'],
  // signed wind component (PuOr)
  wind: ['#2d004b', '#542788', '#8073ac', '#b2abd2', '#d8daeb', '#f7f7f7',
         '#fee0b6', '#fdb863', '#e08214', '#b35806', '#7f3b08'],
  // pressure (viridis)
  msl: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
  // moisture: dry brown -> moist blue
  moist: ['#3b2a1a', '#7a5c2e', '#b89a5a', '#d9cf8f', '#9fd39a', '#4fb38f', '#22858f', '#1d5a8a', '#1a3570'],
  // calibrated probabilities (0..100 %), sequential: dark = unlikely, bright = likely
  prain: ['#10151d', '#16283f', '#1b4266', '#1f6290', '#2b86b5', '#52aed3', '#93d4ea', '#dff4fa'],
  pheat: ['#15100d', '#33180f', '#5a2211', '#8a2f10', '#bb4512', '#e06a1f', '#f6a043', '#fde2a6'],
  // difference: blue (negative) - dark ground (zero) - red (positive); brightness grows with |Δ|
  delta: ['#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#173a66', DELTA_ZERO,
          '#5e1622', '#b2182b', '#d6604d', '#f4a582', '#fddbc7'],
  // the same diverging map for the light "paper" theme: zero = the light ground, depth grows with |Δ|
  deltaLight: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', DELTA_ZERO,
               '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
};
const hexRgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
function fillLut(lut, rgb) {
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1);
    const k = Math.min(Math.floor(t), rgb.length - 2);
    const f = t - k;
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = Math.round(rgb[k][j] + (rgb[k + 1][j] - rgb[k][j]) * f);
  }
  return lut;
}
function deltaLut(lut, zero) {
  const light = 0.2126 * zero[0] + 0.7152 * zero[1] + 0.0722 * zero[2] > 128;     // light ground -> light-theme ramp
  const rgb = CMAP_STOPS[light ? 'deltaLight' : 'delta'].map(hexRgb);
  rgb[(rgb.length - 1) / 2] = zero;
  fillLut(lut, rgb);
  // exact ground colour at the centre of the diverging map so zero renders as the panel ground
  for (const i of [127, 128]) for (let j = 0; j < 3; j++) lut[i * 3 + j] = zero[j];
  return lut;
}

const CMAPS = (() => {
  const luts = {};
  for (const [name, s] of Object.entries(CMAP_STOPS)) {
    if (name === 'deltaLight') continue;           // folded into CMAPS.delta by setDivergingZero()
    const lut = new Uint8ClampedArray(256 * 3);
    luts[name] = name === 'delta' ? deltaLut(lut, hexRgb(DELTA_ZERO)) : fillLut(lut, s.map(hexRgb));
  }
  return luts;
})();

// Rebuild the Δ map in place (same array, so references stay valid) with zero = the given [r, g, b].
function setDivergingZero(rgb) {
  if (!rgb || rgb.length < 3 || !rgb.slice(0, 3).every(Number.isFinite)) return;
  deltaLut(CMAPS.delta, [rgb[0], rgb[1], rgb[2]]);
}

// value -> LUT index for a [lo, hi] range
function cmapIndex(v, lo, hi) {
  if (!(hi > lo)) return 128;
  let t = (v - lo) / (hi - lo);
  if (t < 0) t = 0; else if (t > 1) t = 1;
  return Math.round(t * 255);
}
