// Colour maps as 256-entry RGB lookup tables. No external libraries.
'use strict';

const CMAPS = (() => {
  const stops = {
    // perceptual "turbo"-like ramp for temperature
    temp: ['#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4', '#24eca6', '#61fc6c', '#a4fc3b',
           '#d1e834', '#f3c63a', '#fe9b2d', '#f36315', '#d93806', '#b11901', '#7a0403'],
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
    // difference: blue (negative) - white (zero) - red (positive); RdBu reversed
    delta: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#ffffff',
            '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
  };
  const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const luts = {};
  for (const [name, s] of Object.entries(stops)) {
    const rgb = s.map(hex);
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (rgb.length - 1);
      const k = Math.min(Math.floor(t), rgb.length - 2);
      const f = t - k;
      for (let j = 0; j < 3; j++) lut[i * 3 + j] = Math.round(rgb[k][j] + (rgb[k + 1][j] - rgb[k][j]) * f);
    }
    // exact white at the centre of the diverging map so zero renders pure white
    if (name === 'delta') { lut[128 * 3] = lut[128 * 3 + 1] = lut[128 * 3 + 2] = 255; lut[127 * 3] = lut[127 * 3 + 1] = lut[127 * 3 + 2] = 255; }
    luts[name] = lut;
  }
  return luts;
})();

// value -> LUT index for a [lo, hi] range
function cmapIndex(v, lo, hi) {
  if (!(hi > lo)) return 128;
  let t = (v - lo) / (hi - lo);
  if (t < 0) t = 0; else if (t > 1) t = 1;
  return Math.round(t * 255);
}
