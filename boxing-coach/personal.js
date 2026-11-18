// "Your model": punch examples the user teaches the app, classified with k-nearest-neighbours.
// Everything is stored in this browser only (localStorage). Nothing is uploaded.

export const LABELS = ['jab', 'cross', 'hook', 'uppercut'];
export const TARGET = 10;        // examples per punch for a full teach
export const MIN_PER_CLASS = 5;  // needed before "your model" can be used
const KEY = 'bc.personal.v1';

// ---------------------------------------------------------------- storage
export function loadModel() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return emptyModel();
    const m = JSON.parse(raw);
    if (m.v !== 1 || !Array.isArray(m.examples)) return emptyModel();
    return {
      v: 1, updated: m.updated || null,
      examples: m.examples.map((e) => ({ l: e.l, f: Float32Array.from(e.f), s: e.s })),
    };
  } catch { return emptyModel(); }
}

export function saveModel(m) {
  m.updated = Date.now();
  try {
    localStorage.setItem(KEY, JSON.stringify({
      v: 1, updated: m.updated,
      examples: m.examples.map((e) => ({ l: e.l, f: Array.from(e.f, (x) => Math.round(x * 1000) / 1000), s: Math.round(e.s * 1e4) / 1e4 })),
    }));
    return true;
  } catch { return false; }
}

export function emptyModel() { return { v: 1, updated: null, examples: [] }; }

export function counts(m) {
  const c = Object.fromEntries(LABELS.map((l) => [l, 0]));
  for (const e of m.examples) if (e.l in c) c[e.l]++;
  return c;
}

export function isReady(m) {
  const c = counts(m);
  return LABELS.every((l) => c[l] >= MIN_PER_CLASS);
}

// Wrist-speed trigger level learned from the user's own punches.
export function speedThreshold(m, fallback) {
  const s = m.examples.map((e) => e.s).filter((x) => x > 0).sort((a, b) => a - b);
  if (s.length < 8) return fallback;
  const median = s[s.length >> 1];
  return Math.min(0.3, Math.max(0.06, median * 0.5));
}

// ---------------------------------------------------------------- features
// window: 25 frames at 30 fps, each { pts(16, px), conf(8), world(24, metres) }.
// Uses every 2nd frame. Per frame: 8 joints in 3D (relative to mid-shoulders,
// in shoulder widths) + 8 joints in 2D (same normalisation) = 40 values.
export function features(win) {
  const out = [];
  for (let k = 0; k < win.length; k += 2) {
    const f = win[k];
    const w = f.world;
    const cx = (w[0] + w[3]) / 2, cy = (w[1] + w[4]) / 2, cz = (w[2] + w[5]) / 2;
    const sw = Math.hypot(w[0] - w[3], w[1] - w[4], w[2] - w[5]) || 1;
    for (let i = 0; i < 8; i++) {
      out.push((w[i * 3] - cx) / sw, (w[i * 3 + 1] - cy) / sw, (w[i * 3 + 2] - cz) / sw);
    }
    const p = f.pts;
    const px = (p[0] + p[2]) / 2, py = (p[1] + p[3]) / 2;
    const ps = Math.hypot(p[0] - p[2], p[1] - p[3]) || 1;
    for (let i = 0; i < 8; i++) out.push((p[i * 2] - px) / ps, (p[i * 2 + 1] - py) / ps);
  }
  // Clamp so one badly tracked joint can't dominate the distance.
  return Float32Array.from(out, (v) => Math.max(-4, Math.min(4, v)));
}

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

// ---------------------------------------------------------------- classifier
export class KNN {
  constructor(examples, k = 3) {
    this.ex = examples.filter((e) => e.f && e.f.length);
    this.k = k;
    // Typical distance between a punch and its nearest other example.
    const nn = this.ex.map((e, i) => {
      let best = Infinity;
      this.ex.forEach((o, j) => { if (j !== i) best = Math.min(best, dist(e.f, o.f)); });
      return best;
    }).filter(Number.isFinite).sort((a, b) => a - b);
    // Only reject movements far outside anything taught (e.g. walking up to the phone).
    this.rejectDist = nn.length >= 8
      ? Math.max(nn[nn.length >> 1] * 4, nn[Math.floor(nn.length * 0.9)] * 1.5)
      : Infinity;
  }

  classify(f, skip = -1) {
    const d = [];
    this.ex.forEach((e, i) => { if (i !== skip && e.f.length === f.length) d.push([dist(f, e.f), e.l]); });
    if (!d.length) return null;
    d.sort((a, b) => a[0] - b[0]);
    const top = d.slice(0, this.k);
    const votes = {};
    let total = 0;
    for (const [dd, l] of top) { const w = 1 / (dd + 1e-3); votes[l] = (votes[l] || 0) + w; total += w; }
    let label = null, best = -1;
    for (const l in votes) if (votes[l] > best) { best = votes[l]; label = l; }
    return { label, confidence: best / total, dist: top[0][0], rejected: top[0][0] > this.rejectDist };
  }
}

// Leave-one-out self test: how often each example is recognised from the others.
export function evaluate(m) {
  const knn = new KNN(m.examples);
  const per = Object.fromEntries(LABELS.map((l) => [l, { correct: 0, total: 0, confused: {} }]));
  let correct = 0, total = 0;
  knn.ex.forEach((e, i) => {
    const r = knn.classify(e.f, i);
    if (!r || !per[e.l]) return;
    per[e.l].total++; total++;
    if (r.label === e.l) { per[e.l].correct++; correct++; }
    else per[e.l].confused[r.label] = (per[e.l].confused[r.label] || 0) + 1;
  });
  return { accuracy: total ? correct / total : null, total, per };
}
