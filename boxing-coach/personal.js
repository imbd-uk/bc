// "Your model": punches, combos and stance the user teaches the app, classified with k-nearest-neighbours.
// Everything is stored in this browser only (localStorage). Nothing is uploaded.

export const LABELS = ['jab', 'cross', 'lead_hook', 'rear_hook', 'lead_uppercut', 'rear_uppercut'];
export const NAMES = {
  jab: 'Jab', cross: 'Cross', lead_hook: 'Lead hook', rear_hook: 'Rear hook',
  lead_uppercut: 'Lead uppercut', rear_uppercut: 'Rear uppercut',
};
// Which hand throws each punch.
export const ROLE = {
  jab: 'lead', cross: 'rear', lead_hook: 'lead', rear_hook: 'rear', lead_uppercut: 'lead', rear_uppercut: 'rear',
};
export const TARGET = 10;        // examples per punch for a full teach
export const MIN_PER_CLASS = 5;  // needed before "your model" can be used
export const COMBO_TARGET = 5;   // reps per combo for a full teach
export const COMBOS = [
  { id: '1-2', name: 'Jab, cross', seq: ['jab', 'cross'] },
  { id: '1-1-2', name: 'Double jab, cross', seq: ['jab', 'jab', 'cross'] },
  { id: '1-2-3', name: 'Jab, cross, lead hook', seq: ['jab', 'cross', 'lead_hook'] },
  { id: '1-6-3', name: 'Jab, rear uppercut, lead hook', seq: ['jab', 'rear_uppercut', 'lead_hook'] },
  { id: '2-3-2', name: 'Cross, lead hook, cross', seq: ['cross', 'lead_hook', 'cross'] },
  { id: '5-4', name: 'Lead uppercut, rear hook', seq: ['lead_uppercut', 'rear_hook'] },
];
const HAND_PENALTY = 1.6;        // distance multiplier when the punching hand doesn't match

// Turn a starter-model or older label ('hook' / 'uppercut') into the lead/rear version.
export function sidedLabel(label, role) {
  if (label !== 'hook' && label !== 'uppercut') return label;
  return `${role === 'rear' ? 'rear' : 'lead'}_${label}`;
}
const KEY = 'bc.personal.v1';

// ---------------------------------------------------------------- storage
export function loadModel() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return emptyModel();
    const m = JSON.parse(raw);
    if (m.v !== 1 || !Array.isArray(m.examples)) return emptyModel();
    const combos = {};
    for (const c of COMBOS) {
      const x = m.combos && m.combos[c.id];
      if (x && x.reps > 0) combos[c.id] = { reps: x.reps | 0, gaps: Array.isArray(x.gaps) ? x.gaps.filter(Number.isFinite) : [] };
    }
    const lead = m.stance && (m.stance.lead === 'left' || m.stance.lead === 'right') ? m.stance.lead : null;
    // Older models had one 'hook' and one 'uppercut'. Split them by the hand that threw them;
    // examples with no hand recorded can't be split and are dropped (teach those punches again).
    const examples = [];
    for (const e of m.examples) {
      let l = e.l;
      if (l === 'hook' || l === 'uppercut') {
        const role = roleOf(e.h, lead);
        if (!role) continue;
        l = sidedLabel(l, role);
      }
      if (!LABELS.includes(l)) continue;
      const c = e.c && COMBOS.some((cb) => cb.id === e.c) ? e.c : null;
      examples.push({ l, f: Float32Array.from(e.f), s: e.s, h: e.h || null, c });
    }
    return {
      v: 1, updated: m.updated || null,
      examples,
      stance: lead ? { lead, source: m.stance.source === 'manual' ? 'manual' : 'auto' } : null,
      combos,
    };
  } catch { return emptyModel(); }
}

export function saveModel(m) {
  m.updated = Date.now();
  try {
    localStorage.setItem(KEY, JSON.stringify({
      v: 1, updated: m.updated, stance: m.stance, combos: m.combos,
      examples: m.examples.map((e) => {
        const o = { l: e.l, f: Array.from(e.f, (x) => Math.round(x * 1000) / 1000), s: Math.round(e.s * 1e4) / 1e4 };
        if (e.h) o.h = e.h;
        if (e.c) o.c = e.c;
        return o;
      }),
    }));
    return true;
  } catch { return false; }
}

export function emptyModel() { return { v: 1, updated: null, examples: [], stance: null, combos: {} }; }

// singlesOnly: ignore punches that were taught inside a combo.
export function counts(m, singlesOnly = false) {
  const c = Object.fromEntries(LABELS.map((l) => [l, 0]));
  for (const e of m.examples) if (e.l in c && !(singlesOnly && e.c)) c[e.l]++;
  return c;
}

export function comboReps(m) {
  return Object.fromEntries(COMBOS.map((c) => [c.id, m.combos[c.id]?.reps || 0]));
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

// ---------------------------------------------------------------- stance + combos
export function stanceName(lead) { return lead === 'right' ? 'Southpaw' : 'Orthodox'; }

// 'lead' / 'rear' for a punching hand ('left' / 'right'), given which hand leads.
export function roleOf(hand, lead) { return hand && lead ? (hand === lead ? 'lead' : 'rear') : null; }

// Longest pause that still counts as "the same combo", learned from the user's own combos.
export function comboGap(m, fallback) {
  const g = COMBOS.flatMap((c) => m.combos[c.id]?.gaps || []).sort((a, b) => a - b);
  if (g.length < 4) return fallback;
  const p90 = g[Math.min(g.length - 1, Math.floor(g.length * 0.9))];
  return Math.min(1.6, Math.max(0.6, p90 * 1.8));
}

// Which punch usually comes next after `prefix`, from the combos the user has taught.
export function comboPrior(m, prefix) {
  const p = {};
  let tot = 0;
  for (const c of COMBOS) {
    const reps = m.combos[c.id]?.reps || 0;
    if (!reps || c.seq.length <= prefix.length || !prefix.every((x, i) => c.seq[i] === x)) continue;
    const nx = c.seq[prefix.length];
    p[nx] = (p[nx] || 0) + reps; tot += reps;
  }
  if (!tot) return null;
  for (const k in p) p[k] /= tot;
  return p;
}

export function matchCombo(seq) {
  return COMBOS.find((c) => c.seq.length === seq.length && c.seq.every((x, i) => x === seq[i])) || null;
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
  // Every punch belongs to one hand (ROLE): a jab or lead hook is the lead hand, a cross or
  // rear uppercut the rear hand. When we know which hand moved, punches of the other hand are pushed away.
  constructor(examples, k = 3) {
    this.ex = examples.filter((e) => e.f && e.f.length).map((e) => ({ ...e, r: ROLE[e.l] || null }));
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

  // role: 'lead' / 'rear' hand that threw this punch, if known.
  classify(f, skip = -1, role = null) {
    const d = [];
    this.ex.forEach((e, i) => {
      if (i === skip || e.f.length !== f.length) return;
      let dd = dist(f, e.f);
      if (role && e.r && role !== e.r) dd *= HAND_PENALTY;
      d.push([dd, e.l]);
    });
    if (!d.length) return null;
    d.sort((a, b) => a[0] - b[0]);
    const top = d.slice(0, this.k);
    const votes = {};
    let total = 0;
    for (const [dd, l] of top) { const w = 1 / (dd + 1e-3); votes[l] = (votes[l] || 0) + w; total += w; }
    let label = null, best = -1;
    for (const l in votes) if (votes[l] > best) { best = votes[l]; label = l; }
    const probs = Object.fromEntries(LABELS.map((l) => [l, (votes[l] || 0) / total]));
    return { label, confidence: best / total, probs, dist: top[0][0], rejected: top[0][0] > this.rejectDist };
  }
}

// Leave-one-out self test: how often each example is recognised from the others.
export function evaluate(m) {
  const knn = new KNN(m.examples);
  const per = Object.fromEntries(LABELS.map((l) => [l, { correct: 0, total: 0, confused: {} }]));
  let correct = 0, total = 0;
  knn.ex.forEach((e, i) => {
    // Use the hand that was actually seen moving, so the test is as hard as live training.
    const r = knn.classify(e.f, i, roleOf(e.h, m.stance?.lead));
    if (!r || !per[e.l]) return;
    per[e.l].total++; total++;
    if (r.label === e.l) { per[e.l].correct++; correct++; }
    else per[e.l].confused[r.label] = (per[e.l].confused[r.label] || 0) + 1;
  });
  return { accuracy: total ? correct / total : null, total, per };
}
