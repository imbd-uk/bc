// Who is who in a fight clip: the two fighters and the referee.
// - Pose helpers shared by the breakdown and the film study.
// - Appearance: colours sampled from each person (face, torso, sleeves, hands, trunks).
// - Referee score: a referee wears a shirt with sleeves (torso and arms don't match the face),
//   usually white or light, and has bare hands. Fighters are shirtless with gloves. Comparing
//   the torso to the person's own face works for every skin tone.
// - Tracker: keeps Fighter A, Fighter B and the referee apart by appearance first (trunks,
//   gloves, skin) and position second, so fighters switching sides or a camera cut don't swap them.
// - Knockdown check: body horizontal or low to the canvas.

// MediaPipe landmarks we keep per pose (x px, y px, visibility).
export const KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const SLOT = Object.fromEntries(KEEP.map((mi, i) => [mi, i]));
export const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];
export const COLORS = ['#b9d4ff', '#ffcf70'];
export const REF_COLOR = '#c9ced6';

export const pt = (p, mi) => { const s = SLOT[mi] * 3; return [p[s], p[s + 1], p[s + 2]]; };
export const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
export const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
export function center(p) { const s = mid(pt(p, 11), pt(p, 12)), h = mid(pt(p, 23), pt(p, 24)); return mid(s, h); }
export function torso(p) { return Math.max(1, d2(mid(pt(p, 11), pt(p, 12)), mid(pt(p, 23), pt(p, 24)))); }
export function elbowAngle(p, side) {
  const [s, e, w] = side === 'left' ? [11, 13, 15] : [12, 14, 16];
  const S = pt(p, s), E = pt(p, e), W = pt(p, w);
  const a = [S[0] - E[0], S[1] - E[1]], b = [W[0] - E[0], W[1] - E[1]];
  const c = (a[0] * b[0] + a[1] * b[1]) / ((Math.hypot(...a) * Math.hypot(...b)) || 1);
  return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
}
export function packPose(lm, w, h) {
  const p = new Float32Array(KEEP.length * 3);
  KEEP.forEach((mi, i) => { p[i * 3] = lm[mi].x * w; p[i * 3 + 1] = lm[mi].y * h; p[i * 3 + 2] = lm[mi].visibility ?? 1; });
  return p;
}
// Bounding box of a pose (video px), padded.
export function box(p) {
  const xs = [], ys = [];
  for (const mi of KEEP) { const q = pt(p, mi); if (q[2] > 0.3) { xs.push(q[0]); ys.push(q[1]); } }
  if (!xs.length) return null;
  const L = torso(p);
  return { x0: Math.min(...xs) - 0.25 * L, y0: Math.min(...ys) - 0.45 * L, x1: Math.max(...xs) + 0.25 * L, y1: Math.max(...ys) + 0.15 * L };
}

// ---------------------------------------------------------------- colour
function srgbToLab(r, g, b) {
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const R = lin(r), G = lin(g), B = lin(b);
  const x = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047;
  const y = R * 0.2126 + G * 0.7152 + B * 0.0722;
  const z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}
export const dE = (a, b) => (a && b ? Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) : null);

// Grabs a small copy of each video frame for colour sampling and camera-cut detection.
export class Sampler {
  constructor() {
    this.c = document.createElement('canvas');
    this.ctx = this.c.getContext('2d', { willReadFrequently: true });
    this.img = null; this.prevSig = null;
  }
  // Returns true when this frame looks like a camera cut.
  grab(video) {
    const vw = video.videoWidth || video.width, vh = video.videoHeight || video.height;
    const W = 320, H = Math.max(1, Math.round((320 * vh) / vw));
    if (this.c.width !== W || this.c.height !== H) { this.c.width = W; this.c.height = H; }
    this.s = W / vw;
    this.ctx.drawImage(video, 0, 0, W, H);
    this.img = this.ctx.getImageData(0, 0, W, H);
    // 8x8 grid of brightness: a big jump between frames = a cut
    const sig = new Float32Array(64), d = this.img.data;
    for (let gy = 0; gy < 8; gy++) for (let gx = 0; gx < 8; gx++) {
      const i = (Math.floor((gy + 0.5) * H / 8) * W + Math.floor((gx + 0.5) * W / 8)) * 4;
      sig[gy * 8 + gx] = (d[i] + d[i + 1] + d[i + 2]) / 3;
    }
    let diff = 0;
    if (this.prevSig) for (let i = 0; i < 64; i++) diff += Math.abs(sig[i] - this.prevSig[i]);
    this.prevSig = sig;
    return this.prevSig && diff / 64 > 38;
  }
  // Average Lab colour of a square patch centred at (x, y) in video px.
  patch(x, y, r) {
    const { img, s } = this;
    if (!img) return null;
    const cx = Math.round(x * s), cy = Math.round(y * s), rr = Math.max(1, Math.round(r * s));
    let R = 0, G = 0, B = 0, n = 0;
    for (let yy = cy - rr; yy <= cy + rr; yy += 1) {
      if (yy < 0 || yy >= img.height) continue;
      for (let xx = cx - rr; xx <= cx + rr; xx += 1) {
        if (xx < 0 || xx >= img.width) continue;
        const i = (yy * img.width + xx) * 4;
        R += img.data[i]; G += img.data[i + 1]; B += img.data[i + 2]; n++;
      }
    }
    return n ? srgbToLab(R / n, G / n, B / n) : null;
  }
  // Colours that describe one person.
  appearance(p) {
    const L = torso(p);
    const vis = (mi) => pt(p, mi)[2] > 0.4;
    const sh = mid(pt(p, 11), pt(p, 12)), hp = mid(pt(p, 23), pt(p, 24));
    const avg = (cols) => { const c = cols.filter(Boolean); return c.length ? [0, 1, 2].map((k) => c.reduce((s, x) => s + x[k], 0) / c.length) : null; };
    const face = vis(0) ? this.patch(pt(p, 0)[0], pt(p, 0)[1] + 0.06 * L, 0.07 * L) : null;
    const tor = this.patch(sh[0] + (hp[0] - sh[0]) * 0.4, sh[1] + (hp[1] - sh[1]) * 0.4, 0.14 * L);
    const arm = avg([[11, 13], [12, 14]].filter(([a, b]) => vis(a) && vis(b)).map(([a, b]) => { const m = mid(pt(p, a), pt(p, b)); return this.patch(m[0], m[1], 0.05 * L); }));
    const hand = avg([[13, 15], [14, 16]].filter(([e, w]) => vis(w)).map(([e, w]) => {
      const W = pt(p, w), E = pt(p, e);
      return this.patch(W[0] + (W[0] - E[0]) * 0.2, W[1] + (W[1] - E[1]) * 0.2, 0.07 * L);
    }));
    const trunks = this.patch(hp[0], hp[1] + 0.12 * L, 0.12 * L);
    // colour histogram of the trunks / waist area: the most reliable way to tell two fighters apart
    const hw = Math.max(d2(pt(p, 23), pt(p, 24)) * 0.8, 0.25 * L);
    const th = this.hist(hp[0] - hw, hp[1] - 0.05 * L, hp[0] + hw, hp[1] + 0.35 * L);
    return { face, torso: tor, arm, hand, trunks, th };
  }
  // Normalised colour histogram (3 lightness x 6 x 6 colour bins) of a rectangle in video px.
  hist(x0, y0, x1, y1) {
    const { img, s } = this;
    if (!img) return null;
    const X0 = Math.max(0, Math.floor(x0 * s)), Y0 = Math.max(0, Math.floor(y0 * s));
    const X1 = Math.min(img.width - 1, Math.ceil(x1 * s)), Y1 = Math.min(img.height - 1, Math.ceil(y1 * s));
    if (X1 <= X0 || Y1 <= Y0) return null;
    const h = new Float32Array(108);
    const step = Math.max(1, Math.floor(Math.sqrt(((X1 - X0) * (Y1 - Y0)) / 400)));
    let n = 0;
    for (let y = Y0; y <= Y1; y += step) for (let x = X0; x <= X1; x += step) {
      const i = (y * img.width + x) * 4;
      const [l, a, b] = srgbToLab(img.data[i], img.data[i + 1], img.data[i + 2]);
      const li = l < 35 ? 0 : l < 65 ? 1 : 2;
      const ai = Math.min(5, Math.max(0, Math.floor((a + 30) / 10)));
      const bi = Math.min(5, Math.max(0, Math.floor((b + 30) / 13)));
      h[li * 36 + ai * 6 + bi]++; n++;
    }
    if (!n) return null;
    for (let k = 0; k < 108; k++) h[k] /= n;
    return h;
  }
}

// Hellinger distance between two histograms: 0 = identical, 1 = nothing in common.
function histDist(a, b) {
  if (!a || !b) return null;
  let bc = 0;
  for (let k = 0; k < a.length; k++) bc += Math.sqrt(a[k] * b[k]);
  return Math.sqrt(Math.max(0, 1 - bc));
}

// Someone lying (or falling) on the canvas: body far from upright. Their colour samples land on the
// canvas and shorts, so their look and "referee score" can't be trusted, and a referee never lies down.
// Stance in one frame, facing the opponent: +1 left side forward (orthodox), -1 right side forward
// (southpaw), 0 can't tell. Shoulders, hips, knees and feet vote together.
export function stanceVote(p, opp) {
  const dir = Math.sign(center(opp)[0] - center(p)[0]) || 1;
  let v = 0;
  for (const [a, b] of [[11, 12], [23, 24], [25, 26], [27, 28]]) {
    const P = pt(p, a), Q = pt(p, b);
    if (P[2] > 0.5 && Q[2] > 0.5) v += Math.sign((P[0] - Q[0]) * dir);
  }
  return Math.sign(v);
}

export function lying(p) {
  const sh = mid(pt(p, 11), pt(p, 12)), hp = mid(pt(p, 23), pt(p, 24));
  return Math.atan2(Math.abs(sh[0] - hp[0]), hp[1] - sh[1]) * 180 / Math.PI > 55;
}

// The pose model sometimes returns the same person twice: keep the clearer copy.
export function dedupe(dets, apps) {
  const keep = dets.map(() => true);
  const vis = (p) => [0, 11, 12, 23, 24].reduce((s, mi) => s + pt(p, mi)[2], 0);
  for (let i = 0; i < dets.length; i++) for (let j = i + 1; j < dets.length; j++) {
    if (!keep[i] || !keep[j]) continue;
    const L = Math.min(torso(dets[i]), torso(dets[j]));
    if (d2(center(dets[i]), center(dets[j])) < 0.4 * L) { if (vis(dets[i]) >= vis(dets[j])) keep[j] = false; else keep[i] = false; }
  }
  return { dets: dets.filter((_, i) => keep[i]), apps: apps.filter((_, i) => keep[i]) };
}

// Is this colour skin? Skin of every tone sits in a warm hue band with some colour in it;
// shirts (white, grey, black, light blue) are greyish or cool. Lab colour in, true/false out.
export function isSkin(lab) {
  if (!lab) return null;
  const [L, a, b] = lab;
  const chroma = Math.hypot(a, b), hue = Math.atan2(b, a) * 180 / Math.PI;
  return L > 12 && L < 92 && chroma >= 7 && hue >= 12 && hue <= 88;
}

// Higher = more like a referee. A fighter's torso and arms are bare skin and his hands are gloves;
// a referee's torso and arms are covered by a shirt (often white) and his hands are bare.
// Deliberately doesn't lean on the face: a fighter with his back turned shows hair, not face.
export function refScore(app) {
  if (!app) return 0;
  let s = 0;
  const tS = isSkin(app.torso), aS = isSkin(app.arm), hS = isSkin(app.hand);
  s += tS == null ? 0.4 : tS ? 0 : 1.2;          // shirt over the torso
  s += aS == null ? 0.2 : aS ? 0 : 0.6;          // sleeves
  const t = app.torso;
  if (t && t[0] > 70 && Math.hypot(t[1], t[2]) < 10) s += 0.5;   // properly white shirt
  if (hS === true) s += 0.3;                     // bare hands
  else if (hS === false && tS) s -= 0.4;         // gloves on a bare-chested person
  return s;
}
export const REF_THRESHOLD = 1.3;

// A person's look: trunks histogram + trunks, glove, skin and face colours.
function sigOf(app) {
  return { th: app?.th || null, trunks: app?.trunks || null, torso: app?.torso || null, hand: app?.hand || null, face: app?.face || null };
}
function oneDist(app, sig) {
  let s = 0, w = 0;
  const h = histDist(app.th, sig.th);
  if (h != null) { s += 1.4 * Math.min(1, h * 1.6); w += 1.4; }
  for (const [k, wt] of [['trunks', 0.6], ['hand', 0.6], ['torso', 0.5], ['face', 0.3]]) {
    const d = dE(app[k], sig[k]);
    if (d != null) { s += wt * Math.min(d, 60) / 30; w += wt; }
  }
  return w ? s / w : 0.5;
}
// Distance to the closest of the person's remembered looks (the "look bank").
export function sigDist(app, bank) {
  if (!app || !bank) return 0.5;
  const list = Array.isArray(bank) ? bank : [bank];
  if (!list.length) return 0.5;
  return Math.min(...list.map((s) => oneDist(app, s)));
}
// Remember a new look when it's confidently this person but different from what we have
// (new camera angle, lighting, side of the ring).
function addLook(bank, app) {
  if (!app) return;
  const d = sigDist(app, bank);
  if (d < 0.18) return;
  bank.push(sigOf(app));
  if (bank.length > 12) bank.splice(2, 1);   // keep the confirmed first looks
}

// ---------------------------------------------------------------- tracker
// Tracks: 0 = Fighter A, 1 = Fighter B, 2 = referee.
export class Tracker {
  constructor(width) { this.width = width; this.gate = width * 0.25; this.tracks = [0, 1, 2].map(() => ({ pos: null, bank: null, seen: -Infinity })); this.t = 0; }

  // Best guess at roles for a set of people: 'A' | 'B' | 'R' | null per detection.
  static guess(dets0, apps0) {
    const { dets, apps } = dedupe(dets0, apps0);
    const map = dets.map((d) => dets0.indexOf(d));
    const r = Tracker.guessRaw(dets, apps);
    const out = new Array(dets0.length).fill(null);
    r.forEach((role, i) => { out[map[i]] = role; });
    return out;
  }

  static guessRaw(dets, apps) {
    const n = dets.length, roles = new Array(n).fill(null);
    if (!n) return roles;
    const scores = apps.map(refScore);
    const size = dets.map(torso);
    const big = Math.max(...size);
    scores.forEach((_, i) => { if (size[i] < 0.45 * big) scores[i] = -9; });   // crowd
    let ref = -1;
    if (n >= 2) {
      const i = scores.indexOf(Math.max(...scores));
      if (scores[i] > REF_THRESHOLD) ref = i;
    }
    if (ref >= 0) roles[ref] = 'R';
    // fighters: the two biggest remaining people
    const rest = dets.map((_, i) => i).filter((i) => i !== ref && size[i] >= 0.45 * big).sort((a, b) => size[b] - size[a]).slice(0, 2);
    rest.sort((a, b) => center(dets[a])[0] - center(dets[b])[0]);
    if (rest[0] != null) roles[rest[0]] = 'A';
    if (rest[1] != null) roles[rest[1]] = 'B';
    return roles;
  }

  init(dets, apps, roles) {
    roles.forEach((r, i) => {
      const k = r === 'A' ? 0 : r === 'B' ? 1 : r === 'R' ? 2 : -1;
      if (k < 0) return;
      this.tracks[k] = { pos: center(dets[i]), bank: [sigOf(apps[i])], seen: 0 };
    });
    this.ready = true;
  }

  // -> [poseA, poseB, poseRef] (null where not seen this frame)
  update(dets0, apps0, cut, t = 0) {
    this.t = t;
    const { dets, apps } = dedupe(dets0, apps0);
    if (!this.ready) {
      if (dets.length < 2) return [null, null, null];
      this.init(dets, apps, Tracker.guess(dets, apps));
    }
    const n = dets.length;
    const cs = dets.map(center), scores = apps.map(refScore);
    // people much smaller than the biggest in shot are in the crowd, not in the ring
    const sizes = dets.map(torso), big = Math.max(0, ...sizes);
    const down = dets.map(lying);
    const cost = (k, i) => {
      if (down[i] && k === 2) return Infinity;              // the referee never lies on the canvas
      if (sizes[i] < 0.45 * big && !down[i]) return Infinity;
      const tr = this.tracks[k];
      const stale = t - tr.seen > (down[i] ? 2.0 : 1.0);
      const pos = cut || !tr.pos || stale ? 0.4 : Math.min(2, d2(cs[i], tr.pos) / this.gate);
      // a fallen fighter: colours are unreliable, so lean on where they went down instead
      if (down[i]) return 0.3 + 0.8 * pos + 0.3 * sigDist(apps[i], tr.bank);
      // the referee may walk into shot later: fill his slot from a clear referee look
      if (k === 2 && !tr.bank) return scores[i] > REF_THRESHOLD ? 0.6 + 0.3 * pos : Infinity;
      const app = sigDist(apps[i], tr.bank);
      const role = k === 2 ? Math.max(0, REF_THRESHOLD + 0.2 - scores[i]) * 0.9 : Math.max(0, scores[i] - 0.6) * 1.4;
      // looking like one of the OTHER people we know is strong evidence it isn't this one
      let other = 0;
      for (const j of [0, 1, 2]) {
        const o = this.tracks[j];
        if (j === k || !o.bank) continue;
        const dOther = sigDist(apps[i], o.bank);
        if (tr.bank && dOther + 0.1 < app) other = Math.max(other, (app - dOther) * 1.2);
      }
      // a slot nobody has filled yet must not grab someone who looks like a person we already know
      let lookalike = 0;
      if (!tr.bank) {
        lookalike = 0.6;   // filling an empty slot needs real evidence
        for (const j of [0, 1, 2]) {
          const o = this.tracks[j];
          if (j === k || !o.bank) continue;
          lookalike = Math.max(lookalike, 0.6 + Math.max(0, 0.7 - sigDist(apps[i], o.bank)) * 3);
          // standing where someone we know was a moment ago: it's them, not a new person
          if (o.pos && t - o.seen < 1.5 && d2(cs[i], o.pos) < 1.2 * sizes[i]) lookalike += 2.5;
        }
      }
      // appearance leads; position only breaks ties (and counts for nothing after a camera cut)
      return (cut ? 0.1 : 0.45) * pos + (cut ? 1.6 : 1.1) * app + other + role + lookalike;
    };
    const MISS = 1.3;
    let best = { c: Infinity, a: [-1, -1, -1] };
    const pick = [-1, -1, -1];
    const rec = (k, used, c) => {
      if (c >= best.c) return;
      if (k === 3) { best = { c, a: pick.slice() }; return; }
      pick[k] = -1; rec(k + 1, used, c + MISS);
      for (let i = 0; i < n; i++) if (!(used & (1 << i))) { pick[k] = i; rec(k + 1, used | (1 << i), c + cost(k, i)); }
    };
    rec(0, 0, 0);
    return best.a.map((i, k) => {
      if (i < 0) return null;
      const tr = this.tracks[k];
      const c = cost(k, i);
      tr.pos = cs[i]; tr.seen = t;
      if (down[i]) return dets[i];                            // don't learn a look from a body lying on the canvas
      if (!tr.bank) tr.bank = [sigOf(apps[i])];
      else if (c < 0.7 && !cut) addLook(tr.bank, apps[i]);   // only learn new looks when we're sure
      return dets[i];
    });
  }
}

// ---------------------------------------------------------------- second pass
// After the whole clip has been read, choose who's who for every frame at once (Viterbi): each
// frame's choice must match each person's look AND join up smoothly with the frames before and
// after it. This removes the one-frame swaps and flickers a frame-by-frame tracker can make.
// raw: [{ t, dets, apps, cut }], banks: [bankA, bankB, bankRef] from the tracker.
export function smoothIdentities(raw, banks) {
  const MISS = [1.3, 1.3, 1.1];
  const frames = raw.map((f) => {
    const { dets, apps } = dedupe(f.dets, f.apps);
    const sizes = dets.map(torso), big = Math.max(0, ...sizes);
    const scores = apps.map(refScore);
    const cs = dets.map(center);
    const down = dets.map(lying);
    // every way of giving (some of) the people in this frame the roles A, B, referee
    const states = [];
    const n = dets.length;
    for (let a = -1; a < n; a++) for (let b = -1; b < n; b++) for (let r = -1; r < n; r++) {
      const s = [a, b, r];
      if ((a >= 0 && (a === b || a === r)) || (b >= 0 && b === r)) continue;
      let c = 0;
      for (let k = 0; k < 3; k++) {
        const i = s[k];
        if (i < 0) { c += MISS[k]; continue; }
        if (down[i]) {
          // someone on the canvas is a fighter, never the referee; their colours can't be trusted
          if (k === 2 || !banks[k] || !banks[k].length) { c = Infinity; break; }
          c += 0.4 + 0.35 * sigDist(apps[i], banks[k]);
          continue;
        }
        if (sizes[i] < 0.45 * big) { c = Infinity; break; }
        if (!banks[k] || !banks[k].length) {
          // the referee can be recognised from his look alone; an unseen fighter slot stays empty
          if (k === 2 && scores[i] > REF_THRESHOLD) { c += 0.5 + Math.max(0, REF_THRESHOLD + 0.6 - scores[i]) * 0.5; continue; }
          c = Infinity; break;
        }
        const role = k === 2 ? Math.max(0, REF_THRESHOLD + 0.2 - scores[i]) * 0.55 : Math.max(0, scores[i] - 0.6) * 1.4;
        // the referee slot must also look like the referee seen so far (shirt, trousers), not just "not a fighter"
        c += (k === 2 ? 1.7 : 1.2) * sigDist(apps[i], banks[k]) + role;
      }
      if (c < Infinity) states.push({ s, c });
    }
    return { t: f.t, cut: f.cut, dets, cs, sizes, states, down };
  });
  // dynamic programming over the clip
  let prev = null;
  for (const f of frames) {
    for (const st of f.states) {
      if (!prev) { st.total = st.c; st.from = -1; continue; }
      let best = Infinity, from = -1;
      prev.states.forEach((ps, j) => {
        let tc = 0;
        for (let k = 0; k < 3; k++) {
          const i = st.s[k], pi = ps.s[k];
          if (i >= 0 && pi >= 0) {
            if (f.cut) continue;
            const L = Math.max(f.sizes[i], prev.sizes[pi]);
            const jump = d2(f.cs[i], prev.cs[pi]) / L;
            // people don't teleport (but a fighter going down moves a long way, fast)
            const slack = f.down[i] || prev.down[pi] ? 0.9 : 0.15;
            tc += 0.9 * Math.min(2.5, Math.max(0, jump - slack) / 0.4);
          } else if ((i >= 0) !== (pi >= 0)) tc += 0.15;
        }
        const v = ps.total + tc;
        if (v < best) { best = v; from = j; }
      });
      st.total = best + st.c; st.from = from;
    }
    prev = f;
  }
  // walk back along the cheapest path
  const out = new Array(frames.length);
  let idx = frames.length ? frames[frames.length - 1].states.reduce((b, s, j, arr) => (s.total < arr[b].total ? j : b), 0) : -1;
  for (let fi = frames.length - 1; fi >= 0; fi--) {
    const f = frames[fi], st = f.states[idx];
    out[fi] = { t: f.t, cut: f.cut, poses: [st.s[0] >= 0 ? f.dets[st.s[0]] : null, st.s[1] >= 0 ? f.dets[st.s[1]] : null], ref: st.s[2] >= 0 ? f.dets[st.s[2]] : null };
    idx = st.from;
  }
  return out;
}

// ---------------------------------------------------------------- knockdowns
// True when a pose looks like a fighter on the canvas: torso far from upright, or the
// shoulders down near the feet. standL = the fighter's usual torso length when standing.
export function isDown(p, standL) {
  const sh = mid(pt(p, 11), pt(p, 12)), hp = mid(pt(p, 23), pt(p, 24));
  if (pt(p, 11)[2] < 0.4 || pt(p, 23)[2] < 0.4) return false;
  const dx = sh[0] - hp[0], dy = hp[1] - sh[1];   // dy > 0 when shoulders are above hips
  const tilt = Math.atan2(Math.abs(dx), dy) * 180 / Math.PI;   // 0 = upright
  const L = d2(sh, hp);
  if (L < 0.55 * standL && tilt < 40) return false;   // squashed, unreliable pose (turned away, cropped)
  if (tilt > 55) return true;
  // shoulders down near the feet: only trust feet the model can really see, below the hips
  // (when the legs are out of shot the model guesses ankles, often in impossible places)
  const ankles = [[25, 27], [26, 28]].map(([k, a]) => [pt(p, k), pt(p, a)])
    .filter(([k, a]) => k[2] > 0.6 && a[2] > 0.6 && a[1] > hp[1] + 0.2 * standL && k[1] > hp[1])
    .map(([, a]) => a);
  if (ankles.length) {
    const floor = Math.max(...ankles.map((q) => q[1]));
    if (floor - sh[1] < 1.15 * standL) return true;
  }
  return false;
}
