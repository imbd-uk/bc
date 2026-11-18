// Turns camera frames into punch events.
// 1. MediaPipe pose → 8 joints (2D pixels + 3D metres) per frame.
// 2. Frames are resampled to an even 30 fps stream (the rate the models expect).
// 3. Two detectors run on that stream:
//    - starter: the original project's LSTM (same rules as the original app)
//    - motion:  finds each punch by its wrist-speed peak and hands back the
//               25-frame window, which "your model" (k-NN) classifies or Teach mode stores.

import { features } from './personal.js';

export const STARTER_LABELS = ['jab', 'cross', 'hook', 'uppercut'];
// MediaPipe indices in the original model's order: L/R shoulder, elbow, wrist, hip.
export const MP_IDX = [11, 12, 13, 14, 15, 16, 23, 24];

const C = {
  SEQ: 25, HZ: 30, REF_SIZE: 640,
  KP_CONF: 0.5,
  STARTER_CONF: 0.7, STARTER_MIN_SPEED: 2.0, STARTER_RECOVERY: 0.9,
  MOTION_SPEED: 0.1,        // shoulder widths per frame (default trigger, before teaching)
  MOTION_LOOKAHEAD: 6,      // frames after the peak before deciding (0.2 s)
  MOTION_REFRACTORY: 0.3,   // s between punches from the same hand
  MOTION_MIN_GAP: 0.12,     // s between punches from different hands (fast 1-2s)
  MAX_GAP: 0.3,             // s without a usable frame before the stream restarts
  STILL: 0.04,              // both wrists slower than this = holding guard
  LEAD_HYST: 0.05,          // stance score needed before the lead hand flips
};
const HANDS = ['left', 'right'];

export class Engine {
  constructor(pose, starter) {
    this.pose = pose;
    this.starter = starter;
    this.useStarter = true;
    this.motionSpeed = C.MOTION_SPEED;
    this.reset();
  }

  reset() {
    this.stream = [];       // resampled frames
    this.speed = [[], []];  // wrist speed per stream frame, [left, right]
    this.reach = [[], []];  // wrist-to-own-shoulder distance per stream frame, [left, right]
    this.last = null;
    this.nextT = null;
    this.starterUntil = -Infinity;
    this.lastMotionT = [-Infinity, -Infinity];
    this.lastAnyT = -Infinity;
    this.lastSeenT = -Infinity;
    this.leadEma = 0; this.leadN = 0; this.lead = null;
    this.guard = null;
  }

  // Returns { landmarks, stance, lead, guard, tracking, events: [...] }
  //   lead:  'left' / 'right' hand forward, read from the guard between punches
  //   guard: this frame's raw stance score while holding still (negative = left leads), else null
  process(source, w, h, t) {
    const res = this.pose.detectForVideo(source, performance.now());
    const lm = res.landmarks && res.landmarks[0];
    const wl = res.worldLandmarks && res.worldLandmarks[0];
    const events = [];
    this.guard = null;
    let stance = null;
    if (lm && wl) {
      const f = toFrame(lm, wl, w, h, t);
      stance = detectStance(f);
      if (normalize2d(f)) {
        this.lastSeenT = t;
        for (const g of this.resample(f)) this.onGridFrame(g, events);
      }
    }
    if (this.lead) stance = this.lead === 'left' ? 'Orthodox' : 'Southpaw';
    return { landmarks: lm ? { lm, w, h } : null, stance, lead: this.lead, guard: this.guard, tracking: t - this.lastSeenT < 1, events };
  }

  // The hand that moved fastest over the last ~0.3 s.
  recentHand() {
    const top = (v) => Math.max(0, ...v.slice(-10));
    const l = top(this.speed[0]), r = top(this.speed[1]);
    return l || r ? HANDS[r > l ? 1 : 0] : null;
  }

  resample(f) {
    const out = [];
    if (!this.last || f.t - this.last.t > C.MAX_GAP || f.t < this.last.t) {
      this.stream = []; this.speed = [[], []]; this.reach = [[], []];
      this.last = f; this.nextT = f.t;
    }
    const a = this.last;
    while (this.nextT <= f.t + 1e-6) {
      const u = f.t > a.t ? (this.nextT - a.t) / (f.t - a.t) : 1;
      out.push(lerpFrame(a, f, Math.min(1, Math.max(0, u)), this.nextT));
      this.nextT += 1 / C.HZ;
    }
    this.last = f;
    return out;
  }

  onGridFrame(g, events) {
    const s = this.stream;
    const prev = s[s.length - 1];
    s.push(g);
    const sp = prev ? wristSpeeds(prev, g) : [0, 0];
    const rc = reaches(g);
    for (const h of [0, 1]) { this.speed[h].push(sp[h]); this.reach[h].push(rc[h]); }
    if (s.length > 60) { s.shift(); for (const h of [0, 1]) { this.speed[h].shift(); this.reach[h].shift(); } }

    // --- stance: while holding guard, which hand (and shoulder) sits closer to the camera
    if (prev && sp[0] < C.STILL && sp[1] < C.STILL) {
      const sc = stanceScore(g);
      if (sc != null) {
        this.guard = sc;
        this.leadEma = this.leadN ? this.leadEma * 0.95 + sc * 0.05 : sc;
        this.leadN++;
        if (this.leadEma < -C.LEAD_HYST) this.lead = 'left';
        else if (this.leadEma > C.LEAD_HYST) this.lead = 'right';
      }
    }

    // --- starter model (original LSTM, original gates)
    if (this.useStarter && s.length >= C.SEQ && g.t >= this.starterUntil) {
      const win = s.slice(-C.SEQ);
      const seq = win.map(normalize2d);
      if (seq.every(Boolean)) {
        const { index, confidence } = this.starter.predict(seq);
        if (confidence > C.STARTER_CONF && originalSpeed(win) > C.STARTER_MIN_SPEED) {
          this.starterUntil = g.t + C.STARTER_RECOVERY;
          events.push({ kind: 'punch', label: STARTER_LABELS[index], confidence, model: 'starter', hand: this.recentHand(), t: g.t });
        }
      }
    }

    // --- motion detector: a wrist-speed peak while that arm is extending.
    // Each hand is tracked on its own, so a fast jab-cross gives two punches, not one.
    const n = s.length - 1;
    const L = C.MOTION_LOOKAHEAD;
    const p = n - L;
    if (s.length >= C.SEQ && p >= 2) {
      const cands = [];
      for (const h of [0, 1]) {
        const v = this.speed[h], r = this.reach[h];
        const sm = (i) => (v[i - 1] + v[i] + v[i + 1]) / 3;
        const pk = sm(p);
        let isPeak = pk >= this.motionSpeed;
        for (let i = Math.max(1, p - L); isPeak && i <= n - 1; i++) if (i !== p && sm(i) > pk) isPeak = false;
        if (isPeak && r[p + 1] - r[p - 2] > 0) cands.push({ h, pk });
      }
      cands.sort((a, b) => b.pk - a.pk);
      const pt = s[p].t;
      for (const { h, pk } of cands) {
        if (pt - this.lastMotionT[h] < C.MOTION_REFRACTORY || pt - this.lastAnyT < C.MOTION_MIN_GAP) continue;
        this.lastMotionT[h] = pt; this.lastAnyT = pt;
        events.push({ kind: 'motion', features: features(s.slice(-C.SEQ)), speed: pk, hand: HANDS[h], t: pt });
      }
    }
  }
}

// ---------------------------------------------------------------- frame helpers
function toFrame(lm, wl, w, h, t) {
  const sc = C.REF_SIZE / Math.max(w, h);
  const pts = new Float32Array(16), conf = new Float32Array(8), world = new Float32Array(24);
  MP_IDX.forEach((mi, i) => {
    pts[i * 2] = lm[mi].x * w * sc;
    pts[i * 2 + 1] = lm[mi].y * h * sc;
    conf[i] = lm[mi].visibility ?? 1;
    world[i * 3] = wl[mi].x; world[i * 3 + 1] = wl[mi].y; world[i * 3 + 2] = wl[mi].z;
  });
  return { t, pts, conf, world };
}

function lerpFrame(a, b, u, t) {
  const pts = new Float32Array(16), conf = new Float32Array(8), world = new Float32Array(24);
  for (let i = 0; i < 16; i++) pts[i] = a.pts[i] + (b.pts[i] - a.pts[i]) * u;
  for (let i = 0; i < 8; i++) conf[i] = Math.min(a.conf[i], b.conf[i]);
  for (let i = 0; i < 24; i++) world[i] = a.world[i] + (b.world[i] - a.world[i]) * u;
  return { t, pts, conf, world };
}

// Original normalize_keypoints(): centre on mid-shoulders, divide by shoulder width.
export function normalize2d(f) {
  if (f.conf[0] < C.KP_CONF || f.conf[1] < C.KP_CONF) return null;
  const lx = f.pts[0], ly = f.pts[1], rx = f.pts[2], ry = f.pts[3];
  const d = Math.hypot(lx - rx, ly - ry);
  if (d < 1e-4) return null;
  const cx = (lx + rx) / 2, cy = (ly + ry) / 2;
  const out = new Float32Array(16);
  for (let i = 0; i < 8; i++) {
    if (f.conf[i] > C.KP_CONF) {
      out[i * 2] = (f.pts[i * 2] - cx) / d;
      out[i * 2 + 1] = (f.pts[i * 2 + 1] - cy) / d;
    }
  }
  return out;
}

// Original calculate_punch_speed(): faster wrist over the last 5 frames, in pixels.
function originalSpeed(frames) {
  const recent = frames.slice(-5);
  let best = 0;
  for (const wi of [4, 5]) {
    const pos = recent.filter((f) => f.conf[wi] > C.KP_CONF).map((f) => [f.pts[wi * 2], f.pts[wi * 2 + 1]]);
    if (pos.length < 2) continue;
    let dd = 0;
    for (let i = 0; i < pos.length - 1; i++) dd += Math.hypot(pos[i + 1][0] - pos[i][0], pos[i + 1][1] - pos[i][1]);
    best = Math.max(best, dd / pos.length);
  }
  return best;
}

// Wrist speed relative to the shoulders, in shoulder widths per frame, as [left, right].
// Uses 3D (catches punches thrown straight at the camera) and 2D, whichever is larger.
function wristSpeeds(a, b) {
  const sw3 = Math.hypot(b.world[0] - b.world[3], b.world[1] - b.world[4], b.world[2] - b.world[5]) || 1;
  const sw2 = Math.hypot(b.pts[0] - b.pts[2], b.pts[1] - b.pts[3]) || 1;
  return [4, 5].map((wi) => {
    if (b.conf[wi] < C.KP_CONF) return 0;
    const rel = (f, k) => f.world[wi * 3 + k] - (f.world[k] + f.world[3 + k]) / 2;
    const d3 = Math.hypot(rel(b, 0) - rel(a, 0), rel(b, 1) - rel(a, 1), rel(b, 2) - rel(a, 2)) / sw3;
    const rel2 = (f, k) => f.pts[wi * 2 + k] - (f.pts[k] + f.pts[2 + k]) / 2;
    const d2 = Math.hypot(rel2(b, 0) - rel2(a, 0), rel2(b, 1) - rel2(a, 1)) / sw2;
    return Math.max(d3, d2);
  });
}

// How far each hand is from its own shoulder (3D, shoulder widths), as [left, right].
function reaches(f) {
  const w = f.world;
  const sw = Math.hypot(w[0] - w[3], w[1] - w[4], w[2] - w[5]) || 1;
  const l = Math.hypot(w[12] - w[0], w[13] - w[1], w[14] - w[2]);
  const r = Math.hypot(w[15] - w[3], w[16] - w[4], w[17] - w[5]);
  return [l / sw, r / sw];
}

// In guard the lead hand and lead shoulder sit nearer the camera (smaller world z).
// Negative = left side forward (orthodox), positive = right side forward (southpaw).
function stanceScore(f) {
  if ([0, 1, 4, 5].some((i) => f.conf[i] < C.KP_CONF)) return null;
  const w = f.world;
  const sw = Math.hypot(w[0] - w[3], w[1] - w[4], w[2] - w[5]) || 1;
  return ((w[14] - w[17]) + 0.5 * (w[2] - w[5])) / sw;
}

// Original detect_stance().
function detectStance(f) {
  if ([0, 1, 6, 7].some((i) => f.conf[i] < C.KP_CONF)) return null;
  const left = Math.hypot(f.pts[0] - f.pts[12], f.pts[1] - f.pts[13]);
  const right = Math.hypot(f.pts[2] - f.pts[14], f.pts[3] - f.pts[15]);
  return left < right ? 'Southpaw' : 'Orthodox';
}
