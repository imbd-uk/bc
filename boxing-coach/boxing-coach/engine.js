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
  MOTION_REFRACTORY: 0.35,  // s between punches
  MAX_GAP: 0.3,             // s without a usable frame before the stream restarts
};

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
    this.speed = [];        // smoothed wrist speed per stream frame
    this.reach = [];        // max wrist-to-shoulder distance per stream frame
    this.last = null;
    this.nextT = null;
    this.starterUntil = -Infinity;
    this.lastMotionT = -Infinity;
    this.lastSeenT = -Infinity;
  }

  // Returns { landmarks, stance, tracking, events: [...] }
  process(source, w, h, t) {
    const res = this.pose.detectForVideo(source, performance.now());
    const lm = res.landmarks && res.landmarks[0];
    const wl = res.worldLandmarks && res.worldLandmarks[0];
    const events = [];
    let stance = null;
    if (lm && wl) {
      const f = toFrame(lm, wl, w, h, t);
      stance = detectStance(f);
      if (normalize2d(f)) {
        this.lastSeenT = t;
        for (const g of this.resample(f)) this.onGridFrame(g, events);
      }
    }
    return { landmarks: lm ? { lm, w, h } : null, stance, tracking: t - this.lastSeenT < 1, events };
  }

  resample(f) {
    const out = [];
    if (!this.last || f.t - this.last.t > C.MAX_GAP || f.t < this.last.t) {
      this.stream = []; this.speed = []; this.reach = [];
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
    this.reach.push(reachOf(g));
    this.speed.push(prev ? wristSpeed(prev, g) : 0);
    if (s.length > 60) { s.shift(); this.speed.shift(); this.reach.shift(); }

    // --- starter model (original LSTM, original gates)
    if (this.useStarter && s.length >= C.SEQ && g.t >= this.starterUntil) {
      const win = s.slice(-C.SEQ);
      const seq = win.map(normalize2d);
      if (seq.every(Boolean)) {
        const { index, confidence } = this.starter.predict(seq);
        if (confidence > C.STARTER_CONF && originalSpeed(win) > C.STARTER_MIN_SPEED) {
          this.starterUntil = g.t + C.STARTER_RECOVERY;
          events.push({ kind: 'punch', label: STARTER_LABELS[index], confidence, model: 'starter', t: g.t });
        }
      }
    }

    // --- motion detector: a wrist-speed peak while the arm is extending
    const n = this.speed.length - 1;
    const L = C.MOTION_LOOKAHEAD;
    const p = n - L;
    if (s.length >= C.SEQ && p >= 2) {
      const sm = (i) => (this.speed[i - 1] + this.speed[i] + this.speed[i + 1]) / 3;
      const sp = sm(p);
      let isPeak = sp >= this.motionSpeed;
      for (let i = Math.max(1, p - L); isPeak && i <= n - 1; i++) if (i !== p && sm(i) > sp) isPeak = false;
      const extending = this.reach[p + 1] - this.reach[p - 2] > 0;
      const pt = s[p].t;
      if (isPeak && extending && pt - this.lastMotionT >= C.MOTION_REFRACTORY) {
        this.lastMotionT = pt;
        events.push({ kind: 'motion', features: features(s.slice(-C.SEQ)), speed: sp, t: pt });
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

// Wrist speed relative to the shoulders, in shoulder widths per frame.
// Uses 3D (catches punches thrown straight at the camera) and 2D, whichever is larger.
function wristSpeed(a, b) {
  const sw3 = Math.hypot(b.world[0] - b.world[3], b.world[1] - b.world[4], b.world[2] - b.world[5]) || 1;
  const sw2 = Math.hypot(b.pts[0] - b.pts[2], b.pts[1] - b.pts[3]) || 1;
  let best = 0;
  for (const wi of [4, 5]) {
    if (b.conf[wi] < C.KP_CONF) continue;
    const rel = (f, k) => f.world[wi * 3 + k] - (f.world[k] + f.world[3 + k]) / 2;
    const d3 = Math.hypot(rel(b, 0) - rel(a, 0), rel(b, 1) - rel(a, 1), rel(b, 2) - rel(a, 2)) / sw3;
    const rel2 = (f, k) => f.pts[wi * 2 + k] - (f.pts[k] + f.pts[2 + k]) / 2;
    const d2 = Math.hypot(rel2(b, 0) - rel2(a, 0), rel2(b, 1) - rel2(a, 1)) / sw2;
    best = Math.max(best, d3, d2);
  }
  return best;
}

// How far the hands are from their own shoulders (3D, shoulder widths).
function reachOf(f) {
  const w = f.world;
  const sw = Math.hypot(w[0] - w[3], w[1] - w[4], w[2] - w[5]) || 1;
  const l = Math.hypot(w[12] - w[0], w[13] - w[1], w[14] - w[2]);
  const r = Math.hypot(w[15] - w[3], w[16] - w[4], w[17] - w[5]);
  return Math.max(l, r) / sw;
}

// Original detect_stance().
function detectStance(f) {
  if ([0, 1, 6, 7].some((i) => f.conf[i] < C.KP_CONF)) return null;
  const left = Math.hypot(f.pts[0] - f.pts[12], f.pts[1] - f.pts[13]);
  const right = Math.hypot(f.pts[2] - f.pts[14], f.pts[3] - f.pts[15]);
  return left < right ? 'Southpaw' : 'Orthodox';
}
