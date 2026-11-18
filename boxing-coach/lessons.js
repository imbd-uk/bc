// Lessons: stance, footwork and the six punches, each with an animated 3D boxer.
// The boxer is a small rig (2-bone IK arms and legs) posed by keyframes and drawn
// on a canvas in the same glowing-skeleton style as the live coach. No video files.

import { NAMES } from './personal.js';

// ---------------------------------------------------------------- the rig
// World: metres, y up, z = the way the boxer faces, x = the boxer's left. Drawn orthodox
// (left lead); southpaw is the mirror image.
const BASE = {
  crouch: 0.05,          // pelvis drop
  shift: 0,              // weight: -1 rear foot … +1 lead foot
  lean: 0.04, leanX: 0,  // shoulders forward / sideways of the pelvis
  hipYaw: 0.5, shYaw: 0.6, // turn of hips and shoulders (positive = lead side forward)
  lw: [0.1, 0.1, 0.24], rw: [-0.07, 0.12, 0.13], // wrists, relative to mid-shoulders
  lp: [0.5, -1, 0], rp: [-0.5, -1, 0],            // which way each elbow points
  lf: [0.13, 0.3], rf: [-0.13, -0.22],            // feet (x, z)
  la: -0.35, ra: -0.95,  // toe direction of each foot (radians, 0 = straight ahead)
  lh: 0, rh: 0,          // heel lift
};
const GUARD = {};
const HANDS_HOME = { lw: [0.08, 0.13, 0.18] };

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const rotY = ([x, y, z], t) => [x * Math.cos(t) - z * Math.sin(t), y, x * Math.sin(t) + z * Math.cos(t)];

// Two-bone IK: joint between root and target, bent towards `pole`.
function ik(root, target, a, b, pole) {
  let d = sub(target, root);
  const dl = Math.max(1e-4, len(d));
  const n = mul(d, 1 / dl);
  const r = Math.min(a + b - 1e-3, Math.max(Math.abs(a - b) + 1e-3, dl));
  const end = add(root, mul(n, r));
  const cosA = (a * a + r * r - b * b) / (2 * a * r);
  const along = a * cosA, h = a * Math.sqrt(Math.max(0, 1 - cosA * cosA));
  let q = sub(pole, mul(n, dot(pole, n)));
  const ql = len(q) || 1; q = mul(q, 1 / ql);
  return { mid: add(add(root, mul(n, along)), mul(q, h)), end };
}

function build(p) {
  const lf = [p.lf[0], 0, p.lf[1]], rf = [p.rf[0], 0, p.rf[1]];
  const u = 0.5 + p.shift * 0.35;
  const P = [rf[0] + (lf[0] - rf[0]) * u, 0.93 - p.crouch, rf[2] + (lf[2] - rf[2]) * u];
  const hipL = add(P, rotY([0.1, 0, 0], p.hipYaw)), hipR = add(P, rotY([-0.1, 0, 0], p.hipYaw));
  const S = add(P, [p.leanX, 0.5, p.lean]);
  const shL = add(S, rotY([0.19, 0, 0], p.shYaw)), shR = add(S, rotY([-0.19, 0, 0], p.shYaw));
  const neck = add(S, [p.leanX * 0.3, 0.08, 0.01]);
  const head = add(S, [p.leanX * 0.5, 0.2, 0.03 + p.lean * 0.4]);
  const armL = ik(shL, add(S, p.lw), 0.3, 0.28, p.lp);
  const armR = ik(shR, add(S, p.rw), 0.3, 0.28, p.rp);
  const foot = (f, a, h) => ({ ankle: [f[0], 0.08 + h, f[2]], toe: [f[0] + Math.sin(a) * 0.17, 0.02, f[2] + Math.cos(a) * 0.17] });
  const fL = foot(lf, p.la, p.lh), fR = foot(rf, p.ra, p.rh);
  const legL = ik(hipL, fL.ankle, 0.47, 0.46, [Math.sin(p.la), 0.1, Math.cos(p.la)]);
  const legR = ik(hipR, fR.ankle, 0.47, 0.46, [Math.sin(p.ra), 0.1, Math.cos(p.ra)]);
  return {
    head, gloveL: armL.end, gloveR: armR.end,
    segs: [
      // [a, b, side]  side: L = lead, R = rear, C = centre
      [hipL, hipR, 'C'], [shL, shR, 'C'], [hipL, shL, 'C'], [hipR, shR, 'C'], [S, neck, 'C'],
      [shL, armL.mid, 'L'], [armL.mid, armL.end, 'L'], [shR, armR.mid, 'R'], [armR.mid, armR.end, 'R'],
      [hipL, legL.mid, 'L'], [legL.mid, legL.end, 'L'], [legL.end, fL.toe, 'L'],
      [hipR, legR.mid, 'R'], [legR.mid, legR.end, 'R'], [legR.end, fR.toe, 'R'],
    ],
    joints: [[shL, 'L'], [shR, 'R'], [armL.mid, 'L'], [armR.mid, 'R'], [legL.mid, 'L'], [legR.mid, 'R'], [hipL, 'L'], [hipR, 'R']],
  };
}

// ---------------------------------------------------------------- timeline
// Each frame: move to pose `p` over `t` seconds, hold for `hold`. `hit` marks the glove that lands.
const fr = (p, t, hold, cue, o = {}) => ({ p, t, hold, cue, ...o });
const easeIO = (x) => 0.5 - 0.5 * Math.cos(Math.PI * x);
const easeSnap = (x) => 1 - Math.pow(1 - x, 3);

function full(p) {
  const o = { ...BASE };
  for (const k in p) o[k] = p[k];
  return o;
}
function lerpP(a, b, u) {
  const o = {};
  for (const k in a) o[k] = Array.isArray(a[k]) ? a[k].map((v, i) => v + (b[k][i] - v) * u) : a[k] + (b[k] - a[k]) * u;
  return o;
}

class Timeline {
  constructor(frames) {
    this.frames = frames.map((f) => ({ ...f, P: full(f.p) }));
    this.total = this.frames.reduce((s, f) => s + f.t + f.hold, 0);
  }
  // -> { pose, cue, hit, punching }
  at(time) {
    let x = ((time % this.total) + this.total) % this.total;
    const F = this.frames;
    for (let i = 0; i < F.length; i++) {
      const f = F[i], prev = F[(i - 1 + F.length) % F.length];
      if (x < f.t) {
        const u = (f.ease === 'snap' ? easeSnap : easeIO)(x / f.t);
        return { pose: lerpP(prev.P, f.P, u), cue: f.cue, hit: f.hit || null, moving: true };
      }
      x -= f.t;
      if (x < f.hold) return { pose: f.P, cue: f.cue, hit: f.hit || null, moving: false };
      x -= f.hold;
    }
    return { pose: F[0].P, cue: F[0].cue, hit: null, moving: false };
  }
}

// ---------------------------------------------------------------- lessons
export const LESSONS = [
  {
    id: 'stance', title: 'Stance', sub: 'The base everything starts from',
    frames: [
      fr(GUARD, 0.6, 0.9, 'Lead foot forward, feet a bit wider than shoulders'),
      fr({ crouch: 0.09 }, 0.5, 0.5, 'Knees soft, weight on the balls of your feet'),
      fr(GUARD, 0.5, 0.9, 'Hands up by your cheeks, elbows tucked in'),
      fr({ crouch: 0.09 }, 0.5, 0.5, 'Chin down, eyes up, stay relaxed'),
    ],
    steps: [
      'Stand side-on: lead foot forward, rear foot back and out, about shoulder-width plus a bit.',
      'Lead toes point slightly in, rear foot angled out. Rear heel just off the floor.',
      'Knees soft. Weight evenly split, on the balls of the feet.',
      'Lead hand in front at cheek height, rear glove touching the jaw. Elbows guard the ribs.',
      'Chin tucked, looking through your eyebrows.',
    ],
    mistakes: ['Feet in a line (tightrope): you get pushed over easily.', 'Square to the front: gives a bigger target.', 'Hands drifting down to chest height.'],
  },
  {
    id: 'footwork', title: 'Footwork', sub: 'Step and slide, never cross your feet',
    frames: [
      fr(GUARD, 0.4, 0.6, 'Guard'),
      fr({ lf: [0.13, 0.52] }, 0.25, 0.08, 'Forward: lead foot steps first'),
      fr({ lf: [0.13, 0.52], rf: [-0.13, 0.0] }, 0.25, 0.5, 'Rear foot follows the same distance'),
      fr({ lf: [0.13, 0.52], rf: [-0.13, -0.22] }, 0.25, 0.08, 'Back: rear foot first'),
      fr(GUARD, 0.25, 0.5, 'Lead foot follows'),
      fr({ lf: [0.33, 0.3] }, 0.25, 0.08, 'To your lead side: lead foot first'),
      fr({ lf: [0.33, 0.3], rf: [0.07, -0.22] }, 0.25, 0.5, 'Rear foot follows'),
      fr({ lf: [0.33, 0.3], rf: [-0.13, -0.22] }, 0.25, 0.08, 'To your rear side: rear foot first'),
      fr(GUARD, 0.25, 0.7, 'Lead foot follows. Stance stays the same width'),
    ],
    steps: [
      'Whichever way you go, the foot nearest that direction moves first.',
      'Forward: lead foot steps, rear foot slides up the same distance.',
      'Back: rear foot steps, lead foot follows.',
      'Sideways: lead foot first towards your lead side, rear foot first towards your rear side.',
      'Short steps. Keep your stance width and your guard up the whole time.',
    ],
    mistakes: ['Crossing or bringing the feet together.', 'Big steps that leave you off balance.', 'Bouncing high instead of gliding.'],
  },
  {
    id: 'jab', label: 'jab', title: 'Jab', sub: 'Lead hand, straight. Your most-used punch',
    frames: [
      fr(GUARD, 0.4, 0.7, 'Guard'),
      fr({ lw: [0.03, 0.1, 0.6], lp: [0.3, -1, 0], shYaw: 0.8, hipYaw: 0.55, shift: 0.12 }, 0.13, 0.1, 'Snap it straight out, turn the fist over', { hit: 'lead', ease: 'snap' }),
      fr(GUARD, 0.2, 0.6, 'Straight back to guard'),
      fr({ lw: [0.03, 0.1, 0.6], lp: [0.3, -1, 0], shYaw: 0.8, hipYaw: 0.55, shift: 0.12, lf: [0.13, 0.4] }, 0.13, 0.1, 'Add a small step for range', { hit: 'lead', ease: 'snap' }),
      fr(GUARD, 0.22, 0.8, 'Back to guard'),
    ],
    steps: [
      'From guard, push the lead hand straight out from the chin.',
      'Rotate the fist so the palm faces down as it lands.',
      'Lead shoulder rises to protect the chin. Rear hand stays on the jaw.',
      'Bring it back along the same line, as fast as it went out.',
    ],
    mistakes: ['Dropping the hand on the way back.', 'Swinging it wide instead of straight.', 'Leaning the head forward over the lead foot.'],
    drill: 20,
  },
  {
    id: 'cross', label: 'cross', title: 'Cross', sub: 'Rear hand, straight. Your power punch',
    frames: [
      fr(GUARD, 0.4, 0.7, 'Guard'),
      fr({ rw: [0.01, 0.1, 0.6], rp: [-0.3, -1, 0], ...HANDS_HOME, shYaw: -0.25, hipYaw: -0.05, shift: 0.35, ra: -0.25, rh: 0.07 }, 0.17, 0.12,
        'Pivot the back foot, turn the hips through', { hit: 'rear', ease: 'snap' }),
      fr(GUARD, 0.25, 0.8, 'Back to guard, hands home'),
    ],
    steps: [
      'Push off the rear foot and turn the heel out, like squashing a bug.',
      'Hips then shoulders rotate so the rear shoulder comes forward.',
      'The rear hand travels straight down the middle, palm down at the end.',
      'Lead hand comes back to the face. Return on the same line.',
    ],
    mistakes: ['Arm-only punch with flat feet.', 'Lead hand dropping as the cross goes out.', 'Over-rotating and falling forward.'],
    drill: 20,
  },
  {
    id: 'lead_hook', label: 'lead_hook', title: 'Lead hook', sub: 'Lead hand, around the side',
    frames: [
      fr(GUARD, 0.4, 0.7, 'Guard'),
      fr({ shYaw: 0.4, hipYaw: 0.35, lw: [0.16, 0.1, 0.22], shift: 0.15 }, 0.15, 0.05, 'Weight shifts slightly onto the lead foot'),
      fr({ lw: [-0.2, 0.02, 0.42], lp: [0.6, 0.4, 0.6], shYaw: 1.15, hipYaw: 0.95, shift: -0.1, la: -1.1, lh: 0.06, rw: [-0.05, 0.13, 0.12] }, 0.16, 0.12,
        'Turn on the lead foot. Elbow up level, arm bent', { hit: 'lead', ease: 'snap' }),
      fr(GUARD, 0.25, 0.8, 'Back to guard'),
    ],
    steps: [
      'Shift a little weight onto the lead foot.',
      'Lift the lead elbow until the forearm is level with the floor, bent about 90°.',
      'Turn the lead foot, hip and shoulder together. The arm just goes along for the ride.',
      'Rear hand stays glued to the jaw. Turn back into guard.',
    ],
    mistakes: ['Winding up by pulling the hand back first.', 'Arm opening wide like a swing.', 'Not turning the lead foot, so it has no power.'],
    drill: 15,
  },
  {
    id: 'rear_hook', label: 'rear_hook', title: 'Rear hook', sub: 'Rear hand, around the side',
    frames: [
      fr(GUARD, 0.4, 0.7, 'Guard'),
      fr({ rw: [0.12, 0.02, 0.4], rp: [-0.6, 0.4, 0.6], ...HANDS_HOME, shYaw: -0.45, hipYaw: -0.25, shift: 0.25, ra: -0.15, rh: 0.07 }, 0.18, 0.12,
        'Drive the back hip, pivot the rear foot', { hit: 'rear', ease: 'snap' }),
      fr(GUARD, 0.25, 0.8, 'Back to guard'),
    ],
    steps: [
      'Pivot the rear foot and drive the back hip forward, as for a cross.',
      'Rear elbow lifts so the arm is bent and level.',
      'The fist comes round to the side of the target, not straight.',
      'Lead hand protects the face. Turn back into guard.',
    ],
    mistakes: ['Looping it too wide: slow and easy to see.', 'Dropping the lead hand.', 'Throwing it from too far away.'],
    drill: 15,
  },
  {
    id: 'lead_uppercut', label: 'lead_uppercut', title: 'Lead uppercut', sub: 'Lead hand, upwards',
    frames: [
      fr(GUARD, 0.4, 0.7, 'Guard'),
      fr({ crouch: 0.12, leanX: 0.05, shift: 0.2, lw: [0.13, -0.1, 0.2], lp: [0.3, -1, -0.2], shYaw: 0.5 }, 0.25, 0.05, 'Dip onto the lead side'),
      fr({ crouch: 0.04, lw: [0.0, 0.14, 0.34], lp: [0.2, -1, -0.3], shYaw: 0.95, hipYaw: 0.8, la: -0.8, lh: 0.04 }, 0.15, 0.12,
        'Drive up through the legs, palm facing you', { hit: 'lead', ease: 'snap' }),
      fr(GUARD, 0.25, 0.8, 'Back to guard'),
    ],
    steps: [
      'Dip slightly by bending the knees and leaning onto the lead side.',
      'Lead hand drops a little, palm facing you, elbow close to the body.',
      'Straighten the legs and turn the lead hip: the fist goes up, not out.',
      'Short punch. Finish at chin height and come back to guard.',
    ],
    mistakes: ['Dropping the hand right down to the hip (wind-up).', 'Arm straightening out like a shovel.', 'Leaning back as you punch.'],
    drill: 15,
  },
  {
    id: 'rear_uppercut', label: 'rear_uppercut', title: 'Rear uppercut', sub: 'Rear hand, upwards',
    frames: [
      fr(GUARD, 0.4, 0.7, 'Guard'),
      fr({ crouch: 0.12, leanX: -0.05, shift: -0.15, rw: [-0.1, -0.12, 0.16], rp: [-0.3, -1, -0.2] }, 0.25, 0.05, 'Dip slightly to the rear side'),
      fr({ crouch: 0.04, rw: [0.0, 0.14, 0.36], rp: [-0.2, -1, -0.3], ...HANDS_HOME, shYaw: -0.1, hipYaw: 0.05, shift: 0.3, ra: -0.3, rh: 0.07 }, 0.16, 0.12,
        'Drive up off the back foot, turn the hip', { hit: 'rear', ease: 'snap' }),
      fr(GUARD, 0.25, 0.8, 'Back to guard'),
    ],
    steps: [
      'Bend the knees slightly and sink onto the rear side.',
      'Rear hand drops a little, palm facing you, elbow in.',
      'Push off the back foot and turn the rear hip forward: the fist drives straight up the middle.',
      'Lead hand stays on the face. Back to guard.',
    ],
    mistakes: ['Big wind-up that telegraphs it.', 'Standing up tall with straight legs.', 'Leaving the chin open with the lead hand down.'],
    drill: 15,
  },
];

// ---------------------------------------------------------------- drawing
const VIEWS = { front: 0, three: -0.7, side: -Math.PI / 2 };

export class Player {
  constructor(canvas, cueEl) {
    this.c = canvas; this.ctx = canvas.getContext('2d'); this.cueEl = cueEl;
    this.view = 'three'; this.speed = 1; this.mirror = false;
    this.playing = true; this.time = 0; this.last = null; this.raf = 0; this.trail = [];
    this.tl = null; this.lastCue = '';
  }
  load(lesson) { this.tl = new Timeline(lesson.frames); this.time = 0; this.trail = []; this.lastCue = ''; this.draw(); }
  start() { if (!this.raf) { this.last = null; this.raf = requestAnimationFrame((t) => this.tick(t)); } }
  stop() { cancelAnimationFrame(this.raf); this.raf = 0; }
  tick(now) {
    this.raf = requestAnimationFrame((t) => this.tick(t));
    if (this.last != null && this.playing) this.time += Math.min(0.1, (now - this.last) / 1000) * this.speed;
    this.last = now;
    this.draw();
  }

  draw() {
    if (!this.tl) return;
    const c = this.c, ctx = this.ctx;
    const cw = c.clientWidth, ch = c.clientHeight, dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const st = this.tl.at(this.time);
    if (st.cue !== this.lastCue) { this.lastCue = st.cue; this.cueEl.textContent = st.cue; }
    const fig = build(st.pose);
    const v = VIEWS[this.view];
    const sc = ch / 2.1, ox = cw / 2, oy = ch - 18;
    const cv = Math.cos(v), sv = Math.sin(v);
    const depth = (p) => p[2] * cv - p[0] * sv;
    const proj = (p) => {
      let sx = p[0] * cv + p[2] * sv;
      if (this.mirror) sx = -sx;
      const d = depth(p);
      return [ox + sx * sc, oy - p[1] * sc + d * sc * 0.12, d];
    };

    // floor
    ctx.save();
    const g = ctx.createRadialGradient(ox, oy, 4, ox, oy, sc * 0.7);
    g.addColorStop(0, 'rgba(185,212,255,0.18)'); g.addColorStop(1, 'rgba(185,212,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.ellipse(ox, oy, sc * 0.7, sc * 0.12, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();

    const col = { L: [255, 255, 255], R: [185, 212, 255], C: [235, 240, 248] };
    const alpha = (d) => Math.max(0.4, Math.min(1, 0.72 + d * 1.6));
    const items = [];
    for (const [a, b, s] of fig.segs) {
      const A = proj(a), B = proj(b);
      items.push({ d: (A[2] + B[2]) / 2, draw: () => {
        const [r, gg, bb] = col[s];
        ctx.strokeStyle = `rgba(${r},${gg},${bb},${alpha((A[2] + B[2]) / 2)})`;
        ctx.lineWidth = s === 'C' ? 3 : 3.5;
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
      } });
    }
    for (const [p, s] of fig.joints) {
      const P = proj(p);
      items.push({ d: P[2] + 0.001, draw: () => {
        const [r, gg, bb] = col[s];
        ctx.fillStyle = `rgba(${r},${gg},${bb},${alpha(P[2])})`;
        ctx.beginPath(); ctx.arc(P[0], P[1], 3.5, 0, Math.PI * 2); ctx.fill();
      } });
    }
    const H = proj(fig.head);
    items.push({ d: H[2], draw: () => {
      ctx.strokeStyle = 'rgba(244,246,248,0.95)'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(H[0], H[1], 0.1 * sc, 0, Math.PI * 2); ctx.stroke();
    } });
    // gloves; the one landing glows green with a short trail
    const hot = st.hit;
    for (const [side, p] of [['lead', fig.gloveL], ['rear', fig.gloveR]]) {
      const P = proj(p);
      const on = hot === side;
      if (on) { this.trail.push([P[0], P[1]]); if (this.trail.length > 8) this.trail.shift(); }
      items.push({ d: P[2] + 0.002, draw: () => {
        if (on && this.trail.length > 1) {
          ctx.strokeStyle = 'rgba(126,226,176,0.35)'; ctx.lineWidth = 0.09 * sc; ctx.lineCap = 'round';
          ctx.beginPath(); this.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
        }
        ctx.fillStyle = on ? '#7ee2b0' : side === 'lead' ? '#ffffff' : '#b9d4ff';
        ctx.globalAlpha = alpha(P[2]);
        ctx.beginPath(); ctx.arc(P[0], P[1], 0.065 * sc, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
      } });
    }
    if (!hot) this.trail = [];

    items.sort((a, b) => a.d - b.d);
    ctx.lineCap = 'round';
    ctx.shadowColor = 'rgba(185,212,255,0.7)'; ctx.shadowBlur = 10;
    for (const it of items) it.draw();
    ctx.shadowBlur = 0;
  }
}

// ---------------------------------------------------------------- pages
// opts: { root, getLead(): 'left'|'right'|null, onPractise(label, reps) }
export function initLessons({ root, getLead, onPractise }) {
  const listView = document.createElement('div'); listView.className = 'lesson-list-view';
  const detail = document.createElement('div'); detail.className = 'lesson-detail'; detail.hidden = true;
  root.append(listView, detail);

  // list
  const head = document.createElement('header'); head.className = 'page-head';
  head.innerHTML = '<p class="eyebrow">Lessons</p><h1>Learn the basics, <span class="dim">then drill them.</span></h1>'
    + '<p class="lede">Watch each move from any angle, read the key points, then practise it with the coach counting.</p>';
  const list = document.createElement('div'); list.className = 'card lesson-list';
  LESSONS.forEach((l, i) => {
    const a = document.createElement('a'); a.className = 'lesson-item'; a.href = `#lessons/${l.id}`;
    const n = document.createElement('span'); n.className = 'lesson-num num'; n.textContent = String(i + 1);
    const t = document.createElement('span'); t.className = 'lesson-text';
    const tt = document.createElement('span'); tt.className = 'lesson-title'; tt.textContent = l.title;
    const ts = document.createElement('span'); ts.className = 'lesson-sub'; ts.textContent = l.sub;
    t.append(tt, ts);
    a.append(n, t);
    a.insertAdjacentHTML('beforeend', '<svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>');
    list.append(a);
  });
  listView.append(head, list);

  // detail
  detail.innerHTML = `
    <a class="back-link" href="#lessons"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>Lessons</a>
    <header class="page-head"><p class="eyebrow" data-k="eyebrow"></p><h1 data-k="title"></h1><p class="lede" data-k="sub"></p></header>
    <div class="card demo">
      <div class="demo-stage">
        <canvas class="demo-canvas" aria-label="Animated demonstration"></canvas>
        <p class="demo-cue" data-k="cue"></p>
      </div>
      <div class="demo-controls">
        <button type="button" class="mini" data-k="play">Pause</button>
        <div class="seg" role="radiogroup" aria-label="Speed" data-k="speed">
          <button type="button" role="radio" data-value="0.4">Slow</button>
          <button type="button" role="radio" data-value="1">Normal</button>
        </div>
        <div class="seg" role="radiogroup" aria-label="Camera angle" data-k="view">
          <button type="button" role="radio" data-value="front">Front</button>
          <button type="button" role="radio" data-value="three">¾</button>
          <button type="button" role="radio" data-value="side">Side</button>
        </div>
      </div>
      <p class="card-sub" data-k="stance"></p>
    </div>
    <div class="card tip"><p class="card-title">Key points</p><ol class="lesson-steps" data-k="steps"></ol></div>
    <div class="card tip"><p class="card-title">Common mistakes</p><ul data-k="mistakes"></ul></div>
    <button type="button" class="btn primary" data-k="practise"></button>
    <p class="hint-text" data-k="practise-note"></p>
    <a class="btn ghost" data-k="next" href="#lessons"></a>`;
  const $ = (k) => detail.querySelector(`[data-k="${k}"]`);
  const player = new Player(detail.querySelector('.demo-canvas'), $('cue'));
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  let current = null;

  const syncSeg = (seg, val) => seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === String(val))));
  $('speed').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; player.speed = Number(b.dataset.value); syncSeg($('speed'), b.dataset.value); });
  $('view').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; player.view = b.dataset.value; syncSeg($('view'), b.dataset.value); player.trail = []; player.draw(); });
  $('play').addEventListener('click', () => { player.playing = !player.playing; $('play').textContent = player.playing ? 'Pause' : 'Play'; });
  $('practise').addEventListener('click', () => { if (current?.label) onPractise(current.label, current.drill); });
  syncSeg($('speed'), 1); syncSeg($('view'), 'three');

  function show(id) {
    const i = LESSONS.findIndex((l) => l.id === id);
    if (i < 0) {
      current = null; player.stop();
      listView.hidden = false; detail.hidden = true;
      return;
    }
    const l = LESSONS[i];
    current = l;
    listView.hidden = true; detail.hidden = false;
    const lead = getLead() || 'left';
    player.mirror = lead === 'right';
    $('eyebrow').textContent = `Lesson ${i + 1} of ${LESSONS.length}`;
    $('title').textContent = l.title;
    $('sub').textContent = l.sub;
    $('stance').textContent = `Shown ${lead === 'right' ? 'southpaw (right hand leads)' : 'orthodox (left hand leads)'}`
      + `${getLead() ? ', your stance' : '. Set your stance on the Teach page to flip it'}. Green glove = the punch landing.`;
    $('steps').replaceChildren(...l.steps.map((s) => Object.assign(document.createElement('li'), { textContent: s })));
    $('mistakes').replaceChildren(...l.mistakes.map((s) => Object.assign(document.createElement('li'), { textContent: s })));
    $('practise').hidden = !l.label; $('practise-note').hidden = !l.label;
    if (l.label) {
      const nm = NAMES[l.label].toLowerCase();
      $('practise').textContent = `Practise ${l.drill} ${nm}${nm.endsWith('s') ? 'es' : 's'} with the coach`;
      $('practise-note').textContent = 'Starts the camera. The coach counts only this punch and tells you when you hit the target.';
    }
    const next = LESSONS[i + 1];
    $('next').href = next ? `#lessons/${next.id}` : '#lessons';
    $('next').textContent = next ? `Next: ${next.title}` : 'All lessons';
    player.load(l);
    player.playing = !reduce; $('play').textContent = player.playing ? 'Pause' : 'Play';
    player.start();
  }
  function hide() { player.stop(); }
  document.addEventListener('visibilitychange', () => { if (document.hidden) player.stop(); else if (current && !detail.hidden) player.start(); });
  return { show, hide };
}
