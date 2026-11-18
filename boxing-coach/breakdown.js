// Fight breakdown: upload a clip of two fighters, track both with MediaPipe,
// detect and classify each punch, then group them into exchanges and describe
// who led, the setups, the counters and how each exchange ended.
// Everything runs on this device; the clip is never uploaded.

import { FilesetResolver, PoseLandmarker } from './vendor/vision_bundle.mjs';
import { NAMES, matchCombo } from './personal.js';
import { voice } from './voice.js';

// MediaPipe landmarks we keep per pose (x px, y px, visibility).
const KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const SLOT = Object.fromEntries(KEEP.map((mi, i) => [mi, i]));
const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];
const COLORS = ['#b9d4ff', '#ffcf70'];

const A = {
  SPEED: 2.0,         // wrist speed (torso lengths / s) that can start a punch
  PEAK_WIN: 0.12,     // s either side a peak must beat
  REFRACTORY: 0.22,   // s between punches from the same hand
  MIN_REACH: 0.5,     // fist must get at least this far from its shoulder (torso lengths); hooks look short side-on
  EXCHANGE_GAP: 1.2,  // s of quiet that ends an exchange
  RUN_GAP: 0.8,       // s between punches that still counts as one combination
  COUNTER_WIN: 0.6,   // s after an opponent's punch that counts as a counter
  MOVE: 0.35,         // torso lengths of travel that counts as stepping in / out
};

// ---------------------------------------------------------------- pose helpers
export const pt = (p, mi) => { const s = SLOT[mi] * 3; return [p[s], p[s + 1], p[s + 2]]; };
export const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
export function center(p) { const s = mid(pt(p, 11), pt(p, 12)), h = mid(pt(p, 23), pt(p, 24)); return mid(s, h); }
export function torso(p) { return Math.max(1, d2(mid(pt(p, 11), pt(p, 12)), mid(pt(p, 23), pt(p, 24)))); }
export { BONES, COLORS };
function elbowAngle(p, side) {
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

// ---------------------------------------------------------------- tracking
// Keeps two fighter identities across frames. Fighter 0 starts on the left.
// A third person (usually the referee) is ignored: the fighters are the pair that face
// each other and stand close; a referee tends to face the camera or stand off to one side.
export class Tracker {
  constructor(width) { this.pos = [null, null]; this.gate = width * 0.22; this.width = width; }
  update(dets) {
    const cs = dets.map(center);
    if (!this.pos[0]) {
      if (dets.length < 2) return [null, null];
      // how much a person faces direction dir (+1 = right): nose ahead of the body
      const facing = (k, dir) => Math.max(0, Math.min(1, ((pt(dets[k], 0)[0] - cs[k][0]) * dir) / (torso(dets[k]) * 0.3)));
      let best = null;
      for (let i = 0; i < dets.length; i++) for (let j = i + 1; j < dets.length; j++) {
        const [l, r] = cs[i][0] <= cs[j][0] ? [i, j] : [j, i];
        const cost = d2(cs[l], cs[r]) / this.width - 0.5 * (facing(l, 1) + facing(r, -1));
        if (!best || cost < best.cost) best = { cost, l, r };
      }
      this.pos = [cs[best.l], cs[best.r]];
      return [dets[best.l], dets[best.r]];
    }
    let best = { cost: Infinity, a: -1, b: -1 };
    for (let a = -1; a < dets.length; a++) for (let b = -1; b < dets.length; b++) {
      if (a >= 0 && a === b) continue;
      const ca = a < 0 ? this.gate : d2(cs[a], this.pos[0]);
      const cb = b < 0 ? this.gate : d2(cs[b], this.pos[1]);
      if (ca > this.gate * 1.0001 || cb > this.gate * 1.0001) continue;
      if (ca + cb < best.cost) best = { cost: ca + cb, a, b };
    }
    if (best.a >= 0) this.pos[0] = cs[best.a];
    if (best.b >= 0) this.pos[1] = cs[best.b];
    return [best.a >= 0 ? dets[best.a] : null, best.b >= 0 ? dets[best.b] : null];
  }
}

// ---------------------------------------------------------------- analysis
// frames: [{ t, poses: [pose|null, pose|null] }]  (pose = packPose output)
export function analyse(frames) {
  const series = [0, 1].map((k) => frames.filter((f) => f.poses[k]).map((f) => ({ t: f.t, p: f.poses[k] })));
  const nearest = (k, t) => {
    const s = series[k];
    if (!s.length) return null;
    let lo = 0, hi = s.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (s[m].t < t) lo = m + 1; else hi = m; }
    const c = s[lo], b = s[lo - 1];
    const best = b && Math.abs(b.t - t) < Math.abs(c.t - t) ? b : c;
    return Math.abs(best.t - t) < 0.25 ? best : null;
  };
  const xAt = (k, t) => { const f = nearest(k, t); return f ? center(f.p)[0] : null; };
  // +1 when the opponent is to the right
  const dirAt = (k, t) => {
    const me = xAt(k, t), op = xAt(1 - k, t);
    return me != null && op != null ? Math.sign(op - me) || 1 : (k === 0 ? 1 : -1);
  };
  const torsoAvg = [0, 1].map((k) => {
    const v = series[k].map((f) => torso(f.p)).sort((a, b) => a - b);
    return v.length ? v[v.length >> 1] : 100;
  });
  // How far fighter k moved towards the opponent between t0 and t1 (torso lengths).
  const advance = (k, t0, t1) => {
    const a = xAt(k, t0), b = xAt(k, t1);
    if (a == null || b == null) return 0;
    return ((b - a) * dirAt(k, t0)) / torsoAvg[k];
  };

  const punches = [];
  for (const k of [0, 1]) punches.push(...detect(k));
  punches.sort((a, b) => a.t - b.t);
  const exchanges = group(punches);
  for (const ex of exchanges) describe(ex);
  const duration = frames.length ? frames[frames.length - 1].t - frames[0].t : 0;
  return { punches, exchanges, duration, stats: [0, 1].map((k) => fighterStats(k, punches, exchanges, duration)), coverage: series.map((s) => s.length / Math.max(1, frames.length)) };

  function detect(k) {
    const s = series[k];
    const out = [];
    if (s.length < 5) return out;
    const sides = ['left', 'right'];
    const wristMi = { left: 15, right: 16 }, shMi = { left: 11, right: 12 };
    // per-frame wrist position relative to mid-shoulders, in torso lengths
    let Ls = torso(s[0].p);
    const rows = s.map((f) => {
      Ls = Ls * 0.8 + torso(f.p) * 0.2;
      const ms = mid(pt(f.p, 11), pt(f.p, 12));
      const r = {};
      for (const sd of sides) {
        const w = pt(f.p, wristMi[sd]), sh = pt(f.p, shMi[sd]);
        r[sd] = { rel: [(w[0] - ms[0]) / Ls, (w[1] - ms[1]) / Ls], reach: d2(w, sh) / Ls, vis: w[2] };
      }
      return { t: f.t, p: f.p, r };
    });
    const at = (t) => { let best = rows[0]; for (const r of rows) if (Math.abs(r.t - t) < Math.abs(best.t - t)) best = r; return best; };
    for (const sd of sides) {
      const v = rows.map((r, i) => {
        const q = rows[i - 1];
        if (!q) return 0;
        const dt = r.t - q.t;
        if (dt <= 0 || dt > 0.12) return 0;
        return Math.hypot(r.r[sd].rel[0] - q.r[sd].rel[0], r.r[sd].rel[1] - q.r[sd].rel[1]) / dt;
      });
      const sv = v.map((x, i) => (v[i - 1] ?? x) / 3 + x / 3 + (v[i + 1] ?? x) / 3);
      let lastT = -Infinity;
      for (let i = 1; i < rows.length - 1; i++) {
        const t = rows[i].t;
        if (sv[i] < A.SPEED || rows[i].r[sd].vis < 0.3 || t - lastT < A.REFRACTORY) continue;
        let isPeak = true;
        for (let j = i - 1; j >= 0 && t - rows[j].t <= A.PEAK_WIN; j--) if (sv[j] > sv[i]) { isPeak = false; break; }
        for (let j = i + 1; isPeak && j < rows.length && rows[j].t - t <= A.PEAK_WIN; j++) if (sv[j] >= sv[i]) isPeak = false;
        if (!isPeak) continue;
        if (at(t + 0.08).r[sd].reach - at(t - 0.08).r[sd].reach <= 0.05) continue;  // must be extending
        // furthest point of the punch
        let far = rows[i];
        for (const r of rows) if (r.t >= t - 0.05 && r.t <= t + 0.2 && r.r[sd].reach > far.r[sd].reach) far = r;
        if (far.r[sd].reach < A.MIN_REACH) continue;
        lastT = t;
        out.push(classify(k, sd, t, at(t - 0.15), far));
      }
    }
    return out;
  }

  function classify(k, hand, t, start, far) {
    const dir = dirAt(k, t);
    const disp = [far.r[hand].rel[0] - start.r[hand].rel[0], far.r[hand].rel[1] - start.r[hand].rel[1]];
    const toward = disp[0] * dir, up = -disp[1];
    const elbow = elbowAngle(far.p, hand);
    // lead side = the shoulder nearer the opponent just before the punch
    let lead = 0, n = 0;
    for (const f of series[k]) {
      if (f.t < t - 0.6 || f.t > t - 0.1) continue;
      lead += (pt(f.p, 11)[0] - pt(f.p, 12)[0]) * dir; n++;
    }
    if (!n) { const f = nearest(k, t); if (f) lead = (pt(f.p, 11)[0] - pt(f.p, 12)[0]) * dir; }
    const leadSide = lead >= 0 ? 'left' : 'right';
    const role = hand === leadSide ? 'lead' : 'rear';
    let shape;
    if (up > 0.3 && up > Math.abs(toward) * 0.8) shape = 'uppercut';
    else if (elbow >= 140 && toward > 0.15) shape = 'straight';
    else shape = 'hook';
    const label = shape === 'straight' ? (role === 'lead' ? 'jab' : 'cross') : `${role}_${shape}`;
    // head or body: where the fist ends up against the opponent's torso
    let target = null;
    const op = nearest(1 - k, far.t);
    if (op) {
      const sy = mid(pt(op.p, 11), pt(op.p, 12))[1], hy = mid(pt(op.p, 23), pt(op.p, 24))[1];
      const wy = pt(far.p, hand === 'left' ? 15 : 16)[1];
      target = wy > sy + 0.4 * (hy - sy) ? 'body' : 'head';
    }
    return { f: k, t, hand, role, label, target, tEnd: far.t };
  }

  function group(ps) {
    const out = [];
    for (const p of ps) {
      const cur = out[out.length - 1];
      if (cur && p.t - cur.end <= A.EXCHANGE_GAP) { cur.punches.push(p); cur.end = Math.max(cur.end, p.tEnd); }
      else out.push({ start: p.t, end: p.tEnd, punches: [p] });
    }
    out.forEach((ex, i) => { ex.n = i + 1; });
    return out;
  }

  // Splits an exchange into each fighter's runs of punches (combinations).
  function runs(ex) {
    const rs = [];
    for (const p of ex.punches) {
      const r = rs[rs.length - 1];
      if (r && r.f === p.f && p.t - r.punches[r.punches.length - 1].t <= A.RUN_GAP) r.punches.push(p);
      else rs.push({ f: p.f, punches: [p] });
    }
    return rs;
  }

  // Builds structured notes; names are filled in when rendering ({0} / {1}).
  function describe(ex) {
    const notes = [];
    const ps = ex.punches;
    const init = ps[0].f, other = 1 - init;
    ex.init = init;
    ex.count = [0, 1].map((k) => ps.filter((p) => p.f === k).length);
    ex.runs = runs(ex);

    // how it started
    const inA = advance(init, ex.start - 1, ex.start), inB = advance(other, ex.start - 1, ex.start);
    if (inA > A.MOVE) notes.push({ k: 'move', text: `{${init}} steps in to start it` });
    else if (inB > A.MOVE) notes.push({ k: 'move', text: `{${other}} walks forward and gets met` });
    if (inB < -A.MOVE) notes.push({ k: 'move', text: `{${other}} was backing up` });

    // each combination
    ex.runs.forEach((r, i) => {
      const seq = r.punches.map((p) => p.label);
      const combo = matchCombo(seq);
      const who = `{${r.f}}`;
      const list = r.punches.map(pName).join(', ');
      if (i === 0) notes.push({ k: 'lead', text: `${who} leads with ${list}${combo ? ` (${combo.id})` : ''}` });
      else {
        const prev = ex.runs[i - 1];
        const gap = r.punches[0].t - prev.punches[prev.punches.length - 1].t;
        if (prev.f !== r.f && gap <= A.COUNTER_WIN) {
          const hit = prev.punches[prev.punches.length - 1];
          const c = r.punches[0];
          notes.push({ k: 'counter', text: counterText(who, hit, c, gap) + (r.punches.length > 1 ? `, then ${r.punches.slice(1).map(pName).join(', ')}` : '') });
        } else notes.push({ k: 'lead', text: `${who} ${prev.f === r.f ? 'goes again' : 'answers'} with ${list}` });
      }
      // setups inside the run
      const first = r.punches[0];
      if (seq.length >= 2 && seq[0] === 'jab' && seq[1] === 'jab') notes.push({ k: 'setup', text: `${who} doubles the jab to close the distance` });
      const power = r.punches.find((p, j) => j > 0 && p.role === 'rear');
      if (first.label === 'jab' && power) notes.push({ k: 'setup', text: `The jab sets up the ${pName(power, true)}` });
      const tg = r.punches.map((p) => p.target).filter(Boolean);
      const bodyThenHead = tg.indexOf('body') >= 0 && tg.lastIndexOf('head') > tg.indexOf('body');
      const headThenBody = tg.indexOf('head') >= 0 && tg.lastIndexOf('body') > tg.indexOf('head');
      if (bodyThenHead) notes.push({ k: 'setup', text: `${who} goes to the body first, then comes up top` });
      else if (headThenBody) notes.push({ k: 'setup', text: `${who} draws the guard up, then digs to the body` });
      if (seq.length >= 3) notes.push({ k: 'setup', text: `${who} puts together a ${seq.length}-punch combination` });
    });

    // how it ended
    const last = ps[ps.length - 1];
    if (ps.some((p) => p.f !== last.f)) notes.push({ k: 'end', text: `{${last.f}} has the last word` });
    const outA = advance(init, ex.end, ex.end + 1), outB = advance(other, ex.end, ex.end + 1);
    if (outA < -A.MOVE) notes.push({ k: 'end', text: `{${init}} steps back out` });
    else if (outA > A.MOVE) notes.push({ k: 'end', text: `{${init}} keeps walking forward` });
    if (outB < -A.MOVE && outA >= -A.MOVE) notes.push({ k: 'end', text: `{${other}} gives ground` });
    ex.notes = notes;
  }

  function counterText(who, hit, c, gap) {
    const g = `${gap.toFixed(1)} s`;
    if (gap < 0.2) return `${who} fires back at the same time with ${pName(c)}`;
    if (hit.label === 'jab' && c.label === 'cross') return `${who} counters the jab with a cross over the top (${g} later)`;
    if ((hit.label === 'jab' || hit.label === 'cross') && c.label === 'lead_hook') return `${who} counters the straight with a lead hook (${g} later)`;
    if (/hook/.test(hit.label) && /uppercut/.test(c.label)) return `${who} counters the hook with an uppercut on the inside (${g} later)`;
    return `${who} counters the ${pName(hit, true)} with ${pName(c)} (${g} later)`;
  }
}

function pName(p, bare = false) {
  const n = NAMES[p.label].toLowerCase();
  const tgt = p.target === 'body' ? ' to the body' : '';
  return bare ? n + tgt : `${/^[aeiou]/.test(n) ? 'an' : 'a'} ${n}${tgt}`;
}

function fighterStats(k, punches, exchanges, duration) {
  const mine = punches.filter((p) => p.f === k);
  const by = {};
  for (const p of mine) by[p.label] = (by[p.label] || 0) + 1;
  const body = mine.filter((p) => p.target === 'body').length;
  const tgt = mine.filter((p) => p.target).length;
  const led = exchanges.filter((e) => e.init === k).length;
  let counters = 0;
  const pairs = {};
  for (const ex of exchanges) {
    ex.runs.forEach((r, i) => {
      if (r.f !== k) return;
      const prev = ex.runs[i - 1];
      if (prev && prev.f !== k && r.punches[0].t - prev.punches[prev.punches.length - 1].t <= A.COUNTER_WIN) counters++;
      for (let j = 1; j < r.punches.length; j++) {
        const key = `${r.punches[j - 1].label}>${r.punches[j].label}`;
        pairs[key] = (pairs[key] || 0) + 1;
      }
    });
  }
  const topPair = Object.entries(pairs).sort((a, b) => b[1] - a[1])[0] || null;
  return {
    total: mine.length, by, led, counters,
    bodyPct: tgt ? body / tgt : null,
    perMin: duration >= 10 ? mine.length / (duration / 60) : null,
    topPair: topPair && topPair[1] >= 2 ? { seq: topPair[0].split('>'), n: topPair[1] } : null,
  };
}

// Overall read of the clip, one line each.
export function insights(res, names) {
  const [a, b] = res.stats;
  const out = [];
  const n = res.exchanges.length;
  if (!n) return ['No exchanges found. Try a clip where both fighters stay fully in frame, side-on to the camera.'];
  for (const k of [0, 1]) {
    const s = res.stats[k], nm = names[k];
    if (s.led >= Math.max(2, n * 0.6)) out.push(`${nm} led ${s.led} of ${n} exchanges: the one applying pressure.`);
    if (s.counters >= 2 && s.counters >= s.led) out.push(`${nm} countered ${s.counters} times: happy to let the other man lead and fire back.`);
    if (s.topPair) out.push(`${nm}'s go-to: ${s.topPair.seq.map((l) => NAMES[l].toLowerCase()).join(' → ')} (${s.topPair.n} times).`);
    if (s.total >= 6 && (s.by.jab || 0) / s.total >= 0.4) out.push(`${nm} lives off the jab (${Math.round(((s.by.jab || 0) / s.total) * 100)}% of punches).`);
    if (s.bodyPct != null && s.total >= 6 && s.bodyPct >= 0.3) out.push(`${nm} invests in the body (${Math.round(s.bodyPct * 100)}% of shots).`);
  }
  if (a.total && b.total) {
    const more = a.total > b.total ? 0 : 1, r = res.stats[more].total / Math.max(1, res.stats[1 - more].total);
    if (r >= 1.4) out.push(`${names[more]} out-threw ${names[1 - more]} ${res.stats[more].total} to ${res.stats[1 - more].total}.`);
  }
  return out.length ? out : ['A fairly even clip: neither fighter dominated the leading or the output.'];
}

// ---------------------------------------------------------------- page
let landmarker = null;
async function getLandmarker() {
  if (landmarker) return landmarker;
  const vision = await FilesetResolver.forVisionTasks('./vendor/wasm');
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: './models/pose_landmarker_full.task', delegate },
    runningMode: 'VIDEO', numPoses: 3,
    minPoseDetectionConfidence: 0.4, minPosePresenceConfidence: 0.4, minTrackingConfidence: 0.4,
  });
  try { landmarker = await PoseLandmarker.createFromOptions(vision, opts('GPU')); }
  catch (e) { console.warn('GPU unavailable, using CPU', e); landmarker = await PoseLandmarker.createFromOptions(vision, opts('CPU')); }
  return landmarker;
}

export function initBreakdown({ root }) {
  root.innerHTML = `
    <header class="page-head">
      <p class="eyebrow">Fight breakdown</p>
      <h1>Study the pros, <span class="dim">exchange by exchange.</span></h1>
      <p class="lede">Upload a clip of two fighters. It tracks both, calls every punch, and breaks down who led, the setups and the counters.</p>
    </header>
    <div class="card" data-k="setup">
      <div class="row"><span class="row-label">Left fighter</span><input class="name-in" data-k="n0" value="Left fighter" maxlength="24" aria-label="Left fighter name"></div>
      <div class="row"><span class="row-label">Right fighter</span><input class="name-in" data-k="n1" value="Right fighter" maxlength="24" aria-label="Right fighter name"></div>
      <div class="row"><span class="row-label" id="lbl-quality">Detail</span>
        <div class="seg" role="radiogroup" aria-labelledby="lbl-quality" data-k="quality">
          <button type="button" role="radio" data-value="0.5" aria-checked="true">High (½ speed)</button>
          <button type="button" role="radio" data-value="1" aria-checked="false">Quick</button>
        </div>
      </div>
    </div>
    <label class="btn primary" data-k="pick">Choose a fight clip<input type="file" accept="video/*" hidden data-k="file"></label>
    <p class="hint-text" data-k="hint">Works best side-on, both fighters fully in frame, few camera cuts. Clips up to ~3 minutes.</p>
    <p class="status" data-k="status" role="status"></p>
    <div class="card bd-player" data-k="player" hidden>
      <div class="bd-stage"><video data-k="video" playsinline muted></video><canvas data-k="overlay"></canvas><canvas data-k="film" class="film-canvas" hidden></canvas></div>
      <div class="bd-progress" data-k="progress"><div class="bar"><i data-k="pbar"></i></div><span data-k="ptext">Reading the clip…</span><button type="button" class="mini" data-k="stop">Stop here</button></div>
      <div class="bd-timeline" data-k="timeline" hidden></div>
      <div class="bd-legend" data-k="legend" hidden></div>
    </div>
    <div data-k="results" hidden class="bd-results"></div>
    <p class="foot-note"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>The clip is processed on this phone and never uploaded. Calls are estimates from body pose: it sees punches thrown, not whether they landed.</p>`;
  const $ = (k) => root.querySelector(`[data-k="${k}"]`);
  const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
  let rate = 0.5, frames = [], result = null, processing = false, url = null, watchEnd = null, raf = 0;
  const names = () => [$('n0').value.trim() || 'Left fighter', $('n1').value.trim() || 'Right fighter'];

  $('quality').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    rate = Number(b.dataset.value);
    $('quality').querySelectorAll('button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
  });
  for (const k of ['n0', 'n1']) $(k).addEventListener('input', () => { if (result) render(); });
  $('file').addEventListener('change', () => { const f = $('file').files[0]; $('file').value = ''; if (f) run(f); });
  $('stop').addEventListener('click', () => finish());

  async function run(file) {
    stopAll();
    $('status').textContent = ''; $('status').classList.remove('error');
    $('results').hidden = true; $('timeline').hidden = true; $('legend').hidden = true;
    $('player').hidden = false; $('progress').hidden = false;
    $('ptext').textContent = 'Loading the pose model…'; $('pbar').style.width = '0%';
    let lmk;
    try { lmk = await getLandmarker(); }
    catch (e) { console.error(e); fail('The pose model could not load. Check your connection and try again.'); return; }
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(file);
    video.src = url; video.controls = false; video.muted = true;
    try { await new Promise((ok, bad) => { video.onloadedmetadata = ok; video.onerror = bad; }); }
    catch { fail('That video could not be opened on this device.'); return; }
    frames = []; result = null; processing = true;
    const tracker = new Tracker(video.videoWidth);
    const w = video.videoWidth, h = video.videoHeight;
    let lastT = -1;
    const onFrame = (_now, meta) => {
      if (!processing) return;
      const t = meta ? meta.mediaTime : video.currentTime;
      if (t > lastT + 1e-4) {
        lastT = t;
        try {
          const r = lmk.detectForVideo(video, performance.now());
          const dets = (r.landmarks || []).map((lm) => packPose(lm, w, h));
          frames.push({ t, poses: tracker.update(dets) });
        } catch (e) { console.error(e); }
        $('pbar').style.width = `${Math.min(100, (t / (video.duration || 1)) * 100)}%`;
        $('ptext').textContent = `Tracking both fighters… ${fmt(t)} / ${fmt(video.duration)}`;
        drawAt(t);
      }
      next();
    };
    const next = () => { if (!processing) return; if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame); else requestAnimationFrame(() => onFrame(0, null)); };
    video.onended = () => finish();
    video.playbackRate = rate;
    try { await video.play(); } catch (e) { console.error(e); fail('That video could not be played on this device.'); return; }
    next();
  }

  function finish() {
    if (!processing) return;
    processing = false;
    video.pause();
    $('progress').hidden = true;
    result = analyse(frames);
    if (result.coverage.some((c) => c < 0.3)) {
      $('status').textContent = 'Heads up: one fighter was only tracked for part of the clip, so some punches may be missing.';
    }
    video.controls = true; video.playbackRate = 1;
    video.ontimeupdate = () => { if (watchEnd != null && video.currentTime >= watchEnd) { video.pause(); watchEnd = null; } };
    startOverlay();
    render();
  }

  function fail(msg) {
    processing = false;
    $('progress').hidden = true;
    $('status').textContent = msg; $('status').classList.add('error');
  }

  function stopAll() { processing = false; cancelAnimationFrame(raf); raf = 0; try { video.pause(); } catch { /* */ } }

  function startOverlay() {
    cancelAnimationFrame(raf);
    const loop = () => { drawAt(video.currentTime); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
  }

  // Skeletons for both fighters plus a label on each punch as it happens.
  function drawAt(t) {
    const cw = overlay.clientWidth, ch = overlay.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!cw || !ch) return;
    if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
    octx.setTransform(dpr, 0, 0, dpr, 0, 0); octx.clearRect(0, 0, cw, ch);
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !frames.length) return;
    const sc = Math.min(cw / vw, ch / vh), ox = (cw - vw * sc) / 2, oy = (ch - vh * sc) / 2;
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (frames[m].t < t) lo = m + 1; else hi = m; }
    const fr = frames[lo];
    if (!fr || Math.abs(fr.t - t) > 0.3) return;
    octx.lineCap = 'round'; octx.lineWidth = 2.5; octx.shadowBlur = 8;
    fr.poses.forEach((p, k) => {
      if (!p) return;
      const P = (mi) => { const q = pt(p, mi); return [ox + q[0] * sc, oy + q[1] * sc, q[2]]; };
      octx.strokeStyle = COLORS[k]; octx.shadowColor = COLORS[k];
      for (const [a, b] of BONES) {
        const A1 = P(a), B1 = P(b);
        if (A1[2] < 0.3 || B1[2] < 0.3) continue;
        octx.beginPath(); octx.moveTo(A1[0], A1[1]); octx.lineTo(B1[0], B1[1]); octx.stroke();
      }
      if (result) {
        const live = result.punches.filter((q) => q.f === k && t >= q.t - 0.1 && t <= q.t + 0.5);
        live.forEach((q, i) => {
          const H = P(0);
          const txt = NAMES[q.label] + (q.target === 'body' ? ' (body)' : '');
          octx.shadowBlur = 0; octx.font = '600 13px system-ui, sans-serif';
          const tw = octx.measureText(txt).width + 14;
          const x = Math.max(4, Math.min(cw - tw - 4, H[0] - tw / 2)), y = Math.max(4, H[1] - 44 - i * 24);
          octx.fillStyle = 'rgba(5,6,7,0.8)'; octx.beginPath(); octx.roundRect(x, y, tw, 20, 10); octx.fill();
          octx.fillStyle = COLORS[k]; octx.fillText(txt, x + 7, y + 14);
          octx.shadowBlur = 8;
        });
      }
    });
    octx.shadowBlur = 0;
  }

  let director = null;
  async function filmStudy(record, F) {
    if (director || !result) return;
    voice.unlock();
    const { buildScript, Director } = await import('./filmstudy.js');
    const nm = names();
    const beats = buildScript(result, frames, nm, insights(result, nm));
    const canvas = $('film');
    director = new Director({ canvas, video, frames, beats, onStatus: (m) => { F('status').textContent = m; } });
    F('play').hidden = true; F('save').hidden = true; F('stop').hidden = false;
    cancelAnimationFrame(raf); octx.clearRect(0, 0, overlay.width, overlay.height);
    watchEnd = null;
    $('player').scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      await director.prepare({ all: record });
      if (director.stopped) return;
      if (record && !voice.ai) F('status').textContent = 'Recording… (the phone voice can\'t be recorded: switch Voice to Natural on the Analyse page for a voiced video)';
      else F('status').textContent = record ? 'Recording the breakdown… keep this screen open.' : 'Playing the breakdown…';
      canvas.hidden = false;
      const blob = await director.run({ record });
      if (blob && !director.stopped) {
        await saveVideo(blob);
        F('status').textContent = 'Saved. Find it in your downloads (or the share sheet on iPhone).';
      } else F('status').textContent = director.stopped ? 'Stopped.' : 'Done.';
    } catch (e) {
      console.error(e);
      F('status').textContent = `Something went wrong: ${e.message}`;
    } finally {
      canvas.hidden = true;
      director = null;
      F('play').hidden = false; F('save').hidden = false; F('stop').hidden = true;
      startOverlay();
    }
  }

  async function saveVideo(blob) {
    const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
    const nm = names();
    const name = `${nm[0]} vs ${nm[1]} breakdown.${ext}`.replace(/[^\w .-]+/g, '');
    const file = new File([blob], name, { type: blob.type });
    if (navigator.canShare?.({ files: [file] }) && /iPhone|iPad|Android/i.test(navigator.userAgent)) {
      try { await navigator.share({ files: [file], title: name }); return; } catch { /* fall back to download */ }
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
  }

  function watch(ex) {
    video.currentTime = Math.max(0, ex.start - 0.8);
    watchEnd = ex.end + 0.8;
    video.playbackRate = 0.5;
    video.play().catch(() => {});
    $('player').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function fill(text, nm) {
    // {0} / {1} -> coloured fighter names
    const frag = document.createDocumentFragment();
    text.split(/(\{[01]\})/).forEach((part) => {
      const m = part.match(/^\{([01])\}$/);
      if (m) { const s = document.createElement('b'); s.className = `f${m[1]}`; s.textContent = nm[m[1]]; frag.append(s); }
      else if (part) frag.append(part);
    });
    return frag;
  }

  function render() {
    const nm = names(), res = result;
    const R = $('results');
    R.hidden = false; R.replaceChildren();
    // timeline
    const tl = $('timeline'); tl.hidden = false; tl.replaceChildren();
    const dur = (Number.isFinite(video.duration) && video.duration) || res.duration || 1;
    for (const ex of res.exchanges) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'bd-ex';
      b.style.left = `${(ex.start / dur) * 100}%`; b.style.width = `${Math.max(1.2, ((ex.end - ex.start) / dur) * 100)}%`;
      b.title = `Exchange ${ex.n}`; b.setAttribute('aria-label', `Watch exchange ${ex.n}`);
      b.addEventListener('click', () => watch(ex));
      tl.append(b);
    }
    for (const p of res.punches) {
      const d = document.createElement('i'); d.className = `bd-dot f${p.f}`; d.style.left = `${(p.t / dur) * 100}%`;
      tl.append(d);
    }
    const lg = $('legend'); lg.hidden = false; lg.replaceChildren();
    nm.forEach((n, k) => { const s = document.createElement('span'); s.className = `f${k}`; s.textContent = n; lg.append(s); });

    // AI film study
    const film = document.createElement('div'); film.className = 'card film-card';
    film.innerHTML = `<p class="card-title">AI film study</p>
      <p class="sheet-note">The coach plays the clip, slows down and pauses on the key moments, and talks you through the
      positioning, the setups, the counters and the exits, with highlights drawn on the fighters.</p>
      <div class="row-btns"><button type="button" class="mini" data-f="play">▶ Play breakdown</button>
        <button type="button" class="mini" data-f="save">Save as video</button>
        <button type="button" class="mini" data-f="stop" hidden>Stop</button></div>
      <p class="film-status" data-f="status"></p>`;
    const F = (k) => film.querySelector(`[data-f="${k}"]`);
    F('play').addEventListener('click', () => filmStudy(false, F));
    F('save').addEventListener('click', () => filmStudy(true, F));
    F('stop').addEventListener('click', () => director?.stop());
    if (!video.currentSrc) { F('play').disabled = true; F('save').disabled = true; F('status').textContent = 'Load a clip to play the film study.'; }
    R.append(film);

    // the read
    const ins = document.createElement('div'); ins.className = 'card tip';
    ins.innerHTML = '<p class="card-title">The read</p>';
    const ul = document.createElement('ul');
    for (const line of insights(res, nm)) ul.append(Object.assign(document.createElement('li'), { textContent: line }));
    ins.append(ul); R.append(ins);

    // per fighter
    const grid = document.createElement('div'); grid.className = 'bd-fighters';
    res.stats.forEach((s, k) => {
      const c = document.createElement('div'); c.className = 'card bd-fighter';
      const h = document.createElement('p'); h.className = `card-title f${k}`; h.textContent = nm[k];
      const st = document.createElement('div'); st.className = 'bd-stats';
      for (const [v, l] of [[s.total, 'Punches'], [s.led, 'Led'], [s.counters, 'Counters'], [s.bodyPct == null ? '–' : `${Math.round(s.bodyPct * 100)}%`, 'Body']]) {
        const d = document.createElement('div'); d.className = 'stat';
        d.innerHTML = `<span class="v"></span><span class="k"></span>`;
        d.firstChild.textContent = v; d.lastChild.textContent = l; st.append(d);
      }
      const mix = document.createElement('div'); mix.className = 'bars';
      const max = Math.max(1, ...Object.values(s.by));
      for (const l of Object.keys(NAMES)) {
        if (!s.by[l]) continue;
        const row = document.createElement('div'); row.className = 'brow';
        const n = document.createElement('span'); n.className = 'bname'; n.textContent = NAMES[l];
        const b = document.createElement('div'); b.className = 'bar';
        const i = document.createElement('i'); i.style.width = `${(s.by[l] / max) * 100}%`; b.append(i);
        const v = document.createElement('span'); v.className = 'bn'; v.textContent = s.by[l];
        row.append(n, b, v); mix.append(row);
      }
      c.append(h, st, mix);
      grid.append(c);
    });
    R.append(grid);

    // exchanges
    const t = document.createElement('p'); t.className = 'section-title';
    t.textContent = res.exchanges.length ? `${res.exchanges.length} exchange${res.exchanges.length === 1 ? '' : 's'}` : 'No exchanges found';
    R.append(t);
    for (const ex of res.exchanges) {
      const c = document.createElement('article'); c.className = 'card bd-card';
      const top = document.createElement('div'); top.className = 'bd-card-top';
      const h = document.createElement('div');
      h.innerHTML = '<p class="bd-when num"></p><p class="bd-title"></p>';
      h.firstChild.textContent = `${fmt(ex.start)} · ${(ex.end - ex.start).toFixed(1)} s`;
      h.lastChild.replaceChildren(document.createTextNode(`Exchange ${ex.n} · `), fill(`{${ex.init}} leads`, nm));
      const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'mini'; btn.textContent = 'Watch ½×';
      btn.addEventListener('click', () => watch(ex));
      top.append(h, btn);
      const seq = document.createElement('ol'); seq.className = 'bd-seq';
      for (const p of ex.punches) {
        const li = document.createElement('li'); li.className = `f${p.f}`;
        li.textContent = NAMES[p.label] + (p.target === 'body' ? ' ↓' : '');
        li.title = `${nm[p.f]} · ${fmt(p.t)}`;
        seq.append(li);
      }
      const notes = document.createElement('ul'); notes.className = 'bd-notes';
      for (const n of ex.notes) {
        const li = document.createElement('li'); li.className = `n-${n.k}`;
        li.append(fill(n.text, nm)); notes.append(li);
      }
      c.append(top, seq, notes);
      R.append(c);
    }
    const legend = document.createElement('p'); legend.className = 'hint-text';
    legend.textContent = '↓ = to the body. Lead/rear is read from which shoulder is nearer the opponent.';
    R.append(legend);
  }

  return {
    hide() { if (!processing) { video.pause(); } director?.stop(); },
    // Developer test hook: load a video URL with ready-made frames (skips pose tracking).
    async _testVideo(src, fs) {
      video.src = src; video.controls = true; video.muted = true;
      await new Promise((ok) => { video.onloadeddata = ok; });
      frames = fs; result = analyse(fs);
      $('player').hidden = false; $('progress').hidden = true;
      startOverlay(); render(); return result;
    },
    // Developer test hook: show the breakdown for ready-made frames (no video).
    _test(fs) { frames = fs; result = analyse(fs); $('player').hidden = false; $('progress').hidden = true; render(); return result; },
  };
}

function fmt(s) { s = Math.max(0, Math.floor(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }
