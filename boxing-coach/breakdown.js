// Fight breakdown: upload a clip of two fighters, track both with MediaPipe,
// detect and classify each punch, then group them into exchanges and describe
// who led, the setups, the counters and how each exchange ended.
// Everything runs on this device; the clip is never uploaded.

import { FilesetResolver, PoseLandmarker } from './vendor/vision_bundle.mjs';
import { NAMES, ROLE, matchCombo } from './personal.js';
import { voice } from './voice.js';
import {
  BONES, COLORS, REF_COLOR, pt, mid, d2, center, torso, elbowAngle, packPose, box, Sampler, Tracker, isDown, refScore,
  smoothIdentities, KEEP, sigDist, stanceVote,
} from './people.js';

export { BONES, COLORS, pt, mid, center, torso, packPose, Tracker };

const A = {
  SPEED: 2.4,         // wrist speed (torso lengths / s) that can start a punch
  NOISE_MULT: 3.5,    // ...and at least this many times the fighter's usual hand movement
  PEAK_WIN: 0.12,     // s either side a peak must beat
  REFRACTORY: 0.22,   // s between punches from the same hand
  MIN_REACH: 0.5,     // fist must get at least this far from its shoulder (torso lengths); hooks look short side-on
  MIN_TRAVEL: 0.45,   // fist must travel at least this far during the punch (torso lengths)
  MIN_APPROACH: 0.25, // ...and end at least this much closer to the opponent than it started
  EXCHANGE_GAP: 1.2,  // s of quiet that ends an exchange
  RUN_GAP: 0.8,       // s between punches that still counts as one combination
  COUNTER_WIN: 0.6,   // s after an opponent's punch that counts as a counter
  MOVE: 0.35,         // torso lengths of travel that counts as stepping in / out
  DOWN_MIN: 0.5,      // s on the canvas that counts as a knockdown
  COUNT_PAUSE: 4,     // s after a knockdown when punches aren't counted (the count)
};

// A punch type is a shape (straight, hook, uppercut) thrown with the lead or rear hand. The shape comes
// from how the fist moves; lead or rear comes from the fighter's stance, never from the shape.
const SHAPE = (l) => (l === 'jab' || l === 'cross' ? 'straight' : l.split('_')[1]);
const withRole = (shape, role) => (shape === 'straight' ? (role === 'lead' ? 'jab' : 'cross') : `${role}_${shape}`);

// ---------------------------------------------------------------- analysis
// frames: [{ t, poses: [poseA|null, poseB|null], ref: pose|null, cut: bool }]  (pose = packPose output)
export function analyse(frames, { model = null, overrides = null, net = punchNet } = {}) {
  const series = [0, 1].map((k) => frames.filter((f) => f.poses[k]).map((f) => ({ t: f.t, p: f.poses[k] })));
  const headMotion = [null, null];
  const refSeries = frames.filter((f) => f.ref).map((f) => ({ t: f.t, p: f.ref }));
  const nearest = (k, t, win = 0.25) => {
    const s = series[k];
    if (!s.length) return null;
    let lo = 0, hi = s.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (s[m].t < t) lo = m + 1; else hi = m; }
    const c = s[lo], b = s[lo - 1];
    const best = b && Math.abs(b.t - t) < Math.abs(c.t - t) ? b : c;
    return Math.abs(best.t - t) < win ? best : null;
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
  // A knockdown lasts while the fighter is lying there or out of sight (the pose model often loses them on
  // the canvas). It ends once they're seen back on their feet for a few frames: one stray frame isn't enough.
  const downUntil = (k, from) => {
    let tEnd = from, up = 0;
    for (const f of frames) {
      if (f.t <= from) continue;
      if (f.t > from + 15) break;
      if (f.poses[k] && !isDown(f.poses[k], torsoAvg[k])) { if (++up >= 4) break; continue; }
      if (f.poses[k]) up = 0;
      tEnd = f.t;
    }
    return tEnd;
  };
  // How far fighter k moved towards the opponent between t0 and t1 (torso lengths).
  const advance = (k, t0, t1) => {
    const a = xAt(k, t0), b = xAt(k, t1);
    if (a == null || b == null) return 0;
    return ((b - a) * dirAt(k, t0)) / torsoAvg[k];
  };

  // Stance: which hand leads. Read over the whole clip from the shoulders, hips, knees and feet together:
  // one moment is noisy (a punch turns the shoulders, side-on bodies overlap), but a fighter's stance
  // hardly ever changes. You can also set it, and it's remembered for that fighter's name.
  const stance = [0, 1].map((k) => {
    const set = overrides?.stance?.[k];
    if (set) return { lead: set === 'southpaw' ? 'right' : 'left', set: true, conf: 1, local: [] };
    const local = [];
    for (const f of series[k]) {
      const op = nearest(1 - k, f.t, 0.3);
      if (!op || isDown(f.p, torsoAvg[k])) continue;
      const s = stanceVote(f.p, op.p);
      if (s) local.push({ t: f.t, s });
    }
    const share = local.length ? local.filter((x) => x.s > 0).length / local.length : 0.5;
    return { lead: share >= 0.5 ? 'left' : 'right', set: false, conf: Math.abs(share - 0.5) * 2, local };
  });
  // a switch-hitter: a clear run of the other stance around this moment overrules the clip-wide read
  const leadAt = (k, t) => {
    const st = stance[k];
    if (st.set) return st.lead;
    const near = st.local.filter((x) => Math.abs(x.t - t) <= 3);
    if (near.length >= 20) {
      const lf = near.filter((x) => x.s > 0).length / near.length;
      if (lf >= 0.85) return 'left';
      if (lf <= 0.15) return 'right';
    }
    return st.lead;
  };

  let knockdowns = findKnockdowns();
  // knockdowns you marked or removed by hand win over the detector
  if (overrides?.kdOff) knockdowns = knockdowns.filter((kd) => !overrides.kdOff.some((t) => Math.abs(t - kd.t) < 1));
  for (const m of overrides?.kdAdd || []) {
    if (knockdowns.some((kd) => Math.abs(kd.t - m.t) < 1)) continue;
    knockdowns.push({ f: m.f, t: m.t, tEnd: Math.max(m.t + 1, downUntil(m.f, m.t + 0.5)), marked: true });
  }
  knockdowns.sort((a, b) => a.t - b.t);
  const refIn = findRefIn();
  // no punches are counted while a fighter is down and being counted, or while the referee is between them
  const quiet = [...knockdowns.map((kd) => [kd.t, kd.tEnd + A.COUNT_PAUSE]), ...refIn.map((r) => [r.t, r.tEnd])];
  const inQuiet = (t) => quiet.some(([a, b]) => t >= a && t <= b);
  let punches = [];
  const feintCands = [];
  for (const k of [0, 1]) punches.push(...detect(k));
  punches = punches.filter((p) => !inQuiet(p.t));
  // the pre-trained BoxingVI network decides the punch type (and can veto things that weren't punches)
  if (net) {
    punches = punches.filter((p) => {
      const probs = punchProbs(net, p.feat);
      if (!probs) return true;
      if ((probs.none || 0) >= 0.8) return false;        // only drop it when the network is very sure
      // the network picks the shape; the stance decides lead or rear
      const shapes = {};
      for (const [l, v] of Object.entries(probs)) if (ROLE[l]) shapes[SHAPE(l)] = (shapes[SHAPE(l)] || 0) + v;
      const [shape, conf] = Object.entries(shapes).sort((a, b) => b[1] - a[1])[0];
      p.net = { shape, conf };
      if (conf >= 0.45) p.label = withRole(shape, p.role);
      return true;
    });
  }
  // what you've taught it from corrections overrides the built-in rules
  if (model) punches = punches.filter((p) => applyModel(model, p));
  // punches it missed that you added by hand (the hand comes from the punch type and the stance)
  for (const m of overrides?.add || []) {
    if (punches.some((p) => p.f === m.f && Math.abs(p.t - m.t) < 0.25)) continue;
    const role = ROLE[m.label], lead = leadAt(m.f, m.t);
    const hand = role === 'lead' ? lead : lead === 'left' ? 'right' : 'left';
    punches.push({ f: m.f, t: m.t, tEnd: m.t + 0.2, hand, role, label: m.label, target: m.target || 'head', outcome: m.outcome || null, added: true, corrected: true });
  }
  // corrections you made on this clip win over everything
  if (overrides) {
    punches = punches.filter((p) => {
      const o = overrides[punchKey(p)];
      if (!o) return true;
      if (o.none) return false;
      if (o.label) { p.label = o.label; p.role = ROLE[o.label]; }
      if (o.outcome) p.outcome = o.outcome;
      p.corrected = true;
      return true;
    });
  }
  punches.sort((a, b) => a.t - b.t);

  // feints only count when they do something: the opponent reacts, or a real punch follows
  const motion = [0, 1].map(motionSeries);
  const motionNoise = motion.map((m) => { const v = m.map((x) => x.v).filter((x) => x > 0.05).sort((a, b) => a - b); return v.length ? v[v.length >> 1] : 0.5; });
  const feints = feintCands.filter((c) => !inQuiet(c.t)).map((c) => {
    const o = 1 - c.f;
    const peak = Math.max(0, ...motion[o].filter((x) => x.t > c.t && x.t <= c.t + 0.45).map((x) => x.v));
    // the opponent throwing their own punch isn't a reaction to the feint
    const oppPunched = punches.some((p) => p.f === o && p.t > c.t - 0.1 && p.t <= c.t + 0.5);
    c.reaction = !oppPunched && peak > Math.max(1.6, 2.5 * motionNoise[o]);
    c.followedBy = punches.find((p) => p.f === c.f && p.t > c.t + 0.1 && p.t - c.t <= 0.9) || null;
    return c;
  }).filter((c) => c.reaction || c.followedBy).sort((a, b) => a.t - b.t);
  const levels = [0, 1].flatMap(findLevelChanges).filter((l) => !inQuiet(l.t)).sort((a, b) => a.t - b.t);
  for (const l of levels) l.followedBy = punches.find((p) => p.f === l.f && p.t >= l.t - 0.1 && p.t - l.t <= 0.9) || null;

  const exchanges = group(punches);
  for (const ex of exchanges) describe(ex);

  for (const kd of knockdowns) {
    kd.cause = [...punches].reverse().find((p) => p.f === 1 - kd.f && p.t <= kd.t + 0.1 && kd.t - p.t <= 2.5) || null;
    kd.ex = exchanges.find((e) => kd.t >= e.start - 0.2 && kd.t <= e.end + 2.5) || null;
    if (!kd.cause) {
      // the punch wasn't clear enough to count (blurred or hidden glove): take the opponent's fastest hand just before the fall
      const o = 1 - kd.f, s = series[o];
      let best = null;
      for (let i = 1; i < s.length; i++) {
        const a = s[i - 1], b = s[i], dt = b.t - a.t;
        if (b.t < kd.t - 1.2 || b.t > kd.t + 0.15 || dt <= 0 || dt > 0.25) continue;
        for (const [hand, j] of [['left', 15], ['right', 16]]) {
          const v = d2(pt(a.p, j), pt(b.p, j)) / dt / torsoAvg[o];
          if (!best || v > best.v) best = { v, hand, t: b.t };
        }
      }
      if (best && best.v > 6) kd.hiddenHand = { hand: best.hand, role: roleAt(o, best.hand, best.t), t: best.t };
    }
    if (!kd.ex) {
      // a knockdown always gets its own entry, even when the punch that caused it wasn't visible
      kd.ex = { start: Math.max(0, kd.t - 1.5), end: kd.tEnd, punches: [], runs: [], init: 1 - kd.f, count: [0, 0], notes: [] };
      const h = kd.hiddenHand;
      kd.ex.notes.push({ k: 'lead', text: h ? `{${1 - kd.f}} lands a ${h.role} hand that the camera barely catches` : `{${1 - kd.f}} lands the shot that does it, but the punch is hidden from this camera angle` });
      exchanges.push(kd.ex);
      exchanges.sort((a, b) => a.start - b.start);
      exchanges.forEach((ex, i) => { ex.n = i + 1; });
    }
    {
      // falling over isn't "giving ground"
      kd.ex.notes = kd.ex.notes.filter((n) => !(n.k === 'end' && n.text.startsWith(`{${kd.f}}`)));
      kd.ex.notes.push({ k: 'knockdown', text: `{${kd.f}} goes down${kd.cause ? ` from ${pName(kd.cause)}` : ''}. Knockdown!` });
      kd.ex.knockdown = kd;
    }
  }
  for (const r of refIn) {
    const ex = exchanges.find((e) => r.t >= e.start && r.t <= e.end + 2.5);
    if (ex && !ex.refNote) {
      ex.refNote = true;
      ex.notes.push({ k: 'ref', text: ex.knockdown ? 'The referee steps in and starts the count' : 'The referee steps in to separate them' });
    }
  }
  // who won each exchange: clean punches landed (power shots count a bit more), a knockdown wins it outright
  for (const ex of exchanges) {
    const wt = (p) => (p.label === 'jab' ? 1 : 1.5);
    const score = [0, 1].map((k) => ex.punches.filter((p) => p.f === k && p.outcome === 'landed').reduce((s, p) => s + wt(p), 0));
    ex.landed = [0, 1].map((k) => ex.punches.filter((p) => p.f === k && p.outcome === 'landed').length);
    ex.winner = ex.knockdown ? 1 - ex.knockdown.f : Math.abs(score[0] - score[1]) < 0.5 ? null : score[0] > score[1] ? 0 : 1;
    // the winner you picked for this exchange wins over the scoring
    const pickW = Object.entries(overrides?.win || {}).find(([t]) => Math.abs(Number(t) - ex.start) < 0.6);
    if (pickW) {
      ex.winner = pickW[1] === 'even' ? null : Number(pickW[1]);
      ex.winnerFixed = true;
      ex.notes.push({ k: 'result', text: ex.winner == null ? 'An even exchange (your call)' : `{${ex.winner}} wins the exchange (your call)` });
      continue;
    }
    const w = ex.winner;
    ex.notes.push({
      k: 'result',
      text: w == null
        ? (ex.landed[0] + ex.landed[1] ? `Even exchange: ${ex.landed[0]} landed each way` : 'Nothing clean lands: an even exchange')
        : ex.knockdown ? `{${w}} wins it with the knockdown`
          : `{${w}} wins the exchange, landing ${ex.landed[w]} to ${ex.landed[1 - w]}`,
    });
  }

  const duration = frames.length ? frames[frames.length - 1].t - frames[0].t : 0;
  return {
    punches, exchanges, duration, knockdowns, refIn, feints, levels,
    stance: stance.map((s) => ({ lead: s.lead, set: s.set, conf: s.conf, name: s.lead === 'left' ? 'orthodox' : 'southpaw' })),
    stats: [0, 1].map((k) => ({
      ...fighterStats(k, punches, exchanges, duration, knockdowns),
      feints: feints.filter((x) => x.f === k).length,
      feintReactions: feints.filter((x) => x.f === k && x.reaction).length,
      levels: levels.filter((x) => x.f === k).length,
    })),
    coverage: series.map((s) => s.length / Math.max(1, frames.length)),
    refCoverage: refSeries.length / Math.max(1, frames.length),
  };

  function findKnockdowns() {
    const out = [];
    for (const k of [0, 1]) {
      const s = series[k];
      if (s.length < 5) continue;
      const flags = s.map((f) => ({ t: f.t, d: isDown(f.p, torsoAvg[k]) }));
      const segs = [];
      let seg = null;
      for (const x of flags) {
        if (!x.d) continue;
        if (seg && x.t - seg.end <= 0.3) seg.end = x.t;
        else { if (seg) segs.push(seg); seg = { start: x.t, end: x.t }; }
      }
      if (seg) segs.push(seg);
      for (const g of segs) {
        if (g.end - g.start < A.DOWN_MIN) continue;
        // must have been on their feet just before, so a clip starting mid-count isn't a new knockdown
        if (!flags.some((x) => !x.d && x.t < g.start && x.t >= g.start - 2)) continue;
        // they stay down for as long as they're lying there or out of sight (the pose model often loses them)
        out.push({ f: k, t: g.start, tEnd: downUntil(k, g.end) });
      }

      // The pose model often loses a fighter who is lying on the canvas. So also count: a fast fall
      // (the body drops well over half its height in under 0.7 s, no camera cut), then the fighter is
      // lying down or barely seen for the next 1.5 s while the opponent is still there.
      const L = torsoAvg[k];
      const cutAt = (t0, t1) => frames.some((f) => f.cut && f.t >= t0 && f.t <= t1);
      for (let i = 0; i < s.length; i++) {
        const a = s[i], ya = center(a.p)[1];
        if (isDown(a.p, L) || out.some((kd) => a.t >= kd.t - 1 && a.t <= kd.tEnd + 2)) continue;
        let fallEnd = null;
        for (let j = i + 1; j < s.length && s[j].t - a.t <= 0.7; j++) {
          if (center(s[j].p)[1] - ya >= 0.55 * L || isDown(s[j].p, L)) { fallEnd = s[j]; break; }
        }
        if (!fallEnd || cutAt(a.t - 0.1, fallEnd.t + 0.3)) continue;
        const after = frames.filter((f) => f.t > fallEnd.t && f.t <= fallEnd.t + 1.5);
        if (after.length < 10) continue;
        const seenUp = after.filter((f) => f.poses[k] && !isDown(f.poses[k], L)).length / after.length;
        const oppSeen = after.filter((f) => f.poses[1 - k]).length / after.length;
        if (seenUp > 0.25 || oppSeen < 0.5) continue;
        // how long they stay down (or out of sight)
        const tEnd = Math.max(fallEnd.t + 1.5, downUntil(k, fallEnd.t + 1.5));
        out.push({ f: k, t: a.t, tEnd, inferred: true });
        i = s.findIndex((x) => x.t > tEnd);
        if (i < 0) break;
      }
    }
    return out.sort((a, b) => a.t - b.t);
  }

  // The referee standing between the fighters, close to them: stepping in to break or to stop it.
  function findRefIn() {
    const segs = [];
    let seg = null;
    for (const r of refSeries) {
      const a = nearest(0, r.t), b = nearest(1, r.t);
      if (!a || !b) continue;
      const ca = center(a.p), cb = center(b.p), cr = center(r.p);
      const L = (torso(a.p) + torso(b.p)) / 2;
      const between = cr[0] > Math.min(ca[0], cb[0]) + 0.2 * L && cr[0] < Math.max(ca[0], cb[0]) - 0.2 * L;
      const close = Math.min(d2(cr, ca), d2(cr, cb)) < 1.6 * L;
      // and at the same depth as them: a referee standing further back only looks "between" on screen
      // (he'd look smaller and stand higher up the picture)
      const hipY = (k) => mid(pt(k.p, 23), pt(k.p, 24))[1];
      const sameDepth = torso(r.p) > 0.75 * L && Math.abs(hipY(r) - (hipY(a) + hipY(b)) / 2) < 0.6 * L;
      if (!(between && close && sameDepth)) continue;
      if (seg && r.t - seg.tEnd <= 0.3) seg.tEnd = r.t;
      else { if (seg) segs.push(seg); seg = { t: r.t, tEnd: r.t }; }
    }
    if (seg) segs.push(seg);
    return segs.filter((g) => g.tEnd - g.t >= 0.3);
  }

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
      // this fighter's usual hand movement (guard adjustments, bouncing): a punch must stand well clear of it
      const moving = sv.filter((x) => x > 0.05).sort((a, b) => a - b);
      const noise = moving.length ? moving[moving.length >> 1] : 0;
      const th = Math.max(A.SPEED, A.NOISE_MULT * noise);
      const wMi = wristMi[sd];
      let lastT = -Infinity;
      for (let i = 1; i < rows.length - 1; i++) {
        const t = rows[i].t;
        if (sv[i] < th || rows[i].r[sd].vis < 0.5 || t - lastT < A.REFRACTORY) continue;
        let isPeak = true;
        for (let j = i - 1; j >= 0 && t - rows[j].t <= A.PEAK_WIN; j--) if (sv[j] > sv[i]) { isPeak = false; break; }
        for (let j = i + 1; isPeak && j < rows.length && rows[j].t - t <= A.PEAK_WIN; j++) if (sv[j] >= sv[i]) isPeak = false;
        if (!isPeak) continue;
        if (at(t + 0.08).r[sd].reach - at(t - 0.08).r[sd].reach <= 0.05) continue;  // must be extending
        // furthest point of the punch
        let far = rows[i];
        for (const r of rows) if (r.t >= t - 0.05 && r.t <= t + 0.2 && r.r[sd].reach > far.r[sd].reach) far = r;
        if (far.r[sd].reach < A.MIN_REACH || far.r[sd].vis < 0.5) continue;
        const start = at(t - 0.15);
        // a real punch travels a fair way...
        const travel = Math.hypot(far.r[sd].rel[0] - start.r[sd].rel[0], far.r[sd].rel[1] - start.r[sd].rel[1]);
        if (travel < A.MIN_TRAVEL) continue;
        // ...towards an opponent who is actually there (gestures, guard shuffles and the referee's arms don't).
        // The pose model can lose the opponent for a few frames as a big shot lands, so allow half a second.
        const opp0 = nearest(1 - k, start.t, 0.5), opp1 = nearest(1 - k, far.t, 0.5);
        if (!opp0 || !opp1) continue;
        const aim = (op) => { const hd = pt(op.p, 0); return hd[2] > 0.3 ? hd : center(op.p); };
        const approach = (d2(pt(start.p, wMi), aim(opp0)) - d2(pt(far.p, wMi), aim(opp1))) / torso(far.p);
        if (approach < A.MIN_APPROACH) continue;
        lastT = t;
        const p = classify(k, sd, t, start, far);
        p.feat = punchFeature(at, t, dirAt(k, t), sd, far, p.role);
        out.push(p);
      }

      // feints: a sharp, short move towards the opponent that comes straight back instead of landing
      const punchTimes = out.filter((p) => p.hand === sd).map((p) => p.t);
      let lastF = -Infinity;
      for (let i = 1; i < rows.length - 1; i++) {
        const t = rows[i].t;
        if (sv[i] < Math.max(1.5, 0.5 * th) || rows[i].r[sd].vis < 0.5 || t - lastF < 0.5) continue;
        if (punchTimes.some((x) => Math.abs(x - t) < 0.4)) continue;   // part of a real punch
        let isPeak = true;
        for (let j = i - 1; j >= 0 && t - rows[j].t <= A.PEAK_WIN; j--) if (sv[j] > sv[i]) { isPeak = false; break; }
        for (let j = i + 1; isPeak && j < rows.length && rows[j].t - t <= A.PEAK_WIN; j++) if (sv[j] >= sv[i]) isPeak = false;
        if (!isPeak) continue;
        const start = at(t - 0.15);
        let far = rows[i];
        for (const r of rows) if (r.t >= t - 0.05 && r.t <= t + 0.2 && r.r[sd].reach > far.r[sd].reach) far = r;
        const travel = Math.hypot(far.r[sd].rel[0] - start.r[sd].rel[0], far.r[sd].rel[1] - start.r[sd].rel[1]);
        if (travel < 0.15 || travel >= A.MIN_TRAVEL) continue;          // too small, or a full punch
        // must start from a settled guard (not hands coming back after a flinch or another move)
        let before = 0;
        for (let j = i - 1; j >= 0 && t - rows[j].t <= 0.4; j--) if (t - rows[j].t >= 0.15) before = Math.max(before, sv[j]);
        if (before > 0.5 * sv[i]) continue;
        const back = at(far.t + 0.25);
        if (back.r[sd].reach > far.r[sd].reach - 0.08) continue;         // must come straight back
        const opp0 = nearest(1 - k, start.t), opp1 = nearest(1 - k, far.t);
        if (!opp0 || !opp1) continue;
        const aim = (op) => { const hd = pt(op.p, 0); return hd[2] > 0.3 ? hd : center(op.p); };
        const approach = (d2(pt(start.p, wMi), aim(opp0)) - d2(pt(far.p, wMi), aim(opp1))) / torso(far.p);
        if (approach < 0.08) continue;                                    // aimed at the opponent
        lastF = t;
        feintCands.push({ f: k, t, tEnd: far.t, hand: sd, role: roleAt(k, sd, t) });
      }
    }
    return out;
  }

  // A compact description of a punch's shape, for learning from corrections: both hands' paths
  // (relative to the shoulders, facing the opponent) from just before to just after the punch.
  function punchFeature(at, t, dir, hand, far, role) {
    const other = hand === 'left' ? 'right' : 'left';
    const f = [];
    for (let i = 0; i < 10; i++) {
      const r = at(t - 0.2 + i * 0.05);
      for (const sd of [hand, other]) f.push(r.r[sd].rel[0] * dir, r.r[sd].rel[1], r.r[sd].reach);
    }
    f.push(elbowAngle(far.p, hand) / 180, role === 'lead' ? 1 : 0);
    return f.map((v) => Math.max(-4, Math.min(4, v)));
  }

  // Did it land? Glove ends on the head or body, ideally with the head snapping back; or it ends on
  // the opponent's gloves/forearms (blocked); otherwise it missed.
  function outcomeOf(p, far, k) {
    const op = nearest(1 - k, far.t);
    if (!op) return null;
    const L = torso(op.p), glove = pt(far.p, p.hand === 'left' ? 15 : 16);
    const nose = pt(op.p, 0);
    const sh = mid(pt(op.p, 11), pt(op.p, 12)), hp = mid(pt(op.p, 23), pt(op.p, 24));
    const onHead = nose[2] > 0.3 && d2(glove, nose) < 0.5 * L;
    const onBody = Math.abs(glove[0] - (sh[0] + hp[0]) / 2) < 0.45 * L && glove[1] > sh[1] && glove[1] < hp[1] + 0.1 * L;
    const guard = [13, 14, 15, 16].map((mi) => pt(op.p, mi)).filter((q) => q[2] > 0.3);
    const dGuard = guard.length ? Math.min(...guard.map((q) => d2(glove, q))) / L : 9;
    const snap = headSnap(1 - k, far.t - 0.05, far.t + 0.25);
    const hit = (p.target === 'body' ? onBody : onHead) || (snap && (onHead || onBody));
    if (hit && (dGuard > 0.25 || snap)) return 'landed';
    if (dGuard < 0.3) return 'blocked';
    return 'missed';
  }

  // The opponent's head jerking (relative to their hips) right after a punch arrives.
  function headSnap(k, t0, t1) {
    if (!headMotion[k]) {
      const s = series[k], m = [];
      for (let i = 1; i < s.length; i++) {
        const a = s[i - 1], b = s[i], dt = b.t - a.t;
        if (dt <= 0 || dt > 0.12) continue;
        const na = pt(a.p, 0), nb = pt(b.p, 0);
        if (na[2] < 0.4 || nb[2] < 0.4) continue;
        const ha = mid(pt(a.p, 23), pt(a.p, 24)), hb = mid(pt(b.p, 23), pt(b.p, 24));
        m.push({ t: b.t, v: Math.hypot((nb[0] - hb[0]) - (na[0] - ha[0]), (nb[1] - hb[1]) - (na[1] - ha[1])) / torso(b.p) / dt });
      }
      const v = m.map((x) => x.v).sort((a, b) => a - b);
      headMotion[k] = { m, noise: v.length ? v[v.length >> 1] : 0.5 };
    }
    const { m, noise } = headMotion[k];
    const peak = Math.max(0, ...m.filter((x) => x.t >= t0 && x.t <= t1).map((x) => x.v));
    return peak > Math.max(1.8, 3 * noise);
  }

  // lead or rear hand, from the fighter's stance
  function roleAt(k, hand, t) {
    return hand === leadAt(k, t) ? 'lead' : 'rear';
  }

  // How much a fighter is moving (head and hands, relative to their hips), torso lengths / s.
  function motionSeries(k) {
    const s = series[k];
    const out = [];
    for (let i = 1; i < s.length; i++) {
      const a = s[i - 1], b = s[i], dt = b.t - a.t;
      if (dt <= 0 || dt > 0.12) continue;
      const L = torso(b.p), ha = mid(pt(a.p, 23), pt(a.p, 24)), hb = mid(pt(b.p, 23), pt(b.p, 24));
      let v = 0;
      for (const mi of [0, 15, 16]) {
        const pa = pt(a.p, mi), pb = pt(b.p, mi);
        if (pa[2] < 0.4 || pb[2] < 0.4) continue;
        v = Math.max(v, Math.hypot((pb[0] - hb[0]) - (pa[0] - ha[0]), (pb[1] - hb[1]) - (pa[1] - ha[1])) / L / dt);
      }
      out.push({ t: b.t, v });
    }
    return out;
  }

  // Level changes: the fighter drops their height fast, against their own stance and the opponent's hips.
  function findLevelChanges(k) {
    const s = series[k];
    if (s.length < 10) return [];
    const L = torsoAvg[k];
    const rows = [];
    for (const f of s) {
      const nose = pt(f.p, 0), hp = mid(pt(f.p, 23), pt(f.p, 24));
      if (nose[2] < 0.5 || pt(f.p, 23)[2] < 0.5) continue;
      if (isDown(f.p, L)) continue;
      const op = nearest(1 - k, f.t);
      if (!op) continue;   // a level change only matters with the opponent in front of them
      const opHip = pt(op.p, 23)[2] > 0.5 ? mid(pt(op.p, 23), pt(op.p, 24))[1] : null;
      // head height above the shoulders, measured in this frame's own body size, so camera zoom doesn't fake it
      const sh = mid(pt(f.p, 11), pt(f.p, 12));
      rows.push({ t: f.t, h1: (sh[1] - nose[1]) / torso(f.p), hip: hp[1], opHip });
    }
    const med = (arr) => { const v = arr.filter((x) => x != null).sort((a, b) => a - b); return v.length >= 5 ? v[v.length >> 1] : null; };
    // How far below normal this fighter is: bending at the waist (head towards hips), or bending the
    // knees (hips lower in the picture). For the knees, a camera tilt moves both fighters the same way,
    // so take the smaller of "my drop" and "my drop minus the opponent's": the opponent standing up
    // or the camera moving doesn't count as me dipping.
    const depthAt = (x, b) => {
      const waist = b.h1 != null ? b.h1 - x.h1 : 0;
      let knees = 0;
      if (b.hip != null) {
        const mine = (x.hip - b.hip) / L;
        knees = x.opHip != null && b.opHip != null ? Math.min(mine, mine - (x.opHip - b.opHip) / L) : 0;
      }
      return Math.max(waist, knees);
    };
    const out = [];
    let lastT = -Infinity;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      if (r.t - lastT < 1.0) continue;
      const past = rows.filter((x) => x.t >= r.t - 2 && x.t <= r.t - 0.5);
      const b = { h1: med(past.map((x) => x.h1)), hip: med(past.map((x) => x.hip)), opHip: med(past.map((x) => x.opHip)) };
      const depth = depthAt(r, b);
      if (depth < 0.22) continue;
      // fast: close to normal height half a second earlier
      const before = rows.filter((x) => x.t >= r.t - 0.5 && x.t <= r.t - 0.35);
      if (!before.some((x) => depthAt(x, b) < 0.08)) continue;
      // held at least briefly (not a tracking blip)
      const hold = rows.filter((x) => x.t > r.t && x.t <= r.t + 0.15);
      if (!hold.length || !hold.every((x) => depthAt(x, b) >= 0.15)) continue;
      lastT = r.t;
      out.push({ f: k, t: r.t, depth });
    }
    return out;
  }

  function classify(k, hand, t, start, far) {
    const dir = dirAt(k, t);
    const disp = [far.r[hand].rel[0] - start.r[hand].rel[0], far.r[hand].rel[1] - start.r[hand].rel[1]];
    const toward = disp[0] * dir, up = -disp[1];
    const elbow = elbowAngle(far.p, hand);
    const role = roleAt(k, hand, t);
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
    const res = { f: k, t, hand, role, label, target, tEnd: far.t };
    res.outcome = outcomeOf(res, far, k);
    return res;
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
      // what came just before this combination: a feint, a level change
      const first = r.punches[0];
      const fe = feints.find((x) => x.f === r.f && x.followedBy === first);
      if (fe) {
        ex.feint = ex.feint || fe;
        const what = fe.role === 'lead' ? 'the jab' : 'the rear hand';
        notes.push({ k: 'feint', text: `${who} feints ${what} first${fe.reaction ? `, and {${1 - r.f}} reacts to it` : ''}, then throws ${pName(first)}` });
      }
      const lv = levels.find((x) => x.f === r.f && r.punches.includes(x.followedBy));
      if (lv) {
        ex.level = ex.level || lv;
        const p = lv.followedBy;
        notes.push({
          k: 'level',
          text: p.target === 'body' ? `${who} changes levels and digs ${pName(p)}`
            : /uppercut/.test(p.label) ? `${who} dips down, then comes up with ${pName(p)}`
              : `${who} changes levels to disguise ${pName(p)}`,
        });
      }

      // setups inside the run
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

// ---------------------------------------------------------------- pre-trained punch model
// A small neural network trained on the BoxingVI benchmark (Kumar et al., arXiv:2511.16524):
// thousands of labelled punches from 18 boxers. It reads the same punch shape the app computes
// (punchFeature) and says which of the six punches it was, or that it wasn't a punch at all.
// models/fight_punch_mlp.json holds weights only, no BoxingVI data.
let punchNet = null, punchNetLoading = null;
export function loadPunchModel() {
  if (!punchNetLoading) {
    punchNetLoading = fetch('./models/fight_punch_mlp.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => { punchNet = m && Array.isArray(m.layers) ? m : null; return punchNet; })
      .catch(() => null);
  }
  return punchNetLoading;
}
export function punchProbs(net, feat) {
  if (!net || !feat || feat.length !== net.inputs) return null;
  let h = feat.map((v, i) => (v - net.mean[i]) / net.std[i]);
  net.layers.forEach((L, li) => {
    const out = new Array(L.b.length);
    for (let o = 0; o < L.b.length; o++) {
      let s = L.b[o]; const row = L.w[o];
      for (let i = 0; i < row.length; i++) s += row[i] * h[i];
      out[o] = s;
    }
    h = li < net.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const mx = Math.max(...h), ex = h.map((v) => Math.exp(v - mx)), sum = ex.reduce((a, b) => a + b, 0);
  return Object.fromEntries(net.labels.map((l, i) => [l, ex[i] / sum]));
}

// ---------------------------------------------------------------- fight model (learned from corrections)
// Every time you correct (or confirm) a punch in the results, its shape is saved with the right
// answer. Once there are enough, new clips are checked against them: k-nearest-neighbours, the same
// idea as the Teach page, but for punches seen in fight footage. Stored in this browser only.
const FM_KEY = 'bc.fightModel.v1';
export function loadFightModel() {
  try { const m = JSON.parse(localStorage.getItem(FM_KEY) || 'null'); if (m && Array.isArray(m.ex)) return m; } catch { /* */ }
  return { ex: [] };
}
export function saveFightModel(m) {
  try { localStorage.setItem(FM_KEY, JSON.stringify(m)); return true; } catch { return false; }
}
export const punchKey = (p) => `${p.f}@${p.t.toFixed(2)}`;
function fdist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
// How far apart taught examples typically are (sets what "similar" means for this model).
function modelScale(m) {
  if (m._scale && m._n === m.ex.length) return m._scale;
  const ds = [];
  for (let i = 0; i < m.ex.length; i++) for (let j = i + 1; j < m.ex.length && ds.length < 4000; j++) {
    if (m.ex[i].f.length === m.ex[j].f.length) ds.push(fdist(m.ex[i].f, m.ex[j].f));
  }
  ds.sort((a, b) => a - b);
  m._scale = Math.max(0.05, ds.length ? ds[ds.length >> 1] : 2); m._n = m.ex.length;
  return m._scale;
}
// Returns false if the model says it wasn't a punch; may relabel it. Only punches that really look
// like something you taught it are changed, and the built-in call keeps a vote of its own.
function applyModel(m, p) {
  if (!p.feat || m.ex.length < 8) return true;
  const scale = modelScale(m);
  const near = m.ex.filter((e) => e.f.length === p.feat.length).map((e) => [fdist(e.f, p.feat), e.l])
    .filter(([d]) => d <= 1.5 * scale).sort((a, b) => a[0] - b[0]).slice(0, 5);
  if (!near.length) return true;
  // votes are for the shape (or "not a punch"): lead or rear always comes from the stance
  const key = (l) => (l === 'none' ? 'none' : SHAPE(l));
  const votes = { [key(p.label)]: 1 / (0.7 * scale + 0.05) }; let total = votes[key(p.label)];
  for (const [d, l] of near) { const w = 1 / (d + 0.05); votes[key(l)] = (votes[key(l)] || 0) + w; total += w; }
  const [best, v] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  if (v / total < 0.6) return true;
  if (best === 'none') return false;
  const lab = withRole(best, p.role);
  if (lab !== p.label) { p.label = lab; p.learned = true; }
  return true;
}

function pName(p, bare = false) {
  const n = NAMES[p.label].toLowerCase();
  const tgt = p.target === 'body' ? ' to the body' : '';
  return bare ? n + tgt : `${/^[aeiou]/.test(n) ? 'an' : 'a'} ${n}${tgt}`;
}

function fighterStats(k, punches, exchanges, duration, knockdowns = []) {
  const mine = punches.filter((p) => p.f === k);
  const scored = knockdowns.filter((kd) => kd.f !== k).length, suffered = knockdowns.filter((kd) => kd.f === k).length;
  const landed = mine.filter((p) => p.outcome === 'landed').length;
  const won = exchanges.filter((e) => e.winner === k).length;
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
    total: mine.length, by, led, counters, scored, suffered, landed, won,
    accuracy: mine.length ? landed / mine.length : null,
    bodyPct: tgt ? body / tgt : null,
    perMin: duration >= 10 ? mine.length / (duration / 60) : null,
    topPair: topPair && topPair[1] >= 2 ? { seq: topPair[0].split('>'), n: topPair[1] } : null,
  };
}

// Overall read of the clip, one line each.
export function insights(res, names) {
  const [a, b] = res.stats;
  const out = [];
  for (const kd of res.knockdowns || []) {
    out.push(`${names[1 - kd.f]} dropped ${names[kd.f]}${kd.cause ? ` with ${pName(kd.cause)}` : ''} at ${fmt(kd.t)}.`);
  }
  const n = res.exchanges.length;
  if (!n) return out.length ? out : ['No exchanges found. Try a clip where both fighters stay fully in frame, side-on to the camera.'];
  for (const k of [0, 1]) {
    const s = res.stats[k], nm = names[k];
    if (s.led >= Math.max(2, n * 0.6)) out.push(`${nm} led ${s.led} of ${n} exchanges: the one applying pressure.`);
    if (s.counters >= 2 && s.counters >= s.led) out.push(`${nm} countered ${s.counters} times: happy to let the other man lead and fire back.`);
    if (s.topPair) out.push(`${nm}'s go-to: ${s.topPair.seq.map((l) => NAMES[l].toLowerCase()).join(' → ')} (${s.topPair.n} times).`);
    if (s.total >= 6 && (s.by.jab || 0) / s.total >= 0.4) out.push(`${nm} lives off the jab (${Math.round(((s.by.jab || 0) / s.total) * 100)}% of punches).`);
    if (s.bodyPct != null && s.total >= 6 && s.bodyPct >= 0.3) out.push(`${nm} invests in the body (${Math.round(s.bodyPct * 100)}% of shots).`);
    if (s.feints >= 3) out.push(`${nm} feinted ${s.feints} times${s.feintReactions ? ` and drew ${s.feintReactions} reaction${s.feintReactions > 1 ? 's' : ''}` : ''}: setting traps rather than just throwing.`);
    if (s.levels >= 3) out.push(`${nm} changed levels ${s.levels} times, mixing up where the punches come from.`);
  }
  if (a.won + b.won >= 2 && a.won !== b.won) {
    const w = a.won > b.won ? 0 : 1;
    out.unshift(`${names[w]} won more of the exchanges (${res.stats[w].won} to ${res.stats[1 - w].won}).`);
  }
  if (a.total >= 5 && b.total >= 5 && a.accuracy != null && b.accuracy != null && Math.abs(a.accuracy - b.accuracy) >= 0.15) {
    const w = a.accuracy > b.accuracy ? 0 : 1;
    out.push(`${names[w]} was the more accurate (${Math.round(res.stats[w].accuracy * 100)}% landed vs ${Math.round(res.stats[1 - w].accuracy * 100)}%).`);
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
      <div class="row"><span class="row-label"><i class="dot f0"></i>Fighter A</span><input class="name-in" data-k="n0" value="Fighter A" maxlength="24" aria-label="Fighter A name"></div>
      <div class="row"><span class="row-label"><i class="dot f1"></i>Fighter B</span><input class="name-in" data-k="n1" value="Fighter B" maxlength="24" aria-label="Fighter B name"></div>
      <div class="row"><span class="row-label" id="lbl-quality">Detail</span>
        <div class="seg" role="radiogroup" aria-labelledby="lbl-quality" data-k="quality">
          <button type="button" role="radio" data-value="0.25" aria-checked="true">Thorough</button>
          <button type="button" role="radio" data-value="0.5" aria-checked="false">Standard</button>
          <button type="button" role="radio" data-value="1" aria-checked="false">Quick</button>
        </div>
      </div>
    </div>
    <label class="btn primary" data-k="pick">Choose a fight clip<input type="file" accept="video/*" hidden data-k="file"></label>
    <p class="hint-text" data-k="hint">Thorough reads the clip at ¼ speed and double-checks who's who across the whole clip (a 1-minute clip takes about 4–5 minutes). The breakdown then plays back at normal speed. Works best side-on, both fighters in frame.</p>
    <p class="status" data-k="status" role="status"></p>
    <div class="card bd-player" data-k="player" hidden>
      <div class="bd-stage"><video data-k="video" playsinline muted></video><canvas data-k="overlay"></canvas><canvas data-k="film" class="film-canvas" hidden></canvas></div>
      <div class="bd-who" data-k="who" hidden>
        <p class="bd-who-title">Who's who?</p>
        <p class="sheet-note">Tap a box to change it: Fighter A → Fighter B → Referee → Ignore. It keeps track of them from here,
        even when they swap sides.</p>
        <div class="bd-names">
          <label class="bd-name f0"><span><i class="dot"></i>Fighter A is</span>
            <input data-k="wn0" maxlength="24" placeholder="Type their name" autocomplete="off" aria-label="Fighter A's name"></label>
          <label class="bd-name f1"><span><i class="dot"></i>Fighter B is</span>
            <input data-k="wn1" maxlength="24" placeholder="Type their name" autocomplete="off" aria-label="Fighter B's name"></label>
        </div>
        <div class="bd-stances">
          <div class="bd-stance f0"><span data-k="st0txt">Stance: reading…</span>
            <div class="seg" role="radiogroup" aria-label="Fighter A's stance" data-k="st0">
              <button type="button" role="radio" data-value="auto">Auto</button><button type="button" role="radio" data-value="orthodox">Orthodox</button><button type="button" role="radio" data-value="southpaw">Southpaw</button></div></div>
          <div class="bd-stance f1"><span data-k="st1txt">Stance: reading…</span>
            <div class="seg" role="radiogroup" aria-label="Fighter B's stance" data-k="st1">
              <button type="button" role="radio" data-value="auto">Auto</button><button type="button" role="radio" data-value="orthodox">Orthodox</button><button type="button" role="radio" data-value="southpaw">Southpaw</button></div></div>
        </div>
        <div class="row-btns"><button type="button" class="mini primary-mini" data-k="whook">Looks right, analyse</button>
          <button type="button" class="mini" data-k="whonext">Try a later frame</button></div>
      </div>
      <div class="bd-progress" data-k="progress"><div class="bar"><i data-k="pbar"></i></div><span data-k="ptext">Reading the clip…</span><button type="button" class="mini" data-k="stop">Stop here</button></div>
      <div class="bd-timeline" data-k="timeline" hidden></div>
      <div class="bd-legend" data-k="legend" hidden></div>
    </div>
    <div data-k="results" hidden class="bd-results"></div>
    <p class="foot-note"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>The clip is processed on this phone and never uploaded. Calls are estimates from body pose: it sees punches thrown, not whether they landed.</p>`;
  const $ = (k) => root.querySelector(`[data-k="${k}"]`);
  const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
  loadPunchModel();   // the pre-trained punch network (small; fetched once, then cached offline)
  let rate = 0.25, frames = [], result = null, processing = false, url = null, watchEnd = null, raf = 0;
  let rawFrames = [], liveTracker = null;   // everything seen, for the second pass
  let fightModel = loadFightModel(), overrides = {};   // learned corrections; this clip's corrections
  const OUTCOME_ICON = { landed: '✓', blocked: '⊘', missed: '✗' };
  // this clip's corrections are remembered, so reopening the same clip keeps them
  const FIX_KEY = 'bc.clipFixes.v1';
  let clipId = null;
  // stances you've set, remembered by fighter name ("Moses" is a southpaw on every clip)
  const STANCE_KEY = 'bc.fighterStance.v1';
  const knownStances = () => { try { return JSON.parse(localStorage.getItem(STANCE_KEY) || '{}'); } catch { return {}; } };
  const rememberStance = (name, st) => {
    if (/^Fighter [AB]$/.test(name)) return;
    try { const all = knownStances(); all[name.toLowerCase()] = st; localStorage.setItem(STANCE_KEY, JSON.stringify(all)); } catch { /* ignore */ }
  };
  const setStance = (k, st) => {
    overrides.stance = { ...(overrides.stance || {}), [k]: st };
    rememberStance(names()[k], st);
    saveFixes();
  };
  // this clip's corrections, plus remembered stances for the names typed in
  const opts = () => {
    const known = knownStances(), nm = names(), stance = { ...(overrides.stance || {}) };
    for (const k of [0, 1]) if (!stance[k] && known[nm[k].toLowerCase()]) stance[k] = known[nm[k].toLowerCase()];
    return { model: fightModel, overrides: { ...overrides, stance } };
  };
  const loadFixes = (id) => { try { return JSON.parse(localStorage.getItem(FIX_KEY) || '{}')[id] || {}; } catch { return {}; } };
  const saveFixes = () => {
    if (!clipId) return;
    try { const all = JSON.parse(localStorage.getItem(FIX_KEY) || '{}'); all[clipId] = overrides; localStorage.setItem(FIX_KEY, JSON.stringify(all)); } catch { /* storage full or blocked */ }
  };
  const names = () => [$('n0').value.trim() || 'Fighter A', $('n1').value.trim() || 'Fighter B'];
  let setup = null;   // { dets, roles, boxes } while choosing who's who
  let setupChanged = null;   // re-reads the stances when you change who's who

  $('quality').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    rate = Number(b.dataset.value);
    $('quality').querySelectorAll('button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
  });
  for (const k of ['n0', 'n1']) $(k).addEventListener('input', () => { if (result) render(); });
  // a name with a remembered stance re-reads that fighter's punches once you've finished typing
  for (const k of ['n0', 'n1', 'wn0', 'wn1']) $(k).addEventListener('change', () => { if (result) reanalyse(); });
  // names typed in the who's-who step become the fighters' names everywhere (boxes, write-up, voice-over)
  for (const i of [0, 1]) {
    $(`wn${i}`).addEventListener('input', () => {
      $(`n${i}`).value = $(`wn${i}`).value.trim() || `Fighter ${'AB'[i]}`;
      drawSetup();
      if (result) render();
    });
  }
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
    try { await new Promise((ok, bad) => { video.onloadeddata = ok; video.onerror = bad; }); }
    catch { fail('That video could not be opened on this device.'); return; }
    frames = []; result = null; rawFrames = [];
    clipId = `${file.name}|${file.size}`; overrides = loadFixes(clipId);
    const tracker = new Tracker(video.videoWidth);
    liveTracker = tracker;
    const sampler = new Sampler();
    const w = video.videoWidth, h = video.videoHeight;

    // 1. who's who: find a frame with at least two people, show the guess, let the user fix it
    $('ptext').textContent = 'Finding the fighters and the referee…';
    const confirmed = await chooseWho(lmk, sampler, w, h);
    if (!confirmed) return;
    tracker.init(confirmed.dets, confirmed.apps, confirmed.roles);

    // 2. track the whole clip
    $('progress').hidden = false;
    await seekTo(0);
    processing = true;
    let lastT = -1;
    const onFrame = (_now, meta) => {
      if (!processing) return;
      const t = meta ? meta.mediaTime : video.currentTime;
      if (t > lastT + 1e-4) {
        lastT = t;
        try {
          const r = lmk.detectForVideo(video, performance.now());
          const dets = (r.landmarks || []).map((lm) => packPose(lm, w, h));
          const cut = sampler.grab(video);
          const apps = dets.map((p) => sampler.appearance(p));
          const [a, b, ref] = tracker.update(dets, apps, cut, t);
          frames.push({ t, poses: [a, b], ref, cut });
          rawFrames.push({ t, dets, apps, cut });
        } catch (e) { console.error(e); }
        $('pbar').style.width = `${Math.min(100, (t / (video.duration || 1)) * 100)}%`;
        $('ptext').textContent = `Tracking the fighters and referee… ${fmt(t)} / ${fmt(video.duration)}`;
        drawAt(t);
      }
      next();
    };
    const next = () => { if (!processing) return; if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame); else requestAnimationFrame(() => onFrame(0, null)); };
    video.onended = () => finish();
    // phones stop sending video frames to a page in the background: pause until it's back
    const onVis = () => {
      if (!processing) { document.removeEventListener('visibilitychange', onVis); return; }
      if (document.hidden) { video.pause(); $('ptext').textContent = 'Paused while the app is in the background…'; }
      else { video.play().catch(() => {}); next(); }
    };
    document.addEventListener('visibilitychange', onVis);
    video.playbackRate = rate;
    try { await video.play(); } catch (e) { console.error(e); fail('That video could not be played on this device.'); return; }
    next();
  }

  function seekTo(t) {
    return new Promise((ok) => {
      if (Math.abs(video.currentTime - t) < 0.01 && video.readyState >= 2) { ok(); return; }
      video.addEventListener('seeked', ok, { once: true });
      video.currentTime = t;
      setTimeout(ok, 2500);
    });
  }

  async function detectAt(lmk, sampler, w, h, t) {
    await seekTo(t);
    const r = lmk.detectForVideo(video, performance.now());
    const dets = (r.landmarks || []).map((lm) => packPose(lm, w, h));
    sampler.grab(video);
    return { dets, apps: dets.map((p) => sampler.appearance(p)), t };
  }

  // Shows a frame with everyone boxed and the app's guess; resolves once the user confirms.
  async function chooseWho(lmk, sampler, w, h) {
    const dur = video.duration || 0;
    const scan = async (from) => {
      for (let tt = from; tt < Math.min(dur, from + 15); tt += 0.75) {
        const d = await detectAt(lmk, sampler, w, h, tt);
        if (d.dets.length >= 2) return d;
      }
      return null;
    };
    let found = await scan(Math.min(0.4, dur / 4));
    if (!found) {
      // fall back to any frame with at least one person
      for (let tt = 0.4; tt < Math.min(dur, 10) && !found; tt += 1) { const d = await detectAt(lmk, sampler, w, h, tt); if (d.dets.length) found = d; }
    }
    if (!found) { fail('Couldn\'t find anyone in the first part of this clip. Try a clip where the fighters are in shot.'); return null; }

    // learn each fighter's stance first, from frames spread through the clip, so lead and rear
    // (jab or cross, lead or rear hook) are right from the start
    $('ptext').textContent = 'Learning each fighter\'s stance…';
    const samples = [];
    const span = Math.min(dur, 60), N = 12;
    for (let i = 0; i < N; i++) {
      const tt = 0.3 + (i + 0.5) * (span - 0.3) / N;
      const d = await detectAt(lmk, sampler, w, h, tt);
      if (d.dets.length >= 2) samples.push(d);
      $('pbar').style.width = `${((i + 1) / N) * 100}%`;
    }
    await seekTo(found.t);
    sampler.grab(video);
    $('progress').hidden = true;

    const pick = [null, null];   // your choice; null = the stance it read
    const known = () => knownStances();
    const nameOf = (k) => $(`wn${k}`).value.trim().toLowerCase();
    // match the marked fighters in each sample frame by appearance, then read their stance against each other
    const guessStances = () => {
      const look = ['A', 'B'].map((r) => { const i = setup.roles.indexOf(r); return i >= 0 ? setup.apps[i] : null; });
      const votes = [[], []];
      if (!look[0] || !look[1]) return [null, null];
      for (const s of [setup, ...samples]) {
        const cand = s.dets.map((p, i) => i).filter((i) => refScore(s.apps[i]) < 1.3);
        let best = null;
        for (const i of cand) for (const j of cand) {
          if (i === j) continue;
          const c = sigDist(s.apps[i], look[0]) + sigDist(s.apps[j], look[1]);
          if (!best || c < best.c) best = { i, j, c };
        }
        if (!best || best.c > 1.2) continue;
        const pa = s.dets[best.i], pb = s.dets[best.j];
        const va = stanceVote(pa, pb), vb = stanceVote(pb, pa);
        if (va) votes[0].push(va);
        if (vb) votes[1].push(vb);
      }
      return votes.map((v) => {
        if (v.length < 3) return null;
        const orth = v.filter((x) => x > 0).length / v.length;
        return { stance: orth >= 0.5 ? 'orthodox' : 'southpaw', share: Math.max(orth, 1 - orth), n: v.length };
      });
    };
    let guess = [null, null];
    const showStances = () => {
      guess = guessStances();
      const kn = known();
      for (const k of [0, 1]) {
        const g = guess[k], rem = kn[nameOf(k)];
        $(`st${k}`).querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === (pick[k] || 'auto'))));
        let txt;
        if (pick[k]) txt = `Stance: ${pick[k]} (your call)`;
        else if (rem) txt = `Stance: ${rem} (remembered for this name)`;
        else if (g && g.share >= 0.75 && g.n >= 5) txt = `Stance: looks ${g.stance} (${Math.round(g.share * g.n)} of ${g.n} frames)`;
        else if (g) txt = `Stance: probably ${g.stance}, not sure. Pick one if you know`;
        else txt = 'Stance: will be read from the whole clip';
        $(`st${k}txt`).textContent = txt.replace('Stance:', `${$(`wn${k}`).value.trim() || `Fighter ${'AB'[k]}`}'s stance:`);
      }
    };
    for (const k of [0, 1]) {
      $(`st${k}`).onclick = (e) => {
        const b = e.target.closest('button'); if (!b) return;
        pick[k] = b.dataset.value === 'auto' ? null : b.dataset.value;
        showStances();
      };
      $(`wn${k}`).oninput = () => showStances();   // a remembered name fills its stance in
    }

    return new Promise((resolve) => {
      const show = (f) => {
        setup = { ...f, roles: Tracker.guess(f.dets, f.apps) };
        // prefill with names already typed (but not the "Fighter A" defaults)
        for (const i of [0, 1]) { const v = $(`n${i}`).value.trim(); $(`wn${i}`).value = /^Fighter [AB]$/.test(v) ? '' : v; }
        for (const k of [0, 1]) pick[k] = overrides.stance?.[k] || null;
        $('who').hidden = false;
        overlay.classList.add('picking');
        drawSetup();
        showStances();
      };
      setupChanged = showStances;
      show(found);
      $('whook').onclick = () => {
        // one fighter is enough: the other is picked up as soon as they come into shot
        if (!setup.roles.includes('A') && !setup.roles.includes('B')) {
          $('status').textContent = 'Mark at least one fighter first.'; $('status').classList.add('error');
          return;
        }
        $('status').textContent = ''; $('status').classList.remove('error');
        // lock in the stances: yours, a remembered one, or a confident read
        const kn = known();
        for (const k of [0, 1]) {
          const g = guess[k];
          const st = pick[k] || kn[nameOf(k)] || (g && g.share >= 0.75 && g.n >= 5 ? g.stance : null);
          if (st) setStance(k, st);
        }
        setupChanged = null;
        const s = setup; setup = null;
        $('who').hidden = true; overlay.classList.remove('picking');
        resolve(s);
      };
      $('whonext').onclick = async () => {
        $('whonext').disabled = true;
        const f = await scan(setup.t + 1.5);
        $('whonext').disabled = false;
        if (f) show(f); else $('status').textContent = 'No later frame with two people in shot.';
      };
    });
  }

  // maps between video pixels and the overlay
  function fit() {
    const cw = overlay.clientWidth, ch = overlay.clientHeight;
    const vw = video.videoWidth, vh = video.videoHeight;
    const sc = Math.min(cw / vw, ch / vh);
    return { cw, ch, sc, ox: (cw - vw * sc) / 2, oy: (ch - vh * sc) / 2 };
  }
  function prepCanvas() {
    const { cw, ch } = fit(), dpr = window.devicePixelRatio || 1;
    if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
    octx.setTransform(dpr, 0, 0, dpr, 0, 0); octx.clearRect(0, 0, cw, ch);
  }
  const ROLE_LABEL = { A: () => names()[0], B: () => names()[1], R: () => 'Referee' };
  const ROLE_COLOR = { A: COLORS[0], B: COLORS[1], R: REF_COLOR };

  function drawSetup() {
    if (!setup) return;
    prepCanvas();
    const { sc, ox, oy } = fit();
    setup.dets.forEach((p, i) => {
      const b = box(p); if (!b) return;
      const role = setup.roles[i];
      const col = role ? ROLE_COLOR[role] : 'rgba(255,255,255,0.45)';
      const x = ox + b.x0 * sc, y = oy + b.y0 * sc, bw = (b.x1 - b.x0) * sc, bh = (b.y1 - b.y0) * sc;
      octx.lineWidth = 3; octx.strokeStyle = col; octx.setLineDash(role ? [] : [6, 5]);
      octx.beginPath(); octx.roundRect(x, y, bw, bh, 10); octx.stroke(); octx.setLineDash([]);
      drawSkeleton(p, col, sc, ox, oy, 0.8);
      tag(role ? ROLE_LABEL[role]() : 'Ignore', x + bw / 2, Math.max(14, y - 4), col, true);
    });
  }

  overlay.addEventListener('click', (e) => {
    if (!setup) return;
    const r = overlay.getBoundingClientRect();
    const { sc, ox, oy } = fit();
    const vx = (e.clientX - r.left - ox) / sc, vy = (e.clientY - r.top - oy) / sc;
    const hits = setup.dets.map((p, i) => ({ i, b: box(p) })).filter(({ b }) => b && vx >= b.x0 && vx <= b.x1 && vy >= b.y0 && vy <= b.y1)
      .sort((a, b) => (a.b.x1 - a.b.x0) * (a.b.y1 - a.b.y0) - (b.b.x1 - b.b.x0) * (b.b.y1 - b.b.y0));
    if (!hits.length) return;
    const i = hits[0].i;
    const order = ['A', 'B', 'R', null];
    const next = order[(order.indexOf(setup.roles[i]) + 1) % order.length];
    if (next) setup.roles = setup.roles.map((x) => (x === next ? null : x));   // each role belongs to one person
    setup.roles[i] = next;
    drawSetup();
    setupChanged?.();
  });

  function drawSkeleton(p, col, sc, ox, oy, alpha = 1) {
    const P = (mi) => { const q = pt(p, mi); return [ox + q[0] * sc, oy + q[1] * sc, q[2]]; };
    octx.save();
    octx.globalAlpha = alpha; octx.lineCap = 'round'; octx.lineWidth = 2.5; octx.strokeStyle = col; octx.shadowColor = col; octx.shadowBlur = 8;
    for (const [a, b] of BONES) {
      const A1 = P(a), B1 = P(b);
      if (A1[2] < 0.3 || B1[2] < 0.3) continue;
      octx.beginPath(); octx.moveTo(A1[0], A1[1]); octx.lineTo(B1[0], B1[1]); octx.stroke();
    }
    octx.restore();
  }

  function tag(text, x, y, col, bold = false) {
    const { cw } = fit();
    octx.save();
    octx.font = `${bold ? 700 : 600} 13px system-ui, sans-serif`;
    const tw = octx.measureText(text).width + 14;
    const left = Math.max(4, Math.min(cw - tw - 4, x - tw / 2)), top = Math.max(4, y - 20);
    octx.fillStyle = 'rgba(5,6,7,0.82)'; octx.beginPath(); octx.roundRect(left, top, tw, 20, 10); octx.fill();
    octx.strokeStyle = col; octx.lineWidth = 1; octx.stroke();
    octx.fillStyle = col; octx.fillText(text, left + 7, top + 14);
    octx.restore();
  }

  async function finish() {
    if (!processing) return;
    processing = false;
    video.pause();
    await loadPunchModel();
    // second pass: re-decide who's who for every frame using the whole clip
    if (rawFrames.length && liveTracker) {
      $('ptext').textContent = 'Double-checking who\'s who across the whole clip…';
      try { frames = smoothIdentities(rawFrames, liveTracker.tracks.map((tr) => tr.bank)); } catch (e) { console.error(e); }
    }
    $('progress').hidden = true;
    result = analyse(frames, opts());
    const secs = Math.max(1, (frames.length ? frames[frames.length - 1].t : video.currentTime) || video.duration || 1);
    if (frames.length < secs * 3) {
      $('status').textContent = 'Only part of the clip could be read (the app was probably in the background). Keep it open on screen and try again.';
      $('status').classList.add('error');
    } else if (result.coverage.some((c) => c < 0.3)) {
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

  // Skeletons for both fighters (and the referee, in grey), names, punches and knockdowns.
  function drawAt(t) {
    if (setup) return drawSetup();
    if (!overlay.clientWidth || !video.videoWidth) return;
    prepCanvas();
    if (!frames.length) return;
    const { sc, ox, oy } = fit();
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (frames[m].t < t) lo = m + 1; else hi = m; }
    const fr = frames[lo];
    if (!fr || Math.abs(fr.t - t) > 0.3) return;
    // hold each person's last pose for a moment when the model misses them, so labels don't flicker
    const held = (get) => {
      for (let j = lo; j >= 0 && fr.t - frames[j].t <= 0.3; j--) { if (frames[j].cut && j !== lo) break; const p = get(frames[j]); if (p) return p; }
      return null;
    };
    // glide between the two analysed frames either side of the playback time
    const smooth = (get) => {
      const cur = held(get);
      const a = frames[lo - 1], b = frames[lo];
      if (!cur || !a || !b || b.cut || b.t <= a.t || t < a.t || t > b.t) return cur;
      const pa = get(a), pb = get(b);
      if (!pa || !pb) return cur;
      const u = (t - a.t) / (b.t - a.t), out = new Float32Array(pa.length);
      for (let i = 0; i < pa.length; i++) out[i] = pa[i] + (pb[i] - pa[i]) * u;
      return out;
    };
    const head = (p) => { const q = pt(p, 0); return [ox + q[0] * sc, oy + q[1] * sc]; };
    const refP = smooth((f) => f.ref);
    if (refP) {
      drawSkeleton(refP, REF_COLOR, sc, ox, oy, 0.6);
      const h = head(refP); tag('REF', h[0], h[1] - 26, REF_COLOR);
    }
    const nm = names();
    [0, 1].map((k) => smooth((f) => f.poses[k])).forEach((p, k) => {
      if (!p) return;
      drawSkeleton(p, COLORS[k], sc, ox, oy);
      const h = head(p);
      const labels = [];
      if (result) {
        if (result.knockdowns.some((kd) => kd.f === k && t >= kd.t && t <= kd.tEnd + 1.5)) labels.push(['KNOCKDOWN', '#ff6b6b']);
        for (const q of result.punches) if (q.f === k && t >= q.t - 0.1 && t <= q.t + 0.5) labels.push([NAMES[q.label] + (q.target === 'body' ? ' (body)' : ''), COLORS[k]]);
        for (const q of result.feints) if (q.f === k && t >= q.t - 0.1 && t <= q.t + 0.6) labels.push([q.role === 'lead' ? 'Jab feint' : 'Feint', '#e6b3ff']);
        for (const q of result.levels) if (q.f === k && t >= q.t - 0.1 && t <= q.t + 0.6) labels.push(['Level change', '#9be7ff']);
      }
      if (!labels.length) labels.push([nm[k], COLORS[k]]);
      labels.forEach(([txt, col], i) => tag(txt, h[0], h[1] - 26 - i * 24, col, i === 0));
    });
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
    for (const fe of res.feints) {
      const d = document.createElement('i'); d.className = `bd-feint f${fe.f}`; d.style.left = `${(fe.t / dur) * 100}%`; d.title = `${nm[fe.f]} feint`;
      tl.append(d);
    }
    for (const lv of res.levels) {
      const d = document.createElement('i'); d.className = `bd-level f${lv.f}`; d.style.left = `${(lv.t / dur) * 100}%`; d.title = `${nm[lv.f]} level change`;
      tl.append(d);
    }
    for (const kd of res.knockdowns) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'bd-kd';
      b.style.left = `${(kd.t / dur) * 100}%`; b.title = `Knockdown: ${nm[kd.f]} down`; b.setAttribute('aria-label', `Watch the knockdown at ${fmt(kd.t)}`);
      b.addEventListener('click', () => watch({ start: kd.t - 1.5, end: kd.tEnd }));
      tl.append(b);
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
      const h = document.createElement('p'); h.className = `card-title f${k}`;
      h.textContent = nm[k] + (s.scored ? ` · ${s.scored} knockdown${s.scored > 1 ? 's' : ''}` : '') + (s.suffered ? ` · down ${s.suffered}×` : '');
      const st = document.createElement('div'); st.className = 'bd-stats';
      for (const [v, l] of [[s.total, 'Punches'], [s.accuracy == null ? '–' : `${s.landed} (${Math.round(s.accuracy * 100)}%)`, 'Landed'], [s.won, 'Won'], [s.counters, 'Counters'], [s.feints, 'Feints'], [s.levels, 'Level chg']]) {
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
      // stance: decides lead or rear for every punch this fighter throws
      const sr = document.createElement('div'); sr.className = 'bd-stance-row';
      const cur = res.stance?.[k];
      sr.append(Object.assign(document.createElement('span'), { textContent: cur?.set ? 'Stance' : 'Stance (read)' }));
      for (const v of ['orthodox', 'southpaw']) {
        const b = document.createElement('button'); b.type = 'button'; b.className = 'mini';
        b.textContent = v === 'orthodox' ? 'Orthodox' : 'Southpaw';
        b.setAttribute('aria-pressed', String(cur?.name === v));
        b.addEventListener('click', () => {
          if (cur?.name === v && cur.set) return;
          setStance(k, v); reanalyse();
          $('status').textContent = `${nm[k]} set to ${v}: every punch re-read${/^Fighter [AB]$/.test(nm[k]) ? '' : `, and remembered for ${nm[k]}`}.`;
          $('status').classList.remove('error');
        });
        sr.append(b);
      }
      c.append(h, st, sr, mix);
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
      const fixBox = document.createElement('div'); fixBox.className = 'bd-fix'; fixBox.hidden = true;
      for (const p of ex.punches) {
        const li = document.createElement('li');
        const b = document.createElement('button'); b.type = 'button';
        b.className = `f${p.f} o-${p.outcome || 'unknown'}${p.corrected ? ' fixed' : ''}`;
        b.textContent = `${OUTCOME_ICON[p.outcome] || ''} ${NAMES[p.label]}${p.target === 'body' ? ' ↓' : ''}`.trim();
        b.title = `${nm[p.f]} · ${fmt(p.t)} · ${p.outcome || 'outcome unknown'} · tap to correct`;
        b.addEventListener('click', () => openFix(p, fixBox));
        li.append(b); seq.append(li);
      }
      const notes = document.createElement('ul'); notes.className = 'bd-notes';
      for (const n of ex.notes) {
        const li = document.createElement('li'); li.className = `n-${n.k}`;
        li.append(fill(n.text, nm)); notes.append(li);
      }
      // who won it? tap to set it yourself
      const win = document.createElement('div'); win.className = 'bd-win';
      win.append(Object.assign(document.createElement('span'), { textContent: 'Winner:' }));
      for (const [v, text] of [[0, nm[0]], ['even', 'Even'], [1, nm[1]]]) {
        const b = document.createElement('button'); b.type = 'button'; b.className = `mini${v === 'even' ? '' : ` f${v}`}`; b.textContent = text;
        b.setAttribute('aria-pressed', String(v === 'even' ? ex.winner == null : ex.winner === v));
        b.addEventListener('click', () => {
          overrides.win = Object.fromEntries(Object.entries(overrides.win || {}).filter(([t]) => Math.abs(Number(t) - ex.start) >= 0.6));
          overrides.win[ex.start.toFixed(2)] = v;
          saveFixes(); reanalyse();
          $('status').textContent = 'Winner saved for this clip.'; $('status').classList.remove('error');
        });
        win.append(b);
      }
      if (ex.knockdown) {
        const b = document.createElement('button'); b.type = 'button'; b.className = 'mini'; b.textContent = 'Not a knockdown';
        b.addEventListener('click', () => {
          const kd = ex.knockdown;
          if (kd.marked) overrides.kdAdd = (overrides.kdAdd || []).filter((m) => Math.abs(m.t - kd.t) >= 1);
          else overrides.kdOff = [...(overrides.kdOff || []), kd.t];
          saveFixes(); reanalyse();
          $('status').textContent = 'Knockdown removed for this clip.'; $('status').classList.remove('error');
        });
        win.append(b);
      }
      c.append(top, seq, fixBox, notes, win);
      R.append(c);
    }
    const legend = document.createElement('p'); legend.className = 'hint-text';
    legend.textContent = '✓ landed · ⊘ blocked · ✗ missed · ↓ to the body. Tap any punch to correct it: every correction teaches the fight model. Tap a winner to overrule the scoring.';
    R.append(legend);

    // missed a knockdown? pause on it and mark it
    const mk = document.createElement('div'); mk.className = 'card bd-mark';
    mk.innerHTML = `<p class="card-title">Missed something?</p>
      <p class="sheet-note">Pause the video on the moment, then add it. Saved for this clip.</p>
      <p class="bd-mark-sub">A punch it didn't see</p>
      <div class="bd-addp">
        <select data-m="pf" aria-label="Who threw it"></select>
        <select data-m="pl" aria-label="Which punch"></select>
        <select data-m="po" aria-label="Did it land"><option value="landed">✓ Landed</option><option value="blocked">⊘ Blocked</option><option value="missed">✗ Missed</option></select>
        <button type="button" class="mini primary-mini" data-m="padd">Add punch here</button></div>
      <p class="bd-mark-sub">A knockdown: tap who went down (pause where they start to fall)</p>
      <div class="row-btns"><button type="button" class="mini f0" data-m="0"></button><button type="button" class="mini f1" data-m="1"></button></div>`;
    const M = (k) => mk.querySelector(`[data-m="${k}"]`);
    nm.forEach((n, k) => M('pf').append(new Option(n, String(k))));
    for (const l of Object.keys(NAMES)) M('pl').append(new Option(NAMES[l], l));
    M('padd').addEventListener('click', () => {
      const t = video.currentTime, f = Number(M('pf').value), label = M('pl').value;
      overrides.add = [...(overrides.add || []).filter((m) => !(m.f === f && Math.abs(m.t - t) < 0.25)), { f, t, label, outcome: M('po').value }];
      saveFixes(); reanalyse();
      $('status').textContent = `Added ${nm[f]}'s ${NAMES[label].toLowerCase()} at ${fmt(t)}.`; $('status').classList.remove('error');
    });
    for (const k of [0, 1]) {
      const b = mk.querySelector(`[data-m="${k}"]`);
      b.textContent = `${nm[k]} down here`;
      b.addEventListener('click', () => {
        const t = video.currentTime;
        overrides.kdOff = (overrides.kdOff || []).filter((x) => Math.abs(x - t) >= 1);
        overrides.kdAdd = [...(overrides.kdAdd || []).filter((m) => Math.abs(m.t - t) >= 1), { f: k, t }];
        saveFixes(); reanalyse();
        $('status').textContent = `Knockdown marked at ${fmt(t)}.`; $('status').classList.remove('error');
      });
    }
    R.append(mk);

    // the fight model trained from your corrections
    const fm = document.createElement('div'); fm.className = 'card bd-model';
    const counts = {};
    for (const e of fightModel.ex) counts[e.l] = (counts[e.l] || 0) + 1;
    fm.innerHTML = `<p class="card-title">Fight model</p>
      <p class="sheet-note" data-m="sum"></p>
      <div class="row-btns"><button type="button" class="mini" data-m="export">Export</button>
        <label class="mini">Import<input type="file" accept="application/json" hidden data-m="import"></label>
        <button type="button" class="mini" data-m="reset">Reset</button></div>`;
    fm.querySelector('[data-m="sum"]').textContent = fightModel.ex.length
      ? `Learned from ${fightModel.ex.length} correction${fightModel.ex.length > 1 ? 's' : ''} (${Object.entries(counts).map(([l, n]) => `${l === 'none' ? 'not a punch' : NAMES[l].toLowerCase()} ${n}`).join(', ')}). ${fightModel.ex.length < 8 ? `It starts using them after 8.` : 'It now checks every new clip against them.'}`
      : 'Not trained yet. Tap punches in the exchanges above to correct or confirm them; after 8 it starts using what you taught it on every clip.';
    fm.querySelector('[data-m="export"]').addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(fightModel)], { type: 'application/json' });
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'fight-model.json'; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    });
    fm.querySelector('[data-m="import"]').addEventListener('change', async (e) => {
      const f = e.target.files[0]; if (!f) return;
      try {
        const m = JSON.parse(await f.text());
        if (!Array.isArray(m.ex)) throw new Error('bad file');
        fightModel.ex.push(...m.ex.filter((x) => x && Array.isArray(x.f) && typeof x.l === 'string'));
        saveFightModel(fightModel); reanalyse();
      } catch { $('status').textContent = 'That file isn\'t a fight model export.'; }
    });
    fm.querySelector('[data-m="reset"]').addEventListener('click', (e) => {
      if (!e.target.dataset.sure) { e.target.dataset.sure = '1'; e.target.textContent = 'Tap again to reset'; return; }
      fightModel = { ex: [] }; saveFightModel(fightModel); reanalyse();
    });
    R.append(fm);
  }

  // Correct (or confirm) one punch: type, outcome, or "not a punch". Saved as a training example.
  function openFix(p, box) {
    box.hidden = false;
    box.innerHTML = `<p class="bd-fix-title"></p>
      <div class="bd-fix-row" data-x="labels"></div>
      <div class="seg" role="radiogroup" aria-label="Outcome" data-x="outcome">
        <button type="button" role="radio" data-value="landed">✓ Landed</button>
        <button type="button" role="radio" data-value="blocked">⊘ Blocked</button>
        <button type="button" role="radio" data-value="missed">✗ Missed</button></div>
      <div class="row-btns"><button type="button" class="mini primary-mini" data-x="save">Save</button>
        <button type="button" class="mini" data-x="none">Not a punch</button>
        <button type="button" class="mini" data-x="cancel">Cancel</button></div>`;
    const X = (k) => box.querySelector(`[data-x="${k}"]`);
    box.querySelector('.bd-fix-title').textContent = `${names()[p.f]} at ${fmt(p.t)}: what was it?`;
    let label = p.label, outcome = p.outcome;
    const sync = () => {
      X('labels').querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.value === label)));
      X('outcome').querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === outcome)));
    };
    for (const l of Object.keys(NAMES)) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'mini'; b.dataset.value = l; b.textContent = NAMES[l];
      b.addEventListener('click', () => { label = l; sync(); });
      X('labels').append(b);
    }
    X('outcome').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { outcome = b.dataset.value; sync(); } });
    const commit = (none) => {
      // a lead/rear mix-up means the stance was read wrong: fix it for every punch this fighter throws
      const flip = !none && ROLE[label] !== p.role && !p.added;
      if (flip) {
        const cur = result.stance?.[p.f]?.name;
        setStance(p.f, cur === 'southpaw' ? 'orthodox' : 'southpaw');
      }
      if (p.feat) {
        // the model learns the shape; its lead/rear flag follows the corrected stance
        const f = p.feat.slice(); if (flip) f[f.length - 1] = ROLE[label] === 'lead' ? 1 : 0;
        fightModel.ex.push({ l: none ? 'none' : label, f }); saveFightModel(fightModel);
      }
      if (none && p.added) overrides.add = (overrides.add || []).filter((m) => !(m.f === p.f && Math.abs(m.t - p.t) < 0.05));
      else overrides[punchKey(p)] = none ? { none: true } : { label, outcome };
      saveFixes(); reanalyse();
      const nm = names()[p.f];
      $('status').textContent = none ? 'Removed, and the model will learn it wasn\'t a punch.'
        : flip ? `Saved. That was a stance mix-up, so ${nm} is now ${result.stance?.[p.f]?.name} and all ${nm}'s punches were re-read.`
          : 'Saved. The model learns from every correction.';
      $('status').classList.remove('error');
    };
    X('save').addEventListener('click', () => commit(false));
    X('none').addEventListener('click', () => commit(true));
    X('cancel').addEventListener('click', () => { box.hidden = true; });
    sync();
  }

  function reanalyse() {
    if (!frames.length) return;
    result = analyse(frames, opts());
    render();
  }

  return {
    hide() { if (!processing) { video.pause(); } director?.stop(); },
    get _frames() { return frames; },
    // Developer readout: how often each person was seen.
    get _stats() {
      const n = frames.length || 1;
      return { frames: frames.length, A: frames.filter((f) => f.poses[0]).length / n, B: frames.filter((f) => f.poses[1]).length / n, ref: frames.filter((f) => f.ref).length / n, cuts: frames.filter((f) => f.cut).length, punches: result?.punches.length, knockdowns: result?.knockdowns.length };
    },
    // Developer test hook: load a video URL with ready-made frames (skips pose tracking).
    async _testVideo(src, fs) {
      video.src = src; video.controls = true; video.muted = true;
      await new Promise((ok) => { video.onloadeddata = ok; });
      frames = fs; overrides = {}; result = analyse(fs, opts());
      $('player').hidden = false; $('progress').hidden = true;
      startOverlay(); render(); return result;
    },
    // Developer test hook: show the breakdown for ready-made frames (no video).
    _test(fs) { frames = fs; overrides = {}; result = analyse(fs, opts()); $('player').hidden = false; $('progress').hidden = true; render(); return result; },
  };
}

function fmt(s) { s = Math.max(0, Math.floor(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }
