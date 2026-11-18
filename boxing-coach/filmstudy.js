// AI film study: plays a fight clip like a coach's video breakdown. It slows into each key
// moment, freezes the frame, spotlights the fighter, draws rings, arrows and range lines,
// and talks through the positioning, the setup, the counter and the exit. The whole thing
// is drawn on a canvas, so it can be recorded straight to a video file with the voice-over.

import { NAMES } from './personal.js';
import { pt, mid, center, torso, COLORS } from './breakdown.js';
import { voice } from './voice.js';

const MOVE = 0.35;        // torso lengths that count as stepping in / out
const SLOWMO_LEAD = 0.9;  // s of slow motion before each pause
const SLOWMO_RATE = 0.4;
const MAX_EXCHANGES = 6;

// ---------------------------------------------------------------- pose lookups
function makeLookup(frames) {
  const idx = (t) => {
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (frames[m].t < t) lo = m + 1; else hi = m; }
    return lo;
  };
  // nearest pose of fighter k to time t (searching up to 0.5 s either side)
  const poseAt = (k, t) => {
    const i = idx(t);
    for (let d = 0; d < 30; d++) {
      for (const j of [i - d, i + d]) {
        const f = frames[j];
        if (f && f.poses[k] && Math.abs(f.t - t) <= 0.5) return f.poses[k];
      }
    }
    return null;
  };
  const dirAt = (k, t) => {
    const a = poseAt(k, t), b = poseAt(1 - k, t);
    return a && b ? Math.sign(center(b)[0] - center(a)[0]) || 1 : (k === 0 ? 1 : -1);
  };
  const advance = (k, t0, t1) => {
    const a = poseAt(k, t0), b = poseAt(k, t1);
    if (!a || !b) return 0;
    return ((center(b)[0] - center(a)[0]) * dirAt(k, t0)) / torso(a);
  };
  const rangeAt = (t) => {
    const a = poseAt(0, t), b = poseAt(1, t);
    if (!a || !b) return null;
    return Math.abs(center(a)[0] - center(b)[0]) / ((torso(a) + torso(b)) / 2);
  };
  return { idx, poseAt, dirAt, advance, rangeAt };
}

const lowerName = (l) => NAMES[l].toLowerCase();
const pick = (arr, i) => arr[i % arr.length];

// ---------------------------------------------------------------- the script
// Returns beats: { t, kind, text, marks[] } in time order.
export function buildScript(res, frames, names, readLines = []) {
  const L = makeLookup(frames);
  const N = (k) => names[k];
  const tStart = frames[0].t, tEnd = frames[frames.length - 1].t;
  const beats = [];

  beats.push({
    t: tStart, kind: 'intro',
    text: `Let's break this one down. ${N(0)} is marked in blue, ${N(1)} in gold. I'll stop on the key moments.`,
    marks: [{ type: 'tag', f: 0, text: N(0) }, { type: 'tag', f: 1, text: N(1) }],
  });

  // the most instructive exchanges, in time order
  const score = (ex) => ex.punches.length + 3 * ex.notes.filter((n) => n.k === 'counter').length + 2 * ex.notes.filter((n) => n.k === 'setup').length;
  const chosen = [...res.exchanges].sort((a, b) => score(b) - score(a)).slice(0, MAX_EXCHANGES).sort((a, b) => a.start - b.start);

  let lastBand = null, stayedCount = 0;
  chosen.forEach((ex, n) => {
    const init = ex.init, other = 1 - init;
    const label = `Exchange ${ex.n}`;

    // 1. positioning, just before it starts
    const tp = Math.max(tStart + 0.2, ex.start - 0.45);
    const rng = L.rangeAt(tp);
    const aI = L.advance(init, tp - 1.5, tp), aO = L.advance(other, tp - 1.5, tp);
    const parts = [];
    // Only mention the range when it changes, or on the first stop, so it doesn't repeat.
    const band = rng == null ? null : rng < 2.6 ? 'close' : rng < 3.8 ? 'mid' : 'long';
    if (band && (n === 0 || band !== lastBand)) {
      parts.push(band === 'close'
        ? pick(['They\'re right in the pocket here, close enough to land without stepping.', 'This is close range now: both of them can land from here.', 'Now they\'re in close, where every punch can land.'], n)
        : band === 'mid' ? pick(['They\'re at mid range, one step from punching distance.', 'Look at the distance: mid range, right on the edge.', 'Mid range now: one step and they\'re in.'], n)
          : pick(['They\'re out at long range, nobody can land from here.', 'Long range: whoever wants to score has to close the gap.', 'They\'ve reset to long range.'], n));
    }
    lastBand = band;
    if (aI > MOVE) parts.push(pick([`${N(init)} is walking forward and taking the space.`, `${N(init)} steps in to make it happen.`, `${N(init)} closes the distance.`], n));
    else if (aO > MOVE) parts.push(pick([`${N(other)} is the one pressing forward.`, `${N(other)} comes forward, and walks into it.`], n));
    if (aO < -MOVE) parts.push(pick([`${N(other)} is on the back foot.`, `${N(other)} is giving ground.`], n));
    if (parts.length === 0) parts.push(pick(['Both holding their ground, reading each other.', 'Nobody\'s moving much. Both are waiting for the other to commit.', 'A quiet moment, both measuring.'], n));
    parts.push(pick([`Watch ${N(init)}'s ${lowerName(ex.punches[0].label)}.`, `Keep your eye on ${N(init)}'s ${ex.punches[0].role} hand.`, `${N(init)} is about to go first.`], n));
    const pm = [{ type: 'spot', f: init }, { type: 'range' }];
    if (aI > MOVE) pm.push({ type: 'move', f: init, dir: 1, text: 'Steps in' });
    if (aO > MOVE) pm.push({ type: 'move', f: other, dir: 1, text: 'Presses' });
    if (aO < -MOVE) pm.push({ type: 'move', f: other, dir: -1, text: 'Backs up' });
    beats.push({ t: tp, kind: 'position', label, text: parts.join(' '), marks: pm });

    // 1b. the feint or level change that sets it up gets its own stop
    if (ex.feint) {
      const fe = ex.feint, me = N(fe.f), opp = N(1 - fe.f), glove = fe.hand === 'left' ? 15 : 16;
      const what = fe.role === 'lead' ? 'the jab' : 'the rear hand';
      beats.push({
        t: fe.tEnd, kind: 'feint', label,
        text: `Watch this. ${me} shows ${what} but doesn't throw it. That's a feint${fe.reaction ? `, and ${opp} reacts to it` : ''}. It's there to draw a response, and open the door for the ${lowerName(fe.followedBy.label)}.`,
        marks: [{ type: 'ring', f: fe.f, j: glove }, { type: 'tag', f: fe.f, text: 'FEINT' }, ...(fe.reaction ? [{ type: 'tag', f: 1 - fe.f, text: 'Reacts' }] : [])],
      });
    }
    if (ex.level) {
      const lv = ex.level, p = lv.followedBy, me = N(lv.f);
      beats.push({
        t: lv.t + 0.12, kind: 'level', label,
        text: p.target === 'body' ? `See ${me} change levels here. Dropping the height brings the body shot into range, and it's hard to see coming.`
          : /uppercut/.test(p.label) ? `${me} dips down first. From down there, the uppercut comes straight up the middle.`
            : `${me} changes levels here. It hides where the next punch is coming from.`,
        marks: [{ type: 'spot', f: lv.f }, { type: 'tag', f: lv.f, text: 'LEVEL CHANGE' }],
      });
    }

    // 2. the setup and the counter (at most two punch stops per exchange)
    const stops = [];
    const r0 = ex.runs[0];
    if (r0.punches.length >= 2) stops.push({ kind: 'setup', run: r0, p: r0.punches[0] });
    ex.runs.forEach((r, i) => {
      const prev = ex.runs[i - 1];
      if (prev && prev.f !== r.f && r.punches[0].t - prev.punches[prev.punches.length - 1].t <= 0.6 && stops.length < 2) {
        stops.push({ kind: 'counter', run: r, p: r.punches[0], hit: prev.punches[prev.punches.length - 1] });
      }
    });
    if (!stops.length) stops.push({ kind: 'single', run: r0, p: r0.punches[0] });
    for (const s of stops.slice(0, 2)) {
      const p = s.p, me = N(p.f), opp = N(1 - p.f);
      const glove = p.hand === 'left' ? 15 : 16;
      let text;
      if (s.kind === 'setup') {
        const seq = s.run.punches.map((q) => q.label);
        const nextP = s.run.punches[1];
        if (seq[0] === 'jab' && seq[1] === 'jab') text = `Two jabs. The first finds the range, the second lets ${me} step in behind it.`;
        else if (seq[0] === 'jab' && nextP.role === 'rear') text = `This jab isn't meant to hurt. It occupies ${opp}'s guard and measures the distance, so the ${lowerName(nextP.label)} can come straight in behind it.`;
        else if (p.target === 'body' && nextP.target === 'head') text = `${me} goes downstairs first. That brings ${opp}'s hands down, and opens the head for the ${lowerName(nextP.label)}.`;
        else if (p.target === 'head' && nextP.target === 'body') text = `${me} goes high first to raise the guard, then digs the ${lowerName(nextP.label)} to the body underneath.`;
        else text = `${me} starts the combination with the ${lowerName(p.label)}, and follows with the ${seq.slice(1).map(lowerName).join(', then the ')}.`;
        if (seq.length >= 3) text += ` ${seq.length} punches in all.`;
      } else if (s.kind === 'counter') {
        const h = s.hit;
        const gap = p.t - h.t;
        if (gap < 0.2) text = `And ${me} fires back at the same moment. That's a straight trade.`;
        else if (h.label === 'jab' && p.label === 'cross') text = `There's the counter. As the jab comes out, ${me} fires the cross straight over the top of it.`;
        else if ((h.label === 'jab' || h.label === 'cross') && p.label === 'lead_hook') text = `${me} lets the straight come, then turns over a lead hook as ${opp}'s hand comes back.`;
        else if (/hook/.test(h.label) && /uppercut/.test(p.label)) text = `The hook goes wide, so ${me} shoots the uppercut straight up the middle.`;
        else text = `${me} answers the ${lowerName(h.label)} with a ${lowerName(p.label)}${p.target === 'body' ? ' to the body' : ''}, ${gap.toFixed(1)} seconds later.`;
      } else {
        text = `A single ${lowerName(p.label)}${p.target === 'body' ? ' to the body' : ''} from ${me}. ${p.label === 'jab' ? 'Probing, gathering information.' : 'No set-up behind it.'}`;
      }
      beats.push({
        t: p.tEnd, kind: s.kind, label, text,
        marks: [{ type: 'ring', f: p.f, j: glove }, { type: 'arrow', from: { f: p.f, j: glove }, to: { f: 1 - p.f, part: p.target === 'body' ? 'body' : 'head' }, text: NAMES[p.label] }],
      });
    }

    // 3. the exit
    const nextEx = res.exchanges.find((e) => e.start > ex.end);
    const tx = Math.min(ex.end + 0.6, nextEx ? nextEx.start - 0.3 : tEnd);
    if (tx > ex.end) {
      const last = ex.punches[ex.punches.length - 1].f;
      const oI = L.advance(init, ex.end, Math.min(tEnd, ex.end + 1)), oO = L.advance(other, ex.end, Math.min(tEnd, ex.end + 1));
      const xs = [];
      const xm = [];
      const countered = ex.punches.some((p) => p.f !== init);
      if (oI < -MOVE) {
        xs.push(pick([`Good discipline from ${N(init)}: punches, then steps straight back out of range.`, `Notice the exit. ${N(init)} doesn't hang around, straight back out.`, `${N(init)} gets in and gets out. That's how you avoid the return fire.`], n));
        xm.push({ type: 'move', f: init, dir: -1, text: 'Exits' });
      } else if (oI > MOVE) {
        xs.push(pick([`${N(init)} keeps walking forward after it, keeping the pressure on.`, `No backing off from ${N(init)}: straight back on the front foot.`], n));
        xm.push({ type: 'move', f: init, dir: 1, text: 'Stays on' });
      } else if (countered && stayedCount < 2) {
        // only point out staying in range when it actually got punished, and not every time
        xs.push(pick([`${N(init)} stays right there in range after punching, and that's what lets ${N(other)} counter.`, `See how ${N(init)} lingers in range? That's the opening ${N(other)} takes.`], stayedCount));
        stayedCount++;
      }
      if (oO < -MOVE) { xs.push(pick([`${N(other)} gives ground.`, `${N(other)} backs away to reset.`], n)); xm.push({ type: 'move', f: other, dir: -1, text: 'Gives ground' }); }
      if (countered) { xs.push(pick([`And ${N(last)} gets the last word in that exchange.`, `${N(last)} finishes it.`, `Last punch: ${N(last)}.`], n)); xm.push({ type: 'tag', f: last, text: 'Last word' }); }
      if (xs.length) beats.push({ t: tx, kind: 'exit', label, text: xs.join(' '), marks: [{ type: 'range' }, ...xm] });
    }
  });

  // knockdowns always get their own stop
  for (const kd of res.knockdowns || []) {
    const c = kd.cause;
    beats.push({
      t: Math.min(tEnd, kd.t + 0.4), kind: 'knockdown', label: 'Knockdown',
      text: `Knockdown! ${N(kd.f)} goes down${c ? `, and it's the ${lowerName(c.label)}${c.target === 'body' ? ' to the body' : ''} from ${N(1 - kd.f)} that does it` : ''}.`,
      marks: [{ type: 'spot', f: kd.f }, { type: 'tag', f: kd.f, text: 'KNOCKDOWN' }],
    });
  }

  const read = readLines.slice(0, 3).join(' ');
  beats.push({ t: tEnd, kind: 'outro', text: read ? `So, the takeaways. ${read}` : 'That\'s the breakdown.', marks: [{ type: 'tag', f: 0, text: N(0) }, { type: 'tag', f: 1, text: N(1) }] });

  beats.sort((a, b) => a.t - b.t);
  for (let i = 1; i < beats.length; i++) if (beats[i].t < beats[i - 1].t + 0.2) beats[i].t = beats[i - 1].t + 0.2;
  return beats;
}

// ---------------------------------------------------------------- the director
export class Director {
  constructor({ canvas, video, frames, beats, onStatus }) {
    Object.assign(this, { canvas, video, frames, beats, onStatus });
    this.L = makeLookup(frames);
    this.marks = []; this.caption = ''; this.label = ''; this.markStart = 0;
    this.stopped = false; this.raf = 0;
  }

  // Makes the voice-over line by line in the background (cached, so replays are instant).
  // Resolves once the first line is ready, so playback can start straight away; later lines
  // are usually ready long before the video reaches them. With record: true it waits for all
  // of them, so the saved video never has gaps.
  async prepare({ all = false } = {}) {
    if (!voice.ai) return;
    let failed = false;
    let chain = Promise.resolve();
    for (const b of this.beats) {
      b.ready = chain = chain.then(async () => {
        if (this.stopped || failed) return;
        try { b.buf = await voice.buffer(b.text); }
        catch (e) { failed = true; voice.lastError = e.message; }
      });
    }
    this.onStatus?.('Writing the voice-over… (the first time also downloads the AI voice, ~90 MB)');
    let n = 0;
    this.beats.forEach((b) => b.ready.then(() => {
      n++;
      if (all) this.onStatus?.(`Writing the voice-over ${n}/${this.beats.length}…`);
    }));
    await (all ? chain : this.beats[0].ready);
    if (failed) this.onStatus?.('The AI voice could not start, using the phone voice.');
  }

  stop() {
    this.stopped = true;
    voice.stop();
    try { this.video.pause(); } catch { /* */ }
  }

  // Plays the breakdown. With record: true resolves to a video Blob.
  async run({ record = false } = {}) {
    const { video, canvas } = this;
    const vw = video.videoWidth, vh = video.videoHeight;
    this.s = Math.min(1, 1280 / vw);
    canvas.width = Math.round(vw * this.s); canvas.height = Math.round(vh * this.s);
    this.ctx = canvas.getContext('2d');

    // sound: clip audio (quiet, ducked under the voice) + voice-over, both through Web Audio
    const ac = voice.context();
    await ac.resume();
    if (!video._bcSource) {
      video._bcSource = ac.createMediaElementSource(video);
      video._bcGain = ac.createGain();
      video._bcSource.connect(video._bcGain).connect(ac.destination);
    }
    this.clipGain = video._bcGain;
    this.clipGain.gain.value = 0.35;
    video.muted = false;

    let rec = null, dest = null;
    const chunks = [];
    if (record) {
      dest = ac.createMediaStreamDestination();
      this.clipGain.connect(dest);
      voice.tap = dest;
      const stream = canvas.captureStream(30);
      dest.stream.getAudioTracks().forEach((t) => stream.addTrack(t));
      const type = ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
        .find((m) => window.MediaRecorder?.isTypeSupported?.(m)) || '';
      rec = new MediaRecorder(stream, { mimeType: type || undefined, videoBitsPerSecond: 6_000_000 });
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      rec.start(500);
    }

    const loop = () => { this.draw(); this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);

    try {
      await this.seek(this.frames[0].t);
      for (const b of this.beats) {
        if (this.stopped) break;
        await this.playUntil(b.t);
        if (this.stopped) break;
        await this.beat(b);
      }
      if (!this.stopped) await sleep(600);
    } finally {
      cancelAnimationFrame(this.raf);
      video.pause(); video.playbackRate = 1;
      this.clipGain.gain.value = 1;
      voice.tap = null;
      if (dest) try { this.clipGain.disconnect(dest); } catch { /* */ }
    }
    if (!rec) return null;
    await new Promise((ok) => { rec.onstop = ok; rec.stop(); });
    return new Blob(chunks, { type: rec.mimeType || 'video/webm' });
  }

  seek(t) {
    return new Promise((ok) => {
      const v = this.video;
      if (Math.abs(v.currentTime - t) < 0.01) { ok(); return; }
      v.addEventListener('seeked', ok, { once: true });
      v.currentTime = t;
      setTimeout(ok, 1500);
    });
  }

  // Plays at normal speed, easing into slow motion just before the next stop.
  async playUntil(t) {
    const v = this.video;
    if (v.currentTime >= t - 0.02) return;
    v.playbackRate = 1;
    try { await v.play(); } catch { /* */ }
    await new Promise((done) => {
      const tick = () => {
        if (this.stopped || v.ended || v.currentTime >= t) { v.pause(); done(); return; }
        if (v.currentTime >= t - SLOWMO_LEAD && v.playbackRate !== SLOWMO_RATE) v.playbackRate = SLOWMO_RATE;
        requestAnimationFrame(tick);
      };
      tick();
    });
  }

  async beat(b) {
    this.marks = b.marks; this.caption = b.text; this.label = b.label || '';
    this.markStart = performance.now();
    const ac = voice.context();
    this.clipGain.gain.setTargetAtTime(0.06, ac.currentTime, 0.1);
    await sleep(350);
    if (b.ready) await b.ready;   // normally already done; otherwise hold the frame until the line is ready
    if (this.stopped) return;
    if (b.buf) await voice.playBuffer(b.buf);
    else await voice.say(b.text);
    await sleep(450);
    this.marks = []; this.caption = '';
    this.clipGain.gain.setTargetAtTime(0.35, ac.currentTime, 0.2);
  }

  // ---------------------------------------------------------------- drawing
  draw() {
    const { ctx, canvas, video } = this;
    const W = canvas.width, H = canvas.height, s = this.s;
    ctx.drawImage(video, 0, 0, W, H);
    const t = video.currentTime;
    const P = (k) => this.L.poseAt(k, t);
    const at = (p, mi) => { const q = pt(p, mi); return [q[0] * s, q[1] * s, q[2]]; };
    const a = Math.min(1, (performance.now() - this.markStart) / 450);   // highlight fade-in
    const pulse = 1 + 0.12 * Math.sin(performance.now() / 160);
    const u = H / 720;   // size unit

    for (const m of this.marks) {
      if (m.type === 'spot') {
        const p = P(m.f); if (!p) continue;
        const pts = [0, 11, 12, 15, 16, 23, 24, 27, 28].map((mi) => at(p, mi)).filter((q) => q[2] > 0.2);
        const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
        const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        const rx = (Math.max(...xs) - Math.min(...xs)) / 2 + 60 * u, ry = (Math.max(...ys) - Math.min(...ys)) / 2 + 50 * u;
        ctx.save();
        ctx.fillStyle = `rgba(0,0,0,${0.55 * a})`;
        ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); ctx.fill('evenodd');
        ctx.restore();
      }
    }
    for (const m of this.marks) {
      if (m.type === 'range') {
        const A = P(0), B = P(1); if (!A || !B) continue;
        const ca = center(A).map((v) => v * s), cb = center(B).map((v) => v * s);
        const y = Math.max(ca[1], cb[1]) + 40 * u;
        const xEnd = ca[0] + (cb[0] - ca[0]) * a;
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 3 * u; ctx.setLineDash([10 * u, 8 * u]);
        ctx.beginPath(); ctx.moveTo(ca[0], y); ctx.lineTo(xEnd, y); ctx.stroke();
        ctx.setLineDash([]);
        for (const x of [ca[0], xEnd]) { ctx.beginPath(); ctx.moveTo(x, y - 10 * u); ctx.lineTo(x, y + 10 * u); ctx.stroke(); }
        const d = this.L.rangeAt(t);
        if (d != null && a >= 1) pill(ctx, (ca[0] + cb[0]) / 2, y - 22 * u, d < 2.6 ? 'CLOSE RANGE' : d < 3.8 ? 'MID RANGE' : 'LONG RANGE', '#fff', u);
        ctx.restore();
      } else if (m.type === 'ring') {
        const p = P(m.f); if (!p) continue;
        const q = at(p, m.j);
        ctx.save();
        ctx.strokeStyle = COLORS[m.f]; ctx.lineWidth = 6 * u; ctx.shadowColor = COLORS[m.f]; ctx.shadowBlur = 22 * u;
        ctx.beginPath(); ctx.arc(q[0], q[1], 42 * u * pulse * a, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      } else if (m.type === 'arrow') {
        const p = P(m.from.f), o = P(m.to.f); if (!p || !o) continue;
        const A0 = at(p, m.from.j);
        const B0 = m.to.part === 'body' ? mid(mid(pt(o, 11), pt(o, 12)), mid(pt(o, 23), pt(o, 24))).map((v) => v * s) : at(o, 0);
        const ex = A0[0] + (B0[0] - A0[0]) * a, ey = A0[1] + (B0[1] - A0[1]) * a;
        ctx.save();
        ctx.strokeStyle = COLORS[m.from.f]; ctx.fillStyle = COLORS[m.from.f]; ctx.lineWidth = 5 * u; ctx.lineCap = 'round';
        ctx.shadowColor = 'rgba(0,0,0,0.7)'; ctx.shadowBlur = 8 * u;
        ctx.beginPath(); ctx.moveTo(A0[0], A0[1]); ctx.lineTo(ex, ey); ctx.stroke();
        arrowHead(ctx, A0[0], A0[1], ex, ey, 18 * u);
        ctx.restore();
        if (m.text && a >= 1) pill(ctx, A0[0], A0[1] - 70 * u, m.text.toUpperCase(), COLORS[m.from.f], u);
      } else if (m.type === 'move') {
        const p = P(m.f); if (!p) continue;
        const c = center(p).map((v) => v * s);
        const feetY = Math.max(at(p, 27)[1], at(p, 28)[1]) + 26 * u;
        const len = torso(p) * s * 1.1 * a;
        const dir = this.L.dirAt(m.f, t) * m.dir;
        ctx.save();
        ctx.strokeStyle = COLORS[m.f]; ctx.fillStyle = COLORS[m.f]; ctx.lineWidth = 6 * u; ctx.lineCap = 'round';
        ctx.shadowColor = 'rgba(0,0,0,0.7)'; ctx.shadowBlur = 8 * u;
        ctx.beginPath(); ctx.moveTo(c[0], feetY); ctx.lineTo(c[0] + dir * len, feetY); ctx.stroke();
        arrowHead(ctx, c[0], feetY, c[0] + dir * len, feetY, 18 * u);
        ctx.restore();
        if (a >= 1) pill(ctx, c[0] + dir * len / 2, feetY + 30 * u, m.text.toUpperCase(), COLORS[m.f], u);
      } else if (m.type === 'tag') {
        const p = P(m.f); if (!p) continue;
        const h = at(p, 0);
        ctx.globalAlpha = a;
        pill(ctx, h[0], h[1] - 80 * u, m.text, COLORS[m.f], u, 28);
        ctx.globalAlpha = 1;
      }
    }

    // header + caption (drawn into the video so they are recorded too)
    pill(ctx, 18 * u, 30 * u, `AI FILM STUDY${this.label ? ` · ${this.label.toUpperCase()}` : ''}`, '#b9d4ff', u, 20, 'left');
    if (this.caption) caption(ctx, this.caption, W, H, u);
  }
}

// ---------------------------------------------------------------- drawing helpers
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function arrowHead(ctx, x0, y0, x1, y1, size) {
  const ang = Math.atan2(y1 - y0, x1 - x0);
  if (Math.hypot(x1 - x0, y1 - y0) < size) return;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - size * Math.cos(ang - 0.45), y1 - size * Math.sin(ang - 0.45));
  ctx.lineTo(x1 - size * Math.cos(ang + 0.45), y1 - size * Math.sin(ang + 0.45));
  ctx.closePath(); ctx.fill();
}

function pill(ctx, x, y, text, color, u, size = 24, align = 'center') {
  ctx.save();
  ctx.font = `700 ${size * u}px system-ui, -apple-system, Segoe UI, sans-serif`;
  const w = ctx.measureText(text).width + 22 * u, h = (size + 14) * u;
  const left = align === 'left' ? x : Math.max(6 * u, Math.min(ctx.canvas.width - w - 6 * u, x - w / 2));
  const top = Math.max(6 * u, y - h / 2);
  ctx.fillStyle = 'rgba(5,6,7,0.82)';
  ctx.beginPath(); ctx.roundRect(left, top, w, h, h / 2); ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = 1.5 * u; ctx.stroke();
  ctx.fillStyle = color; ctx.textBaseline = 'middle';
  ctx.fillText(text, left + 11 * u, top + h / 2 + 1 * u);
  ctx.restore();
}

function caption(ctx, text, W, H, u) {
  ctx.save();
  const size = 36 * u;
  ctx.font = `600 ${size}px system-ui, -apple-system, Segoe UI, sans-serif`;
  const maxW = W * 0.9;
  const words = text.split(' '), lines = [];
  let line = '';
  for (const w of words) {
    const tryL = line ? `${line} ${w}` : w;
    if (ctx.measureText(tryL).width > maxW && line) { lines.push(line); line = w; } else line = tryL;
  }
  if (line) lines.push(line);
  const lh = size * 1.3, pad = 16 * u;
  const boxH = lines.length * lh + pad * 2, boxW = Math.min(W - 24 * u, Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2);
  const x = (W - boxW) / 2, y = H - boxH - 22 * u;
  ctx.fillStyle = 'rgba(5,6,7,0.84)';
  ctx.beginPath(); ctx.roundRect(x, y, boxW, boxH, 14 * u); ctx.fill();
  ctx.fillStyle = '#b9d4ff'; ctx.fillRect(x, y + 12 * u, 4 * u, boxH - 24 * u);
  ctx.fillStyle = '#f4f6f8'; ctx.textBaseline = 'top';
  lines.forEach((l, i) => ctx.fillText(l, x + pad, y + pad + i * lh));
  ctx.restore();
}
