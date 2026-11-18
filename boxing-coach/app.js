// Boxing Coach: pages, live sessions and Teach mode.
// All camera processing happens on this device. Nothing is uploaded.

import { FilesetResolver, PoseLandmarker } from './vendor/vision_bundle.mjs';
import { LSTMClassifier } from './lstm.js';
import { Engine, MP_IDX } from './engine.js';
import {
  LABELS, TARGET, loadModel, saveModel, emptyModel, counts, isReady, KNN, evaluate, speedThreshold,
} from './personal.js';
import { startPlexus } from './plexus.js';

const CFG = {
  COMBO_TIMEOUT: 3.0, ENCOURAGE_EVERY: 5, VOICE_COOLDOWN: 1.6, REST_S: 60,
  PERSONAL_MIN_CONF: 0.5, TEACH_READY_S: 3, TEACH_GAP_S: 0.45,
};
const PLURAL = { jab: 'jabs', cross: 'crosses', hook: 'hooks', uppercut: 'uppercuts' };
const CALLS = {
  jab: ['Clean jab', 'Sharp jab', 'Good jab', 'Nice jab'],
  cross: ['Powerful cross', 'Good cross', 'Nice cross', 'Strong cross'],
  hook: ['Solid hook', 'Good hook', 'Nice hook'],
  uppercut: ['Solid uppercut', 'Good uppercut', 'Nice uppercut', 'Strong uppercut'],
};
const ENCOURAGE = ['Keep it up', 'Stay sharp', 'Hands up, keep going', 'Nice rhythm', 'Good work, keep moving'];
const SKELETON = [[0, 1], [0, 2], [2, 4], [1, 3], [3, 5], [0, 6], [1, 7], [6, 7]];

const $ = (id) => document.getElementById(id);
const el = {
  stage: $('stage'), cam: $('cam'), overlay: $('overlay'),
  hudTrain: $('hud-train'), hudTeach: $('hud-teach'), hint: $('hint'),
  btnStart: $('btn-start'), file: $('file-input'), status: $('status'),
  modelCard: $('model-card'), modelDot: $('model-dot'), modelTitle: $('model-title'), modelSub: $('model-sub'),
  rowModel: $('row-model'), lastCard: $('last-card'), lastStats: $('last-stats'),
  clock: $('clock'), roundLabel: $('round-label'), punchCount: $('punch-count'), comboCount: $('combo-count'),
  stance: $('stance'), modelPill: $('model-pill'), callout: $('callout'), recent: $('recent'),
  btnVoice: $('btn-voice'), btnFlip: $('btn-flip'), btnEnd: $('btn-end'),
  teachStep: $('teach-step'), teachLabel: $('teach-label'), teachRing: $('teach-ring'), teachCount: $('teach-count'),
  teachCenter: $('teach-center'), teachMsg: $('teach-msg'),
  btnUndo: $('btn-undo'), btnSkip: $('btn-skip'), btnFinish: $('btn-finish'),
  btnTeach: $('btn-teach'), btnClear: $('btn-clear'), punchList: $('punch-list'),
  scoreCard: $('score-card'), scoreNum: $('score-num'), scoreNote: $('score-note'),
  sheet: $('sheet'), sheetTitle: $('sheet-title'), sheetBody: $('sheet-body'), sheetClose: $('btn-sheet-close'),
};
const octx = el.overlay.getContext('2d');
const plexus = startPlexus($('bg'));

// ---------------------------------------------------------------- settings (per device)
const settings = Object.assign({ camera: 'user', round: '180', voice: 'on', model: 'personal' }, readJSON('bc.settings'));
function saveSettings() { writeJSON('bc.settings', settings); }
const segSyncs = [];
document.querySelectorAll('.seg').forEach((seg) => {
  const key = seg.dataset.setting;
  const sync = () => seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === settings[key])));
  seg.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    settings[key] = b.dataset.value; saveSettings(); sync(); refreshModelUI();
  });
  segSyncs.push(sync);
  sync();
});

// ---------------------------------------------------------------- pages
const PAGES = ['analyse', 'teach', 'tips', 'lessons'];
function route() {
  const page = PAGES.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'analyse';
  document.querySelectorAll('.page').forEach((p) => { p.hidden = p.dataset.page !== page; });
  document.querySelectorAll('.tabbar a').forEach((a) => {
    if (a.dataset.tab === page) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  window.scrollTo(0, 0);
}
addEventListener('hashchange', route);
route();

// ---------------------------------------------------------------- models
let engine = null;
let personal = loadModel();
let knn = null, evalResult = null;
function rebuildPersonal() {
  knn = isReady(personal) ? new KNN(personal.examples) : null;
  evalResult = personal.examples.length >= 8 ? evaluate(personal) : null;
}
rebuildPersonal();

const ready = (async () => {
  const [vision, starter] = await Promise.all([
    FilesetResolver.forVisionTasks('./vendor/wasm'),
    LSTMClassifier.load('./models/punch_lstm.bin'),
  ]);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: './models/pose_landmarker_full.task', delegate },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  let pose;
  try { pose = await PoseLandmarker.createFromOptions(vision, opts('GPU')); }
  catch (e) { console.warn('GPU unavailable, using CPU', e); pose = await PoseLandmarker.createFromOptions(vision, opts('CPU')); }
  engine = new Engine(pose, starter);
})();

ready.then(() => { refreshModelUI(); }).catch((e) => {
  console.error(e);
  setStatus('The coach could not load. Check your connection and reload the page.', true);
  el.btnStart.textContent = 'Could not load'; el.btnTeach.textContent = 'Could not load';
});

function usingPersonal() { return !!knn && settings.model !== 'starter'; }

function refreshModelUI() {
  const c = counts(personal);
  const readyP = !!knn;
  const loaded = !!engine;
  // Analyse page
  el.modelDot.classList.toggle('ready', usingPersonal());
  if (usingPersonal()) {
    el.modelTitle.textContent = 'Your model';
    el.modelSub.textContent = `${personal.examples.length} punches taught` + (evalResult?.accuracy != null ? ` · ${pct(evalResult.accuracy)} self-test` : '');
  } else if (readyP) {
    el.modelTitle.textContent = 'Starter model';
    el.modelSub.textContent = 'Your model is switched off below';
  } else {
    el.modelTitle.textContent = 'Starter model';
    el.modelSub.textContent = 'Teach it your punches for better accuracy';
  }
  el.rowModel.hidden = !readyP;
  el.btnStart.disabled = !loaded;
  if (loaded) el.btnStart.textContent = 'Start training';

  // Teach page
  el.punchList.replaceChildren(...LABELS.map((l) => {
    const li = document.createElement('li'); li.className = 'punch-row';
    const name = document.createElement('span'); name.className = 'punch-name'; name.textContent = l;
    const bar = document.createElement('div'); bar.className = 'bar' + (c[l] >= TARGET ? ' full' : '');
    const fill = document.createElement('i'); fill.style.width = `${Math.min(100, (c[l] / TARGET) * 100)}%`; bar.append(fill);
    const n = document.createElement('span'); n.className = 'punch-n'; n.textContent = `${c[l]}/${TARGET}`;
    const redo = document.createElement('button'); redo.type = 'button'; redo.className = 'mini';
    redo.textContent = c[l] ? 'Redo' : 'Teach'; redo.disabled = !loaded;
    redo.addEventListener('click', () => startTeach([{ label: l, need: TARGET }], l));
    li.append(name, bar, n, redo);
    return li;
  }));
  const total = personal.examples.length;
  const allFull = LABELS.every((l) => c[l] >= TARGET);
  el.btnTeach.disabled = !loaded;
  if (loaded) el.btnTeach.textContent = total === 0 ? 'Start teaching' : allFull ? 'Add 5 more of each' : 'Continue teaching';
  el.btnClear.hidden = total === 0;
  el.btnClear.classList.remove('confirm'); el.btnClear.textContent = 'Delete my model';

  if (evalResult && evalResult.accuracy != null) {
    el.scoreCard.hidden = false;
    el.scoreNum.textContent = pct(evalResult.accuracy);
    el.scoreNote.textContent = weakestNote(evalResult) || 'Every punch type is being told apart well.';
  } else el.scoreCard.hidden = true;

  renderLast();
}

function weakestNote(r) {
  let worst = null;
  for (const l of LABELS) {
    const p = r.per[l];
    if (!p.total) continue;
    const acc = p.correct / p.total;
    if (!worst || acc < worst.acc) worst = { l, acc, p };
  }
  if (!worst || worst.acc >= 0.85) return '';
  const conf = Object.entries(worst.p.confused).sort((a, b) => b[1] - a[1])[0];
  return `${cap(PLURAL[worst.l])} are weakest (${pct(worst.acc)})` + (conf ? `, mostly mistaken for ${PLURAL[conf[0]]}` : '') + '. Redo them for a boost.';
}

el.modelCard.addEventListener('click', () => { location.hash = '#teach'; });

// ---------------------------------------------------------------- session state
let mode = null;      // 'train' | 'file' | 'teach'
let S = null;         // training session
let T = null;         // teach session
let running = false, rafId = 0, lastVideoTime = -1, stream = null, wakeLock = null;
let facing = 'user', voiceOn = true;
let lastLandmarks = null, highlightUntil = 0, nowT = 0;

function newTrainSession(src) {
  return {
    src, punches: 0, combo: 0, bestCombo: 0, lastPunchT: -Infinity,
    byType: Object.fromEntries(LABELS.map((l) => [l, 0])), history: [],
    startT: null, nowT: 0, lastVoiceT: -Infinity,
    roundLen: src === 'live' ? Number(settings.round) || 0 : 0,
    phase: 'work', phaseKey: '', model: usingPersonal() ? 'personal' : 'starter',
  };
}

// ---------------------------------------------------------------- frame loop
function loop() {
  if (!running) return;
  const v = el.cam;
  if (v.readyState >= 2 && v.videoWidth && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    const t = mode === 'file' ? v.currentTime : performance.now() / 1000;
    try { step(v, v.videoWidth, v.videoHeight, t); } catch (e) { console.error(e); }
  }
  draw();
  if (mode === 'file' && v.ended) { endTrain(); return; }
  rafId = requestAnimationFrame(loop);
}

function step(source, w, h, t) {
  nowT = t;
  const r = engine.process(source, w, h, t);
  lastLandmarks = r.landmarks;
  el.hint.hidden = r.tracking;
  if (mode === 'teach') return teachStep(r, t);

  if (S.startT === null) S.startT = t;
  S.nowT = t;
  updatePhase();
  el.stance.textContent = r.stance || 'Finding your stance…';
  if (S.phase !== 'work') return;
  for (const ev of r.events) {
    if (S.model === 'starter' && ev.kind === 'punch') onPunch(ev.label, ev.confidence, t);
    if (S.model === 'personal' && ev.kind === 'motion') {
      const c = knn.classify(ev.features);
      if (c && !c.rejected && c.confidence >= CFG.PERSONAL_MIN_CONF) onPunch(c.label, c.confidence, t);
    }
  }
}

// ---------------------------------------------------------------- training mode
function onPunch(type, confidence, t) {
  S.punches++;
  S.combo = t - S.lastPunchT < CFG.COMBO_TIMEOUT ? S.combo + 1 : 1;
  S.bestCombo = Math.max(S.bestCombo, S.combo);
  S.lastPunchT = t;
  S.byType[type]++;
  S.history.push(type); if (S.history.length > 5) S.history.shift();
  highlightUntil = t + 0.4;
  const quality = confidence > 0.9 ? 'Excellent' : confidence > 0.75 ? 'Good' : 'Nice';
  showCallout(quality, type);
  renderCounts();
  if (S.combo >= 3) speak(`${S.combo} punch combo`, t);
  else if (S.punches % CFG.ENCOURAGE_EVERY === 0) speak(pick(ENCOURAGE), t);
  else speak(pick(CALLS[type]), t);
}

function updatePhase() {
  const elapsed = S.nowT - S.startT;
  let label = 'Session', shown = elapsed, phase = 'work';
  if (S.roundLen) {
    const cycle = S.roundLen + CFG.REST_S;
    const n = Math.floor(elapsed / cycle);
    const inCycle = elapsed - n * cycle;
    if (inCycle < S.roundLen) { label = `Round ${n + 1}`; shown = S.roundLen - inCycle; }
    else { phase = 'rest'; label = 'Rest'; shown = cycle - inCycle; }
    const key = `${phase}${n}`;
    if (key !== S.phaseKey) {
      if (S.phaseKey) bell(phase === 'work' ? 3 : 1);
      if (phase === 'rest') speakNow('Rest');
      else if (n > 0) speakNow(`Round ${n + 1}. Box`);
      S.phaseKey = key;
    }
  }
  S.phase = phase;
  el.roundLabel.textContent = label;
  el.clock.textContent = fmt(shown);
  el.clock.parentElement.classList.toggle('rest', phase === 'rest');
}

function renderCounts() {
  el.punchCount.textContent = S.punches;
  el.comboCount.textContent = S.combo;
  el.recent.replaceChildren(...S.history.slice(-3).map((p) => { const li = document.createElement('li'); li.textContent = p; return li; }));
}

function showCallout(quality, type) {
  el.callout.classList.remove('show');
  const q = document.createElement('span'); q.className = 'q'; q.textContent = quality;
  el.callout.replaceChildren(q, document.createTextNode(type));
  void el.callout.offsetWidth;
  el.callout.classList.add('show');
}

async function startTrain() {
  unlockAudio(); setStatus('');
  el.btnStart.disabled = true; el.btnStart.textContent = 'Starting camera…';
  try { await ready; facing = settings.camera; await openCamera(); }
  catch (e) { console.error(e); setStatus(cameraError(e), true); refreshModelUI(); return; }
  refreshModelUI();
  beginTrain('live');
}

async function startFile(file) {
  unlockAudio(); setStatus('');
  try { await ready; } catch { return; }
  stopStream();
  el.cam.srcObject = null;
  el.cam.src = URL.createObjectURL(file);
  setMirror(false);
  try { await el.cam.play(); }
  catch (e) { console.error(e); setStatus('That video could not be played on this phone.', true); return; }
  beginTrain('file');
}

function beginTrain(src) {
  mode = src === 'live' ? 'train' : 'file';
  S = newTrainSession(src);
  voiceOn = settings.voice === 'on';
  engine.reset();
  engine.useStarter = S.model === 'starter';
  engine.motionSpeed = speedThreshold(personal, 0.1);
  el.modelPill.textContent = S.model === 'personal' ? 'Your model' : 'Starter model';
  el.btnFlip.hidden = src !== 'live';
  renderCounts();
  el.callout.classList.remove('show');
  el.stance.textContent = 'Finding you…';
  el.clock.textContent = '00:00';
  setVoiceButton();
  showStage('train');
  if (S.roundLen) bell(3);
}

function endTrain() {
  if (!running) return;
  stopStage();
  const secs = Math.max(0, S.nowT - (S.startT ?? S.nowT));
  const summary = {
    when: Date.now(), secs, punches: S.punches, best: S.bestCombo,
    ppm: secs >= 10 ? Math.round(S.punches / (secs / 60)) : null, byType: S.byType, model: S.model,
  };
  if (S.src === 'live' && secs >= 10) writeJSON('bc.last', summary);
  renderLast();
  openSheet('Session done', sessionBody(summary));
}

function sessionBody(s) {
  const frag = document.createDocumentFragment();
  frag.append(stats([[s.punches, 'Punches'], [fmt(s.secs), 'Time'], [s.best, 'Best combo'], [s.ppm ?? '–', 'Per minute']]));
  const max = Math.max(1, ...Object.values(s.byType));
  const bars = document.createElement('div'); bars.className = 'bars';
  for (const l of LABELS) {
    const row = document.createElement('div'); row.className = 'brow';
    const n = document.createElement('span'); n.className = 'bname'; n.textContent = l;
    const b = document.createElement('div'); b.className = 'bar';
    const i = document.createElement('i'); i.style.width = `${(s.byType[l] / max) * 100}%`; b.append(i);
    const c = document.createElement('span'); c.className = 'bn'; c.textContent = s.byType[l];
    row.append(n, b, c); bars.append(row);
  }
  const note = document.createElement('p'); note.className = 'sheet-note';
  note.textContent = s.model === 'personal' ? 'Called by your model.' : 'Called by the starter model. Teach it your punches for better accuracy.';
  frag.append(bars, note);
  return frag;
}

function renderLast() {
  const s = readJSON('bc.last');
  if (!s) { el.lastCard.hidden = true; return; }
  el.lastCard.hidden = false;
  el.lastStats.replaceChildren(...[[s.punches, 'Punches'], [fmt(s.secs), 'Time'], [s.best, 'Best combo'], [s.ppm ?? '–', 'Per min']].map(statEl));
}

// ---------------------------------------------------------------- teach mode
async function startTeach(plan, redoLabel) {
  unlockAudio();
  try { await ready; facing = settings.camera; await openCamera(); }
  catch (e) { console.error(e); location.hash = '#analyse'; setStatus(cameraError(e), true); return; }
  if (redoLabel) personal.examples = personal.examples.filter((e) => e.l !== redoLabel);
  mode = 'teach';
  voiceOn = settings.voice === 'on';
  engine.reset();
  engine.useStarter = false;
  engine.motionSpeed = 0.1;
  T = { plan, i: -1, got: 0, need: 0, readyUntil: 0, added: [], label: null, wentGo: false, changed: !!redoLabel };
  setVoiceButton();
  showStage('teach');
  nextTeachLabel(performance.now() / 1000);
}

function teachPlan() {
  const c = counts(personal);
  const allFull = LABELS.every((l) => c[l] >= TARGET);
  return LABELS.map((l) => ({ label: l, need: allFull ? 5 : Math.max(0, TARGET - c[l]) })).filter((p) => p.need > 0);
}

function nextTeachLabel(t) {
  T.i++;
  if (T.i >= T.plan.length) { finishTeach(); return; }
  const { label, need } = T.plan[T.i];
  Object.assign(T, { label, need, got: 0, readyUntil: t + CFG.TEACH_READY_S, wentGo: false, lastCapT: -Infinity });
  el.teachStep.textContent = `Punch ${T.i + 1} of ${T.plan.length}`;
  el.teachLabel.textContent = label;
  el.teachMsg.textContent = `Get ready to throw ${need} ${PLURAL[label]}`;
  updateRing();
  speakNow(`${need} ${PLURAL[label]}. Get ready.`);
}

function teachStep(r, t) {
  // countdown
  if (t < T.readyUntil) {
    el.teachCenter.className = 'teach-center';
    el.teachCenter.textContent = String(Math.ceil(T.readyUntil - t));
    return;
  }
  if (!T.wentGo) {
    T.wentGo = true;
    flashCenter('Go');
    bell(1);
    el.teachMsg.textContent = `Throw a ${T.label}, then back to guard`;
  }
  for (const ev of r.events) {
    if (ev.kind !== 'motion' || ev.t - T.lastCapT < CFG.TEACH_GAP_S) continue;
    const ex = { l: T.label, f: ev.features, s: ev.speed };
    personal.examples.push(ex);
    T.added.push(ex); T.changed = true;
    T.got++; T.lastCapT = ev.t;
    highlightUntil = t + 0.35;
    tick();
    flashCenter(`${T.got}`);
    updateRing();
    if (T.got >= T.need) {
      el.teachMsg.textContent = 'Nice. Next punch…';
      nextTeachLabel(t);
      return;
    }
  }
}

function updateRing() {
  const p = T.need ? (T.got / T.need) * 100 : 0;
  el.teachRing.style.strokeDasharray = `${p} 100`;
  el.teachCount.textContent = `${T.got}/${T.need}`;
}

function flashCenter(text) {
  el.teachCenter.className = 'teach-center' + (text.length > 2 ? ' small' : '');
  el.teachCenter.textContent = text;
  void el.teachCenter.offsetWidth;
  el.teachCenter.classList.add('flash');
}

function undoTeach() {
  if (!T || !T.got) return;
  const ex = T.added.pop();
  personal.examples.splice(personal.examples.lastIndexOf(ex), 1);
  T.got--; updateRing();
  el.teachMsg.textContent = 'Removed the last one';
}

function finishTeach() {
  if (!running) return;
  stopStage();
  if (!T.changed) { refreshModelUI(); return; }
  const saved = saveModel(personal);
  rebuildPersonal();
  if (knn) { settings.model = 'personal'; saveSettings(); }
  segSyncs.forEach((f) => f());
  refreshModelUI();
  openSheet('Model updated', teachBody(saved), () => { location.hash = '#teach'; });
}

function teachBody(saved) {
  const frag = document.createDocumentFragment();
  const c = counts(personal);
  if (evalResult && evalResult.accuracy != null) {
    frag.append(stats([[pct(evalResult.accuracy), 'Self-test'], [personal.examples.length, 'Examples']]));
  }
  const bars = document.createElement('div'); bars.className = 'bars';
  for (const l of LABELS) {
    const p = evalResult?.per[l];
    const acc = p && p.total ? p.correct / p.total : null;
    const row = document.createElement('div'); row.className = 'brow';
    const n = document.createElement('span'); n.className = 'bname'; n.textContent = l;
    const b = document.createElement('div'); b.className = 'bar' + (acc != null && acc >= 0.85 ? ' full' : '');
    const i = document.createElement('i'); i.style.width = `${acc != null ? acc * 100 : 0}%`; b.append(i);
    const v = document.createElement('span'); v.className = 'bn'; v.textContent = acc != null ? pct(acc) : `${c[l]}`;
    row.append(n, b, v); bars.append(row);
  }
  frag.append(bars);
  const note = document.createElement('p'); note.className = 'sheet-note';
  if (!saved) note.textContent = 'Could not save on this phone (private browsing?). Your model will be lost when you close the app.';
  else if (!knn) note.textContent = `Teach at least 5 of each punch to switch on your model.`;
  else note.textContent = (evalResult && weakestNote(evalResult)) || 'Your model is on. Head to Analyse and start training.';
  frag.append(note);
  return frag;
}

el.btnTeach.addEventListener('click', () => {
  const plan = teachPlan();
  if (plan.length) startTeach(plan);
});
el.btnClear.addEventListener('click', () => {
  if (!el.btnClear.classList.contains('confirm')) {
    el.btnClear.classList.add('confirm');
    el.btnClear.textContent = 'Tap again to delete all examples';
    setTimeout(() => { el.btnClear.classList.remove('confirm'); el.btnClear.textContent = 'Delete my model'; }, 4000);
    return;
  }
  personal = emptyModel(); saveModel(personal); rebuildPersonal(); refreshModelUI();
});
el.btnUndo.addEventListener('click', undoTeach);
el.btnSkip.addEventListener('click', () => { if (T) nextTeachLabel(nowT || performance.now() / 1000); });
el.btnFinish.addEventListener('click', finishTeach);

// ---------------------------------------------------------------- stage
function showStage(which) {
  el.hudTrain.hidden = which !== 'train';
  el.hudTeach.hidden = which !== 'teach';
  el.teachCenter.textContent = '';
  el.stage.hidden = false;
  document.body.style.overflow = 'hidden';
  plexus.pause();
  running = true; lastVideoTime = -1; lastLandmarks = null;
  requestWakeLock();
  rafId = requestAnimationFrame(loop);
}

function stopStage() {
  running = false;
  cancelAnimationFrame(rafId);
  stopStream();
  el.cam.pause();
  if (el.cam.getAttribute('src')) { URL.revokeObjectURL(el.cam.src); el.cam.removeAttribute('src'); el.cam.load(); }
  releaseWakeLock();
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  el.stage.hidden = true;
  document.body.style.overflow = '';
  plexus.resume();
}

function draw() {
  const cw = el.overlay.clientWidth, ch = el.overlay.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (el.overlay.width !== Math.round(cw * dpr) || el.overlay.height !== Math.round(ch * dpr)) {
    el.overlay.width = Math.round(cw * dpr); el.overlay.height = Math.round(ch * dpr);
  }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, cw, ch);
  if (!lastLandmarks) return;
  const { lm, w, h } = lastLandmarks;
  const sc = Math.max(cw / w, ch / h);
  const ox = (cw - w * sc) / 2, oy = (ch - h * sc) / 2;
  const P = MP_IDX.map((mi) => ({ x: ox + lm[mi].x * w * sc, y: oy + lm[mi].y * h * sc, v: lm[mi].visibility ?? 1 }));
  const hot = nowT < highlightUntil;
  const col = hot ? '#7ee2b0' : 'rgba(255,255,255,0.9)';
  octx.lineCap = 'round';
  octx.shadowColor = hot ? '#7ee2b0' : 'rgba(185,212,255,0.8)';
  octx.shadowBlur = 12;
  octx.strokeStyle = col; octx.lineWidth = 2.5;
  for (const [a, b] of SKELETON) {
    if (P[a].v < 0.5 || P[b].v < 0.5) continue;
    octx.beginPath(); octx.moveTo(P[a].x, P[a].y); octx.lineTo(P[b].x, P[b].y); octx.stroke();
  }
  P.forEach((p, i) => {
    if (p.v < 0.5) return;
    const wrist = i === 4 || i === 5;
    octx.fillStyle = wrist ? (hot ? '#7ee2b0' : '#b9d4ff') : col;
    octx.beginPath(); octx.arc(p.x, p.y, wrist ? 9 : 4.5, 0, Math.PI * 2); octx.fill();
  });
  octx.shadowBlur = 0;
}

// ---------------------------------------------------------------- camera
function setMirror(on) { el.cam.classList.toggle('mirror', on); el.overlay.classList.toggle('mirror', on); }

async function openCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw Object.assign(new Error('insecure'), { name: 'InsecureContext' });
  }
  stopStream();
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
  });
  el.cam.removeAttribute('src');
  el.cam.srcObject = stream;
  setMirror(facing === 'user');
  await el.cam.play();
}

function stopStream() {
  if (stream) stream.getTracks().forEach((tr) => tr.stop());
  stream = null;
}

function cameraError(e) {
  const name = e && e.name;
  if (name === 'NotAllowedError') return 'Camera access is blocked. On iPhone open Settings › Safari › Camera, choose Allow, then reload.';
  if (name === 'InsecureContext') return 'The camera only works when the app is opened from its https:// link.';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found on this device.';
  return 'The camera could not start. Close other apps using it and try again.';
}

// ---------------------------------------------------------------- sheet
let sheetOnClose = null;
function openSheet(title, body, onClose) {
  el.sheetTitle.textContent = title;
  el.sheetBody.replaceChildren(body);
  el.sheet.hidden = false;
  sheetOnClose = onClose || null;
  el.sheetClose.focus();
}
function closeSheet() {
  el.sheet.hidden = true;
  const f = sheetOnClose; sheetOnClose = null; if (f) f();
}
el.sheetClose.addEventListener('click', closeSheet);
el.sheet.addEventListener('click', (e) => { if (e.target === el.sheet) closeSheet(); });

function stats(pairs) { const d = document.createElement('div'); d.className = 'stats4'; d.append(...pairs.map(statEl)); return d; }
function statEl([v, k]) {
  const d = document.createElement('div'); d.className = 'stat';
  const a = document.createElement('span'); a.className = 'v'; a.textContent = v;
  const b = document.createElement('span'); b.className = 'k'; b.textContent = k;
  d.append(a, b); return d;
}

// ---------------------------------------------------------------- voice + sounds
function setVoiceButton() {
  el.btnVoice.textContent = voiceOn ? 'Voice on' : 'Voice off';
  el.btnVoice.setAttribute('aria-pressed', String(voiceOn));
}
function speak(text, t) {
  if (!voiceOn || !('speechSynthesis' in window)) return;
  if (t - S.lastVoiceT < CFG.VOICE_COOLDOWN) return;
  S.lastVoiceT = t; speakNow(text);
}
function speakNow(text) {
  if (!voiceOn || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); u.rate = 1.1; u.lang = 'en-GB';
  speechSynthesis.speak(u);
}
let audioCtx = null;
function unlockAudio() {
  try {
    if ('speechSynthesis' in window) { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); }
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume();
  } catch { /* sound is optional */ }
}
function tone(freq, start, dur, vol, type = 'sine') {
  if (!audioCtx) return;
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, start);
  g.gain.exponentialRampToValueAtTime(vol, start + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  o.connect(g).connect(audioCtx.destination); o.start(start); o.stop(start + dur + 0.05);
}
function bell(times) { if (!audioCtx) return; for (let i = 0; i < times; i++) tone(1320, audioCtx.currentTime + i * 0.35, 0.9, 0.5, 'triangle'); }
function tick() { if (!audioCtx) return; tone(880, audioCtx.currentTime, 0.12, 0.35); tone(1760, audioCtx.currentTime + 0.06, 0.15, 0.2); }

// ---------------------------------------------------------------- wake lock
async function requestWakeLock() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { wakeLock = null; } }
function releaseWakeLock() { try { wakeLock?.release(); } catch { /* ignore */ } wakeLock = null; }
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && running) requestWakeLock(); });

// ---------------------------------------------------------------- helpers
function fmt(s) { s = Math.max(0, Math.floor(s)); return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; }
function pct(x) { return `${Math.round(x * 100)}%`; }
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function pick(a) { return a[Math.floor(Math.random() * a.length)]; }
function setStatus(msg, isError = false) { el.status.textContent = msg; el.status.classList.toggle('error', isError); }
function readJSON(k) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : null; } catch { return null; } }
function writeJSON(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } }

// ---------------------------------------------------------------- events
el.btnStart.addEventListener('click', startTrain);
el.file.addEventListener('change', () => { const f = el.file.files[0]; el.file.value = ''; if (f) startFile(f); });
el.btnEnd.addEventListener('click', endTrain);
el.btnVoice.addEventListener('click', () => {
  voiceOn = !voiceOn;
  if (!voiceOn && 'speechSynthesis' in window) speechSynthesis.cancel();
  setVoiceButton();
});
el.btnFlip.addEventListener('click', async () => {
  facing = facing === 'user' ? 'environment' : 'user';
  engine.reset();
  try { await openCamera(); } catch (e) { console.error(e); facing = facing === 'user' ? 'environment' : 'user'; }
});

refreshModelUI();

// Offline support once installed from an https link.
if ('serviceWorker' in navigator && location.protocol === 'https:') {
  navigator.serviceWorker.register('./sw.js').catch(() => {});
}

// Developer test hook: feed frames from any image/canvas source without a camera.
window.boxingCoach = {
  ready,
  startTest(m = 'train', plan) {
    mode = m; running = true; engine.reset();
    if (m === 'teach') {
      engine.useStarter = false; engine.motionSpeed = 0.1;
      T = { plan: plan || teachPlan(), i: -1, got: 0, need: 0, readyUntil: 0, added: [], label: null, wentGo: false, changed: false };
      nextTeachLabel(0); T.readyUntil = 0;
    } else {
      S = newTrainSession('file');
      engine.useStarter = S.model === 'starter'; engine.motionSpeed = speedThreshold(personal, 0.1);
    }
    return mode;
  },
  feed(source, w, h, t) {
    const before = mode === 'teach' ? T.added.length : S.punches;
    step(source, w, h, t);
    return mode === 'teach' ? { label: T.label, got: T.got, added: T.added.length - before } : { punches: S.punches, last: S.history.at(-1), new: S.punches - before };
  },
  finish() { running = true; return mode === 'teach' ? finishTeach() : endTrain(); },
  get personal() { return personal; },
  get evalResult() { return evalResult; },
};
