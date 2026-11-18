// Boxing Coach: pages, live sessions and Teach mode.
// All camera processing happens on this device. Nothing is uploaded.

import { FilesetResolver, PoseLandmarker } from './vendor/vision_bundle.mjs';
import { LSTMClassifier } from './lstm.js';
import { Engine, MP_IDX } from './engine.js';
import {
  LABELS, NAMES, ROLE, sidedLabel, TARGET, COMBOS, COMBO_TARGET, loadModel, saveModel, emptyModel, counts, comboReps, isReady,
  KNN, evaluate, speedThreshold, stanceName, roleOf, comboGap, comboPrior, matchCombo,
} from './personal.js';
import { startPlexus } from './plexus.js';

const CFG = {
  COMBO_GAP: 1.2, ENCOURAGE_EVERY: 5, VOICE_COOLDOWN: 1.6, REST_S: 60,
  PERSONAL_MIN_CONF: 0.5, TEACH_READY_S: 3, TEACH_GAP_S: 0.45,
  TEACH_COMBO_END_S: 0.75,  // pause after the last punch that ends a taught combo
  TEACH_STANCE_FRAMES: 30,  // still guard frames needed to read the stance (~1 s)
  TEACH_STANCE_TIMEOUT: 10, TEACH_NEXT_PAUSE: 0.9,
  PRIOR_WEIGHT: 0.3,        // how much a taught combo can sway the next punch's call
};
const PLURAL = {
  jab: 'jabs', cross: 'crosses', lead_hook: 'lead hooks', rear_hook: 'rear hooks',
  lead_uppercut: 'lead uppercuts', rear_uppercut: 'rear uppercuts',
};
const CALLS = {
  jab: ['Clean jab', 'Sharp jab', 'Good jab', 'Nice jab'],
  cross: ['Powerful cross', 'Good cross', 'Nice cross', 'Strong cross'],
  lead_hook: ['Solid lead hook', 'Good lead hook', 'Nice lead hook'],
  rear_hook: ['Solid rear hook', 'Good rear hook', 'Big rear hook'],
  lead_uppercut: ['Good lead uppercut', 'Nice lead uppercut', 'Sharp lead uppercut'],
  rear_uppercut: ['Solid rear uppercut', 'Strong rear uppercut', 'Nice rear uppercut'],
};
const lower = (l) => NAMES[l].toLowerCase();
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
  btnTeach: $('btn-teach'), btnClear: $('btn-clear'), punchList: $('punch-list'), comboList: $('combo-list'),
  stanceNow: $('stance-now'), stanceSeg: $('stance-seg'), stanceSub: $('stance-sub'),
  scoreCard: $('score-card'), scoreNum: $('score-num'), scoreNote: $('score-note'),
  sheet: $('sheet'), sheetTitle: $('sheet-title'), sheetBody: $('sheet-body'), sheetClose: $('btn-sheet-close'),
};
const octx = el.overlay.getContext('2d');
const plexus = startPlexus($('bg'));

// ---------------------------------------------------------------- settings (per device)
const settings = Object.assign({ camera: 'user', round: '180', voice: 'on', model: 'personal' }, readJSON('bc.settings'));
function saveSettings() { writeJSON('bc.settings', settings); }
const segSyncs = [];
document.querySelectorAll('.seg[data-setting]').forEach((seg) => {
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
  const st = personal.stance;
  el.stanceNow.textContent = st ? `${stanceName(st.lead)} · ${st.lead} hand leads` : 'Not read yet';
  el.stanceSub.textContent = st
    ? (st.source === 'manual' ? 'Set by you. Jabs are your lead hand, crosses your rear hand.' : 'Read from your guard. Tap the other side if it got it wrong.')
    : 'Read from your guard at the start of teaching. Tap to set it yourself.';
  el.stanceSeg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(!!st && b.dataset.value === st.lead)));

  const cs = counts(personal, true);
  el.punchList.replaceChildren(...LABELS.map((l) => teachRow(NAMES[l], cs[l], TARGET, loaded,
    () => startTeach(withStance([{ type: 'punch', label: l, need: TARGET }]), { type: 'punch', id: l }))));
  const reps = comboReps(personal);
  el.comboList.replaceChildren(...COMBOS.map((cb) => teachRow(cb.name, reps[cb.id], COMBO_TARGET, loaded,
    () => startTeach(withStance([{ type: 'combo', combo: cb, need: COMBO_TARGET }]), { type: 'combo', id: cb.id }), 'combo-row')));
  const total = personal.examples.length;
  el.btnTeach.disabled = !loaded;
  if (loaded) el.btnTeach.textContent = total === 0 ? 'Start teaching' : nothingMissing() ? 'Add 5 more of each' : 'Continue teaching';
  el.btnClear.hidden = total === 0;
  el.btnClear.classList.remove('confirm'); el.btnClear.textContent = 'Delete my model';

  if (evalResult && evalResult.accuracy != null) {
    el.scoreCard.hidden = false;
    el.scoreNum.textContent = pct(evalResult.accuracy);
    el.scoreNote.textContent = weakestNote(evalResult) || 'Every punch type is being told apart well.';
  } else el.scoreCard.hidden = true;

  renderLast();
}

function teachRow(label, n, target, loaded, onClick, extra = '') {
  const li = document.createElement('li'); li.className = 'punch-row' + (extra ? ` ${extra}` : '');
  const name = document.createElement('span'); name.className = 'punch-name'; name.textContent = label;
  const bar = document.createElement('div'); bar.className = 'bar' + (n >= target ? ' full' : '');
  const fill = document.createElement('i'); fill.style.width = `${Math.min(100, (n / target) * 100)}%`; bar.append(fill);
  const num = document.createElement('span'); num.className = 'punch-n'; num.textContent = `${n}/${target}`;
  const redo = document.createElement('button'); redo.type = 'button'; redo.className = 'mini';
  redo.textContent = n ? 'Redo' : 'Teach'; redo.disabled = !loaded;
  redo.addEventListener('click', onClick);
  li.append(name, bar, num, redo);
  return li;
}

// Manual stance override on the Teach page.
el.stanceSeg.addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  personal.stance = { lead: b.dataset.value, source: 'manual' };
  saveModel(personal); rebuildPersonal(); refreshModelUI();
});

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
    comboGap: comboGap(personal, CFG.COMBO_GAP), comboSeq: [], comboDone: true,
    comboHits: Object.fromEntries(COMBOS.map((c) => [c.id, 0])),
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
  const lead = r.lead || personal.stance?.lead || null;
  el.stance.textContent = stanceText(r, lead);
  if (S.phase !== 'work') return;
  checkComboEnd(t);
  for (const ev of r.events) {
    // The starter model only knows "hook" / "uppercut": the hand that moved says lead or rear.
    if (S.model === 'starter' && ev.kind === 'punch') onPunch(sidedLabel(ev.label, roleOf(ev.hand, lead)), ev.confidence, t);
    if (S.model === 'personal' && ev.kind === 'motion') {
      const c = knn.classify(ev.features, -1, roleOf(ev.hand, lead));
      if (!c || c.rejected) continue;
      const { label, confidence } = withComboPrior(c, t);
      if (confidence >= CFG.PERSONAL_MIN_CONF) onPunch(label, confidence, t);
    }
  }
}

function stanceText(r, lead) {
  if (!lead) return r.tracking ? 'Finding your stance…' : 'Finding you…';
  const mine = personal.stance?.lead;
  if (mine && r.lead && r.lead !== mine) return `${stanceName(r.lead)} · switched`;
  return stanceName(lead);
}

// Mid-combo, nudge the call towards the punch that usually comes next in the user's taught combos.
function withComboPrior(c, t) {
  const prefix = t - S.lastPunchT < S.comboGap ? S.comboSeq : [];
  const prior = prefix.length ? comboPrior(personal, prefix) : null;
  if (!prior) return c;
  const post = LABELS.map((l) => [l, (c.probs[l] + 0.05) * (CFG.PRIOR_WEIGHT + (prior[l] || 0))]);
  const sum = post.reduce((a, [, v]) => a + v, 0);
  post.sort((a, b) => b[1] - a[1]);
  return { label: post[0][0], confidence: post[0][1] / sum };
}

// When the combo window closes, name the combo if it is one we know.
function checkComboEnd(t) {
  if (S.comboDone || t - S.lastPunchT < S.comboGap) return;
  S.comboDone = true;
  const n = S.comboSeq.length;
  if (n < 2) return;
  const m = matchCombo(S.comboSeq);
  if (m) {
    S.comboHits[m.id]++;
    showCallout('Combo', m.name);
    speakNow(m.name); S.lastVoiceT = t;
  } else if (n >= 3) { speakNow(`${n} punch combo`); S.lastVoiceT = t; }
}

// ---------------------------------------------------------------- training mode
function onPunch(type, confidence, t) {
  S.punches++;
  const inCombo = t - S.lastPunchT < S.comboGap;
  if (!inCombo) S.comboSeq = [];
  S.comboSeq.push(type); S.comboDone = false;
  S.combo = inCombo ? S.combo + 1 : 1;
  S.bestCombo = Math.max(S.bestCombo, S.combo);
  S.lastPunchT = t;
  S.byType[type]++;
  S.history.push(type); if (S.history.length > 5) S.history.shift();
  highlightUntil = t + 0.4;
  const quality = confidence > 0.9 ? 'Excellent' : confidence > 0.75 ? 'Good' : 'Nice';
  showCallout(quality, lower(type));
  renderCounts();
  if (S.combo >= 2) { /* the whole combo is named when it ends */ }
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
  el.recent.replaceChildren(...S.history.slice(-3).map((p) => { const li = document.createElement('li'); li.textContent = lower(p); return li; }));
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
    combos: S.comboHits,
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
    const n = document.createElement('span'); n.className = 'bname'; n.textContent = NAMES[l];
    const b = document.createElement('div'); b.className = 'bar';
    const i = document.createElement('i'); i.style.width = `${(s.byType[l] / max) * 100}%`; b.append(i);
    const c = document.createElement('span'); c.className = 'bn'; c.textContent = s.byType[l];
    row.append(n, b, c); bars.append(row);
  }
  frag.append(bars);
  const hits = COMBOS.filter((cb) => s.combos?.[cb.id]);
  if (hits.length) {
    const t = document.createElement('p'); t.className = 'card-title'; t.textContent = 'Combos';
    const list = document.createElement('div'); list.className = 'bars';
    for (const cb of hits) {
      const row = document.createElement('div'); row.className = 'brow';
      const n = document.createElement('span'); n.className = 'bname'; n.style.textTransform = 'none'; n.textContent = cb.name;
      const c = document.createElement('span'); c.className = 'bn'; c.textContent = `×${s.combos[cb.id]}`;
      row.style.gridTemplateColumns = '1fr auto';
      row.append(n, c); list.append(row);
    }
    frag.append(t, list);
  }
  const note = document.createElement('p'); note.className = 'sheet-note';
  note.textContent = s.model === 'personal' ? 'Called by your model.' : 'Called by the starter model. Teach it your punches for better accuracy.';
  frag.append(note);
  return frag;
}

function renderLast() {
  const s = readJSON('bc.last');
  if (!s) { el.lastCard.hidden = true; return; }
  el.lastCard.hidden = false;
  el.lastStats.replaceChildren(...[[s.punches, 'Punches'], [fmt(s.secs), 'Time'], [s.best, 'Best combo'], [s.ppm ?? '–', 'Per min']].map(statEl));
}

// ---------------------------------------------------------------- teach mode
// Plan steps: { type: 'stance' } | { type: 'punch', label, need } | { type: 'combo', combo, need }
async function startTeach(plan, redo) {
  unlockAudio();
  try { await ready; facing = settings.camera; await openCamera(); }
  catch (e) { console.error(e); location.hash = '#analyse'; setStatus(cameraError(e), true); return; }
  if (redo?.type === 'punch') personal.examples = personal.examples.filter((e) => !(e.l === redo.id && !e.c));
  if (redo?.type === 'combo') { personal.examples = personal.examples.filter((e) => e.c !== redo.id); delete personal.combos[redo.id]; }
  mode = 'teach';
  voiceOn = settings.voice === 'on';
  engine.reset();
  engine.useStarter = false;
  engine.motionSpeed = 0.1;
  T = newTeach(plan, !!redo);
  setVoiceButton();
  showStage('teach');
  nextTeachLabel(performance.now() / 1000);
}

function newTeach(plan, changed) {
  return { plan, i: -1, step: null, got: 0, need: 0, readyUntil: 0, nextAt: 0, added: [], label: null, wentGo: false, changed, buf: [], guard: [] };
}

// Read the stance first whenever it hasn't been read yet (a manual setting is kept).
function withStance(plan) { return personal.stance ? plan : [{ type: 'stance' }, ...plan]; }

function nothingMissing() {
  const c = counts(personal, true), reps = comboReps(personal);
  return !!personal.stance && LABELS.every((l) => c[l] >= TARGET) && COMBOS.every((cb) => reps[cb.id] >= COMBO_TARGET);
}

function teachPlan() {
  const c = counts(personal, true), reps = comboReps(personal);
  const more = nothingMissing();
  const plan = [{ type: 'stance' }];
  for (const l of LABELS) plan.push({ type: 'punch', label: l, need: more ? 5 : Math.max(0, TARGET - c[l]) });
  if (!more) for (const cb of COMBOS) plan.push({ type: 'combo', combo: cb, need: Math.max(0, COMBO_TARGET - reps[cb.id]) });
  // Keep an existing manual stance; otherwise re-read it (the user may have moved the phone).
  return plan.filter((p) => (p.type === 'stance' ? personal.stance?.source !== 'manual' : p.need > 0));
}

function nextTeachLabel(t) {
  T.i++;
  if (T.i >= T.plan.length) { finishTeach(); return; }
  const st = T.plan[T.i];
  Object.assign(T, {
    step: st, label: st.label || null, need: st.type === 'stance' ? 1 : st.need, got: 0,
    readyUntil: t + CFG.TEACH_READY_S, nextAt: 0, wentGo: false, goT: 0, lastCapT: -Infinity, buf: [], guard: [],
  });
  el.teachStep.textContent = `Step ${T.i + 1} of ${T.plan.length}`;
  el.teachLabel.classList.toggle('long', st.type !== 'punch' || NAMES[st.label].length > 8);
  if (st.type === 'stance') {
    el.teachLabel.textContent = 'Your stance';
    el.teachMsg.textContent = 'Face the phone in your fighting stance';
    speakNow('Get in your stance and hold your guard.');
  } else if (st.type === 'combo') {
    el.teachLabel.textContent = st.combo.name;
    el.teachMsg.textContent = `Get ready to throw this combo ${st.need} times`;
    speakNow(`${st.combo.name}. ${st.need} times. Get ready.`);
  } else {
    el.teachLabel.textContent = NAMES[st.label];
    el.teachMsg.textContent = `Get ready to throw ${st.need} ${PLURAL[st.label]}${handHint(st.label)}`;
    speakNow(`${st.need} ${PLURAL[st.label]}. Get ready.`);
  }
  updateRing();
}

// " with your left hand" etc., once the stance is known.
function handHint(label) {
  const lead = personal.stance?.lead;
  if (!lead || !ROLE[label]) return '';
  return ` with your ${ROLE[label] === 'lead' ? lead : lead === 'left' ? 'right' : 'left'} hand`;
}

// Short pause after a step finishes so the result can be read, then move on.
function stepDone(t, msg) {
  el.teachMsg.textContent = msg;
  T.nextAt = t + CFG.TEACH_NEXT_PAUSE;
}

function teachStep(r, t) {
  if (T.nextAt) { if (t >= T.nextAt) nextTeachLabel(t); return; }
  // countdown
  if (t < T.readyUntil) {
    el.teachCenter.className = 'teach-center';
    el.teachCenter.textContent = String(Math.ceil(T.readyUntil - t));
    return;
  }
  const st = T.step;
  if (!T.wentGo) {
    T.wentGo = true; T.goT = t;
    flashCenter(st.type === 'stance' ? 'Hold' : 'Go');
    bell(1);
    el.teachMsg.textContent = st.type === 'stance' ? 'Hold your guard still…'
      : st.type === 'combo' ? `Throw ${st.combo.seq.map(lower).join(', ')}, then back to guard`
        : `Throw a ${lower(T.label)}${handHint(T.label)}, then back to guard`;
  }
  if (st.type === 'stance') return teachStance(r, t);
  if (st.type === 'combo') return teachCombo(r, t);

  const lead = personal.stance?.lead;
  for (const ev of r.events) {
    if (ev.kind !== 'motion' || ev.t - T.lastCapT < CFG.TEACH_GAP_S) continue;
    const ex = { l: T.label, f: ev.features, s: ev.speed, h: ev.hand || null, c: null };
    personal.examples.push(ex);
    T.added.push({ exs: [ex] }); T.changed = true;
    T.got++; T.lastCapT = ev.t;
    highlightUntil = t + 0.35;
    tick();
    flashCenter(`${T.got}`);
    updateRing();
    // Every punch belongs to one hand: point it out if the other hand seemed to throw it.
    const role = roleOf(ev.hand, lead);
    if (role && ROLE[T.label] && role !== ROLE[T.label]) {
      el.teachMsg.textContent = `That looked like your ${role} hand. Undo it if it was wrong.`;
    }
    if (T.got >= T.need) { stepDone(t, 'Nice. Next…'); return; }
  }
}

function teachStance(r, t) {
  if (r.guard != null) T.guard.push(r.guard);
  if (T.guard.length >= CFG.TEACH_STANCE_FRAMES) {
    const g = [...T.guard].sort((a, b) => a - b);
    const lead = g[g.length >> 1] < 0 ? 'left' : 'right';
    personal.stance = { lead, source: 'auto' };
    T.changed = true; T.got = 1; updateRing();
    tick(); flashCenter(stanceName(lead));
    speakNow(`${stanceName(lead)}. ${cap(lead)} hand leads.`);
    stepDone(t, `${stanceName(lead)}: your ${lead} hand leads. You can change it on the Teach page.`);
  } else if (t - T.goT > CFG.TEACH_STANCE_TIMEOUT) {
    stepDone(t, personal.stance ? 'Could not read your guard. Keeping your saved stance.' : 'Could not read your guard. Set it on the Teach page.');
  }
}

function teachCombo(r, t) {
  const cb = T.step.combo;
  for (const ev of r.events) {
    if (ev.kind !== 'motion' || ev.t < T.goT) continue;
    T.buf.push(ev);
    highlightUntil = t + 0.25;
  }
  if (!T.buf.length || t - T.buf[T.buf.length - 1].t < CFG.TEACH_COMBO_END_S) return;
  const buf = T.buf; T.buf = [];
  if (buf.length !== cb.seq.length) {
    el.teachMsg.textContent = buf.length < cb.seq.length
      ? `Only caught ${buf.length} of ${cb.seq.length} punches. Snap each one out and go again.`
      : `Caught ${buf.length} punches, expected ${cb.seq.length}. Go again.`;
    return;
  }
  const exs = buf.map((ev, i) => ({ l: cb.seq[i], f: ev.features, s: ev.speed, h: ev.hand || null, c: cb.id }));
  const gaps = buf.slice(1).map((ev, i) => ev.t - buf[i].t);
  personal.examples.push(...exs);
  const rec = personal.combos[cb.id] || (personal.combos[cb.id] = { reps: 0, gaps: [] });
  rec.reps++; rec.gaps.push(...gaps);
  T.added.push({ exs, combo: cb.id, gaps: gaps.length }); T.changed = true;
  T.got++;
  tick(); flashCenter(`${T.got}`); updateRing();
  if (T.got >= T.need) stepDone(t, 'Nice. Next…');
  else el.teachMsg.textContent = `Got it. Again: ${cb.seq.map(lower).join(', ')}`;
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
  if (!T || !T.got || T.step?.type === 'stance' || !T.added.length) return;
  const g = T.added.pop();
  for (const ex of g.exs) { const i = personal.examples.lastIndexOf(ex); if (i >= 0) personal.examples.splice(i, 1); }
  if (g.combo) {
    const rec = personal.combos[g.combo];
    if (rec) { rec.reps--; rec.gaps.splice(rec.gaps.length - g.gaps, g.gaps); if (rec.reps <= 0) delete personal.combos[g.combo]; }
  }
  if (T.nextAt) T.nextAt = 0;  // stay on this step
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
    const n = document.createElement('span'); n.className = 'bname'; n.textContent = NAMES[l];
    const b = document.createElement('div'); b.className = 'bar' + (acc != null && acc >= 0.85 ? ' full' : '');
    const i = document.createElement('i'); i.style.width = `${acc != null ? acc * 100 : 0}%`; b.append(i);
    const v = document.createElement('span'); v.className = 'bn'; v.textContent = acc != null ? pct(acc) : `${c[l]}`;
    row.append(n, b, v); bars.append(row);
  }
  frag.append(bars);
  const reps = comboReps(personal);
  const learned = COMBOS.filter((cb) => reps[cb.id] > 0).length;
  const extra = document.createElement('p'); extra.className = 'sheet-note';
  extra.textContent = `Stance: ${personal.stance ? `${stanceName(personal.stance.lead)} (${personal.stance.lead} lead)` : 'not set'} · Combos learned: ${learned} of ${COMBOS.length}`;
  frag.append(extra);
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
      T = newTeach(plan || teachPlan(), false);
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
    return mode === 'teach' ? { step: T.step?.type, label: T.label, got: T.got, added: T.added.length - before } : { punches: S.punches, last: S.history.at(-1), new: S.punches - before };
  },
  finish() { running = true; return mode === 'teach' ? finishTeach() : endTrain(); },
  get personal() { return personal; },
  get engine() { return engine; },
  rebuild() { rebuildPersonal(); refreshModelUI(); },
  get evalResult() { return evalResult; },
};
