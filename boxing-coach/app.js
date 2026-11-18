// Boxing Coach: on-device punch recognition.
// Pose: MediaPipe Pose Landmarker (runs in the browser via WebAssembly/WebGL).
// Punch classifier: the original project's trained LSTM, ported to JavaScript.
// Nothing here sends camera frames anywhere.

import { FilesetResolver, PoseLandmarker } from './vendor/vision_bundle.mjs';
import { LSTMClassifier } from './lstm.js';

// Settings carried over from the original shadow_boxing_analyzer.py.
// Frame counts there assumed 30 fps; here they are converted to seconds.
const CFG = {
  SEQ: 25,              // frames per LSTM window
  HZ: 30,               // the window is resampled to 30 fps to match training
  CONFIDENCE: 0.7,      // min softmax confidence for a punch
  KP_CONF: 0.5,         // min keypoint confidence
  RECOVERY_S: 2.0,      // 60 frames: ignore new punches while recovering
  MIN_SPEED: 2.0,       // wrist speed gate (pixels/frame at 640 px)
  COMBO_TIMEOUT: 3.0,
  ENCOURAGE_EVERY: 5,
  VOICE_COOLDOWN: 2.0,
  REF_SIZE: 640,        // original webcam frame size
  REST_S: 60,
};
const PUNCHES = ['jab', 'cross', 'hook', 'uppercut'];
// MediaPipe landmark indices in the original model's keypoint order:
// L/R shoulder, L/R elbow, L/R wrist, L/R hip.
const MP_IDX = [11, 12, 13, 14, 15, 16, 23, 24];
const SKELETON = [[0, 1], [0, 2], [2, 4], [1, 3], [3, 5], [0, 6], [1, 7], [6, 7]];

const CALLS = {
  jab: ['Clean jab', 'Sharp jab', 'Good jab', 'Nice jab'],
  cross: ['Powerful cross', 'Good cross', 'Nice cross', 'Strong cross'],
  hook: ['Solid hook', 'Good hook', 'Nice hook'],
  uppercut: ['Solid uppercut', 'Good uppercut', 'Nice uppercut', 'Strong uppercut'],
};
const ENCOURAGE = ['Keep it up', 'Stay sharp', 'Hands up, keep going', 'Nice rhythm', 'Good work, keep moving'];

const $ = (id) => document.getElementById(id);
const el = {
  start: $('start'), summary: $('summary'), stage: $('stage'),
  cam: $('cam'), overlay: $('overlay'),
  btnStart: $('btn-start'), btnAgain: $('btn-again'), btnEnd: $('btn-end'),
  btnFlip: $('btn-flip'), btnVoice: $('btn-voice'), file: $('file-input'),
  optCamera: $('opt-camera'), optRound: $('opt-round'), optVoice: $('opt-voice'),
  status: $('status'), clock: $('clock'), roundLabel: $('round-label'),
  punchCount: $('punch-count'), comboCount: $('combo-count'),
  stance: $('stance'), callout: $('callout'), hint: $('hint'), recent: $('recent'),
};
const ctx = el.overlay.getContext('2d');

// ---------------------------------------------------------------- models
let pose = null, punchModel = null;
const ready = (async () => {
  const [vision, punch] = await Promise.all([
    FilesetResolver.forVisionTasks('./vendor/wasm'),
    LSTMClassifier.load('./models/punch_lstm.bin'),
  ]);
  punchModel = punch;
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: './models/pose_landmarker_full.task', delegate },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { pose = await PoseLandmarker.createFromOptions(vision, opts('GPU')); }
  catch (e) { console.warn('GPU unavailable, using CPU', e); pose = await PoseLandmarker.createFromOptions(vision, opts('CPU')); }
})();

ready.then(() => {
  el.btnStart.disabled = false;
  el.btnStart.textContent = 'Start training';
}).catch((e) => {
  console.error(e);
  setStatus('The coach could not load. Check your connection and reload the page.', true);
  el.btnStart.textContent = 'Could not load';
});

// ---------------------------------------------------------------- session state
let S = null;
function newSession(mode) {
  return {
    mode,                         // 'live' | 'file'
    buffer: [],                   // valid frames: { t, pts: Float32Array(16) px, conf: Float32Array(8) }
    recoverUntil: -Infinity,
    punches: 0, combo: 0, bestCombo: 0, lastPunchT: -Infinity,
    byType: { jab: 0, cross: 0, hook: 0, uppercut: 0 },
    history: [],
    startT: null, nowT: 0,
    lastVoiceT: -Infinity,
    roundLen: Number(el.optRound.value) || 0,
    lastSeenT: -Infinity,
    phase: 'work', phaseKey: '',
  };
}

let running = false, rafId = 0, lastVideoTime = -1, stream = null, wakeLock = null;
let facing = 'user';
let voiceOn = true;

// ---------------------------------------------------------------- pose → features
// Pixel coords scaled so the longer side is 640 px, like the original webcam frames.
function toFrame(landmarks, w, h, t) {
  const s = CFG.REF_SIZE / Math.max(w, h);
  const pts = new Float32Array(16), conf = new Float32Array(8);
  MP_IDX.forEach((mi, i) => {
    const p = landmarks[mi];
    pts[i * 2] = p.x * w * s;
    pts[i * 2 + 1] = p.y * h * s;
    conf[i] = p.visibility ?? 1;
  });
  return { t, pts, conf };
}

// Same as normalize_keypoints(): centre on mid-shoulders, divide by shoulder width.
function normalize(f) {
  if (f.conf[0] < CFG.KP_CONF || f.conf[1] < CFG.KP_CONF) return null;
  const lx = f.pts[0], ly = f.pts[1], rx = f.pts[2], ry = f.pts[3];
  const d = Math.hypot(lx - rx, ly - ry);
  if (d < 1e-4) return null;
  const cx = (lx + rx) / 2, cy = (ly + ry) / 2;
  const out = new Float32Array(16);
  for (let i = 0; i < 8; i++) {
    if (f.conf[i] > CFG.KP_CONF) {
      out[i * 2] = (f.pts[i * 2] - cx) / d;
      out[i * 2 + 1] = (f.pts[i * 2 + 1] - cy) / d;
    }
  }
  return out;
}

// Resample the recent frames onto an even 30 fps grid so the window covers the
// same time span the model was trained on, whatever the phone's frame rate.
function resampledWindow() {
  const b = S.buffer;
  if (b.length < 4) return null;
  const tEnd = b[b.length - 1].t;
  const tStart = tEnd - (CFG.SEQ - 1) / CFG.HZ;
  if (b[0].t > tStart + 0.02) return null;
  const out = [];
  let j = 0;
  for (let k = 0; k < CFG.SEQ; k++) {
    const t = tStart + k / CFG.HZ;
    while (j < b.length - 2 && b[j + 1].t < t) j++;
    const a = b[j], c = b[j + 1];
    if (c.t - a.t > 0.3) return null;          // lost track mid-window
    const u = Math.min(1, Math.max(0, (t - a.t) / Math.max(1e-6, c.t - a.t)));
    const pts = new Float32Array(16), conf = new Float32Array(8);
    for (let i = 0; i < 16; i++) pts[i] = a.pts[i] + (c.pts[i] - a.pts[i]) * u;
    for (let i = 0; i < 8; i++) conf[i] = Math.min(a.conf[i], c.conf[i]);
    out.push({ t, pts, conf });
  }
  return out;
}

// Same as calculate_punch_speed(): faster wrist over the last 5 frames.
function punchSpeed(frames) {
  const recent = frames.slice(-5);
  let best = 0;
  for (const wi of [4, 5]) {
    const pos = recent.filter((f) => f.conf[wi] > CFG.KP_CONF).map((f) => [f.pts[wi * 2], f.pts[wi * 2 + 1]]);
    if (pos.length < 2) continue;
    let dist = 0;
    for (let i = 0; i < pos.length - 1; i++) dist += Math.hypot(pos[i + 1][0] - pos[i][0], pos[i + 1][1] - pos[i][1]);
    best = Math.max(best, dist / pos.length);
  }
  return best;
}

// Same as detect_stance().
function stance(f) {
  if ([0, 1, 6, 7].some((i) => f.conf[i] < CFG.KP_CONF)) return null;
  const left = Math.hypot(f.pts[0] - f.pts[12], f.pts[1] - f.pts[13]);
  const right = Math.hypot(f.pts[2] - f.pts[14], f.pts[3] - f.pts[15]);
  return left < right ? 'Southpaw' : 'Orthodox';
}

// ---------------------------------------------------------------- per-frame processing
let lastLandmarks = null;
function process(source, w, h, t) {
  const res = pose.detectForVideo(source, performance.now());
  const lm = res.landmarks && res.landmarks[0];
  lastLandmarks = lm ? { lm, w, h } : null;
  if (S.startT === null) S.startT = t;
  S.nowT = t;
  updatePhase();

  if (lm) {
    const f = toFrame(lm, w, h, t);
    if (normalize(f)) {
      S.buffer.push(f);
      S.lastSeenT = t;
    }
    const st = stance(f);
    el.stance.textContent = st ? `Stance: ${st}` : 'Stance: step back so your hips show';
  }
  while (S.buffer.length && S.buffer[0].t < t - 2) S.buffer.shift();
  el.hint.hidden = t - S.lastSeenT < 1.0;

  if (S.phase === 'work' && t >= S.recoverUntil) analyse(t);
}

function analyse(t) {
  const win = resampledWindow();
  if (!win) return;
  const seq = win.map(normalize);
  if (seq.some((x) => !x)) return;
  const { index, confidence } = punchModel.predict(seq);
  if (confidence > CFG.CONFIDENCE && punchSpeed(win) > CFG.MIN_SPEED) {
    onPunch(PUNCHES[index], confidence, t);
  }
}

function onPunch(type, confidence, t) {
  S.punches++;
  S.combo = t - S.lastPunchT < CFG.COMBO_TIMEOUT ? S.combo + 1 : 1;
  S.bestCombo = Math.max(S.bestCombo, S.combo);
  S.lastPunchT = t;
  S.byType[type]++;
  S.history.push(type);
  if (S.history.length > 5) S.history.shift();
  S.recoverUntil = t + CFG.RECOVERY_S;

  const quality = confidence > 0.9 ? 'Excellent' : confidence > 0.8 ? 'Good' : 'Nice';
  showCallout(quality, type);
  renderCounts();

  if (S.combo >= 3) speak(`${S.combo} punch combo`, t);
  else if (S.punches % CFG.ENCOURAGE_EVERY === 0) speak(pick(ENCOURAGE), t);
  else speak(pick(CALLS[type]), t);
}

// ---------------------------------------------------------------- rounds
function updatePhase() {
  const elapsed = S.nowT - S.startT;
  let label = 'Session', shown = elapsed, phase = 'work';
  if (S.roundLen && S.mode === 'live') {
    const cycle = S.roundLen + CFG.REST_S;
    const n = Math.floor(elapsed / cycle);
    const inCycle = elapsed - n * cycle;
    if (inCycle < S.roundLen) { phase = 'work'; label = `Round ${n + 1}`; shown = S.roundLen - inCycle; }
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

// ---------------------------------------------------------------- UI
function renderCounts() {
  el.punchCount.textContent = S.punches;
  el.comboCount.textContent = S.combo;
  el.recent.replaceChildren(...S.history.slice(-3).map((p) => {
    const li = document.createElement('li'); li.textContent = p; return li;
  }));
}

function showCallout(quality, type) {
  el.callout.classList.remove('show');
  el.callout.replaceChildren();
  const q = document.createElement('span'); q.className = 'q'; q.textContent = quality;
  el.callout.append(q, document.createTextNode(`${type}!`));
  void el.callout.offsetWidth; // restart animation
  el.callout.classList.add('show');
}

function draw() {
  const cw = el.overlay.clientWidth, ch = el.overlay.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (el.overlay.width !== Math.round(cw * dpr) || el.overlay.height !== Math.round(ch * dpr)) {
    el.overlay.width = Math.round(cw * dpr); el.overlay.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!lastLandmarks) return;
  const { lm, w, h } = lastLandmarks;
  // Map video pixels to the screen the same way object-fit: cover does.
  const sc = Math.max(cw / w, ch / h);
  const ox = (cw - w * sc) / 2, oy = (ch - h * sc) / 2;
  const P = MP_IDX.map((mi) => ({ x: ox + lm[mi].x * w * sc, y: oy + lm[mi].y * h * sc, v: lm[mi].visibility ?? 1 }));
  const recovering = S && S.nowT < S.recoverUntil;
  const col = recovering ? '#f2b33d' : '#f3efe6';
  ctx.lineCap = 'round';
  ctx.strokeStyle = col; ctx.lineWidth = 4; ctx.globalAlpha = 0.85;
  for (const [a, b] of SKELETON) {
    if (P[a].v < CFG.KP_CONF || P[b].v < CFG.KP_CONF) continue;
    ctx.beginPath(); ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[b].x, P[b].y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  P.forEach((p, i) => {
    if (p.v < CFG.KP_CONF || i > 5) return;
    const wrist = i === 4 || i === 5;
    ctx.fillStyle = wrist ? '#e5484d' : col;
    ctx.beginPath(); ctx.arc(p.x, p.y, wrist ? 11 : 6, 0, Math.PI * 2); ctx.fill();
  });
}

function show(which) {
  el.start.hidden = which !== 'start';
  el.summary.hidden = which !== 'summary';
  el.stage.hidden = which !== 'stage';
}

function setStatus(msg, isError = false) {
  el.status.textContent = msg;
  el.status.classList.toggle('error', isError);
}

function resetHud() {
  renderCounts();
  el.callout.classList.remove('show');
  el.stance.textContent = 'Stance: finding you…';
  el.hint.hidden = false;
  el.clock.textContent = '00:00';
  el.btnVoice.textContent = voiceOn ? 'Voice on' : 'Voice off';
  el.btnVoice.setAttribute('aria-pressed', String(voiceOn));
}

// ---------------------------------------------------------------- voice + bell
function speak(text, t) {
  if (!voiceOn || !('speechSynthesis' in window)) return;
  if (t - S.lastVoiceT < CFG.VOICE_COOLDOWN) return;
  S.lastVoiceT = t;
  speakNow(text);
}
function speakNow(text) {
  if (!voiceOn || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.1; u.lang = 'en-GB';
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
function bell(times) {
  if (!audioCtx) return;
  for (let i = 0; i < times; i++) {
    const t0 = audioCtx.currentTime + i * 0.35;
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.type = 'triangle'; o.frequency.value = 1320;
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(0.5, t0 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.9);
    o.connect(g).connect(audioCtx.destination);
    o.start(t0); o.stop(t0 + 1);
  }
}

// ---------------------------------------------------------------- camera / file
async function openCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw Object.assign(new Error('insecure'), { name: 'InsecureContext' });
  }
  stopStream();
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
  });
  el.cam.srcObject = stream;
  el.cam.classList.toggle('mirror', facing === 'user');
  el.overlay.classList.toggle('mirror', facing === 'user');
  await el.cam.play();
}

function stopStream() {
  if (stream) stream.getTracks().forEach((tr) => tr.stop());
  stream = null;
}

function cameraError(e) {
  const name = e && e.name;
  if (name === 'NotAllowedError') return 'Camera access is blocked. On iPhone open Settings › Safari › Camera, choose Allow, then reload this page.';
  if (name === 'InsecureContext') return 'The camera only works when the app is opened from its https:// link.';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found on this device.';
  return 'The camera could not start. Close other apps using it and try again.';
}

async function startLive() {
  unlockAudio();
  setStatus('');
  el.btnStart.disabled = true; el.btnStart.textContent = 'Starting camera…';
  try {
    await ready;
    facing = el.optCamera.value;
    voiceOn = el.optVoice.value === 'on';
    await openCamera();
  } catch (e) {
    console.error(e);
    setStatus(cameraError(e), true);
    el.btnStart.disabled = false; el.btnStart.textContent = 'Start training';
    return;
  }
  el.btnStart.disabled = false; el.btnStart.textContent = 'Start training';
  el.btnFlip.hidden = false;
  S = newSession('live');
  if (S.roundLen) bell(3);
  begin();
}

async function startFile(file) {
  unlockAudio();
  setStatus('');
  try { await ready; } catch { return; }
  stopStream();
  voiceOn = el.optVoice.value === 'on';
  el.cam.srcObject = null;
  el.cam.src = URL.createObjectURL(file);
  el.cam.classList.remove('mirror'); el.overlay.classList.remove('mirror');
  el.btnFlip.hidden = true;
  S = newSession('file');
  try { await el.cam.play(); }
  catch (e) { console.error(e); setStatus('That video could not be played on this phone.', true); return; }
  begin();
}

function begin() {
  resetHud();
  show('stage');
  running = true; lastVideoTime = -1;
  requestWakeLock();
  rafId = requestAnimationFrame(loop);
}

function loop() {
  if (!running) return;
  const v = el.cam;
  if (v.readyState >= 2 && v.videoWidth && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    const t = S.mode === 'file' ? v.currentTime : performance.now() / 1000;
    try { process(v, v.videoWidth, v.videoHeight, t); } catch (e) { console.error(e); }
  }
  draw();
  if (S.mode === 'file' && v.ended) { end(); return; }
  rafId = requestAnimationFrame(loop);
}

function end() {
  if (!running) return;
  running = false;
  cancelAnimationFrame(rafId);
  stopStream();
  el.cam.pause();
  if (el.cam.src) { URL.revokeObjectURL(el.cam.src); el.cam.removeAttribute('src'); el.cam.load(); }
  releaseWakeLock();
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  renderSummary();
  show('summary');
}

function renderSummary() {
  const secs = Math.max(0, S.nowT - (S.startT ?? S.nowT));
  $('sum-total').textContent = S.punches;
  $('sum-time').textContent = fmt(secs);
  $('sum-best').textContent = S.bestCombo;
  $('sum-ppm').textContent = secs >= 10 ? Math.round(S.punches / (secs / 60)) : '–';
  const max = Math.max(1, ...Object.values(S.byType));
  $('sum-bars').replaceChildren(...PUNCHES.map((p) => {
    const row = document.createElement('div'); row.className = 'bar';
    const name = document.createElement('span'); name.className = 'name'; name.textContent = p;
    const track = document.createElement('div'); track.className = 'track';
    const fill = document.createElement('div'); fill.className = 'fill';
    fill.style.width = `${(S.byType[p] / max) * 100}%`;
    track.append(fill);
    const n = document.createElement('span'); n.className = 'n'; n.textContent = S.byType[p];
    row.append(name, track, n);
    return row;
  }));
}

// ---------------------------------------------------------------- wake lock
async function requestWakeLock() {
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { wakeLock = null; }
}
function releaseWakeLock() { try { wakeLock?.release(); } catch { /* ignore */ } wakeLock = null; }
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && running) requestWakeLock();
});

// ---------------------------------------------------------------- helpers
function fmt(s) {
  s = Math.max(0, Math.floor(s));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
function pick(a) { return a[Math.floor(Math.random() * a.length)]; }

// ---------------------------------------------------------------- events
el.btnStart.addEventListener('click', startLive);
el.file.addEventListener('change', () => { const f = el.file.files[0]; el.file.value = ''; if (f) startFile(f); });
el.btnEnd.addEventListener('click', end);
el.btnAgain.addEventListener('click', () => show('start'));
el.btnVoice.addEventListener('click', () => {
  voiceOn = !voiceOn;
  if (!voiceOn && 'speechSynthesis' in window) speechSynthesis.cancel();
  el.btnVoice.textContent = voiceOn ? 'Voice on' : 'Voice off';
  el.btnVoice.setAttribute('aria-pressed', String(voiceOn));
});
el.btnFlip.addEventListener('click', async () => {
  facing = facing === 'user' ? 'environment' : 'user';
  S.buffer = [];
  try { await openCamera(); } catch (e) { console.error(e); facing = facing === 'user' ? 'environment' : 'user'; }
});

// Offline support once installed from an https link.
if ('serviceWorker' in navigator && location.protocol === 'https:') {
  navigator.serviceWorker.register('./sw.js').catch(() => {});
}

// Test hook: lets a developer feed frames from any image/canvas source.
window.boxingCoach = {
  ready,
  startTest() { S = newSession('file'); resetHud(); show('stage'); running = true; return true; },
  feed(source, w, h, t) { process(source, w, h, t); draw(); return { punches: S.punches, history: [...S.history], byType: { ...S.byType } }; },
};
