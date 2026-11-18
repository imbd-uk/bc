// Voice for the coach and the film study.
// - Natural (default): the free, open-source Kokoro neural voice, running on this device in a
//   worker. No account, no key, works offline after a one-time ~90 MB download. Every line is
//   cached after it is first spoken, so repeats are instant.
// - Phone voice: the device's own built-in voice, as the fallback.

const KEY = 'bc.voice';
const CACHE = 'bc-voice-kokoro-v1';

export const NATURAL_VOICES = [
  { id: 'bm_george', name: 'George · British, male' },
  { id: 'bm_fable', name: 'Fable · British, male' },
  { id: 'bm_lewis', name: 'Lewis · British, male' },
  { id: 'am_michael', name: 'Michael · American, male' },
  { id: 'am_fenrir', name: 'Fenrir · American, male' },
  { id: 'am_puck', name: 'Puck · American, male' },
  { id: 'af_heart', name: 'Heart · American, female' },
  { id: 'af_bella', name: 'Bella · American, female' },
  { id: 'bf_emma', name: 'Emma · British, female' },
];

// Chatterbox (Resemble AI, open source) running on the user's own PC: see chatterbox-server.
export const CHATTER_URL = 'http://127.0.0.1:8001';

function load() {
  const d = {
    provider: 'natural', voiceId: 'bm_george', deviceVoice: '',
    cbVoice: 'default', cbModel: 'turbo', cbEnergy: 0.6,
  };
  try {
    const s = JSON.parse(localStorage.getItem(KEY) || '{}');
    if (s.provider === 'elevenlabs') delete s.provider;   // older setting
    if (s.voiceId && !NATURAL_VOICES.some((v) => v.id === s.voiceId)) delete s.voiceId;
    return Object.assign(d, s);
  } catch { return d; }
}

class Voice {
  constructor() {
    this.cfg = load();
    this.ctx = null;
    this.playing = null;     // current BufferSource
    this.tap = null;         // optional extra output (recording)
    this.mem = new Map();    // decoded audio, this session
    this.lastError = '';
    this.gen = 0;            // bumps on stop() so stale calls don't play
    this.worker = null; this.jobs = new Map(); this.nextId = 1;
    this.ready = false;      // natural voice model loaded
    this.onProgress = null;  // (fraction) during the model download
    this.loadErr = '';
  }

  save() { try { localStorage.setItem(KEY, JSON.stringify(this.cfg)); } catch { /* storage unavailable */ } }
  set(patch) { Object.assign(this.cfg, patch); this.save(); }
  get natural() { return this.cfg.provider === 'natural'; }
  get chatter() { return this.cfg.provider === 'chatterbox'; }
  // an AI voice whose audio we generate (so it can be cached and recorded into videos)
  get ai() { return this.natural || this.chatter; }

  // ---------------------------------------------------------------- Chatterbox (local PC server)
  async chatterHealth() {
    const r = await fetch(`${CHATTER_URL}/health`, { signal: AbortSignal.timeout(2500) });
    if (!r.ok) throw new Error('bad response');
    return r.json();
  }
  async chatterVoices() {
    const r = await fetch(`${CHATTER_URL}/voices`, { signal: AbortSignal.timeout(2500) });
    return (await r.json()).voices;
  }
  async chatterSay(text) {
    let r;
    try {
      r = await fetch(`${CHATTER_URL}/tts`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice: this.cfg.cbVoice, model: this.cfg.cbModel, exaggeration: this.cfg.cbEnergy }),
      });
    } catch {
      throw new Error('The Chatterbox voice server isn\'t running on your PC. Start it with start-chatterbox.bat.');
    }
    if (!r.ok) {
      let d = ''; try { d = (await r.json()).detail; } catch { /* */ }
      throw new Error(d || `Chatterbox error ${r.status}`);
    }
    return r.arrayBuffer();
  }

  context() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    return this.ctx;
  }
  // Call from a tap: phones only allow sound after a user gesture.
  unlock() {
    try {
      this.context().resume();
      if ('speechSynthesis' in window) { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); }
    } catch { /* sound is optional */ }
  }

  stop() {
    this.gen++;
    try { this.playing?.stop(); } catch { /* already stopped */ }
    this.playing = null;
    if ('speechSynthesis' in window) speechSynthesis.cancel();
  }

  // ---------------------------------------------------------------- natural voice (Kokoro)
  ensureWorker() {
    if (this.worker) return;
    this.worker = new Worker(new URL('./voice-worker.js', import.meta.url), { type: 'module' });
    const bytes = new Map();
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        bytes.set(m.file, [m.loaded, m.total]);
        let a = 0, b = 0; for (const [x, y] of bytes.values()) { a += x; b += y; }
        this.onProgress?.(b ? a / b : 0);
        return;
      }
      const job = this.jobs.get(m.id);
      if (!job) return;
      this.jobs.delete(m.id);
      if (m.error) job.reject(new Error(m.error)); else job.resolve(m);
    };
    this.worker.onerror = (e) => {
      this.loadErr = 'The natural voice could not start on this device.';
      for (const j of this.jobs.values()) j.reject(new Error(this.loadErr));
      this.jobs.clear();
      console.error(e);
    };
  }
  ask(msg) {
    this.ensureWorker();
    const id = this.nextId++;
    return new Promise((resolve, reject) => { this.jobs.set(id, { resolve, reject }); this.worker.postMessage({ id, ...msg }); });
  }
  // Downloads (first time) and starts the voice model.
  async warm() {
    if (this.ready) return;
    await this.ask({ type: 'load' });
    this.ready = true;
  }

  cacheKey(text) {
    const who = this.chatter
      ? `cb/${this.cfg.cbModel}/${encodeURIComponent(this.cfg.cbVoice)}/${this.cfg.cbModel === 'standard' ? this.cfg.cbEnergy : 0}`
      : this.cfg.voiceId;
    return `https://voice.cache/${who}/${encodeURIComponent(text)}`;
  }

  async cached(text) {
    const k = this.cacheKey(text);
    if (this.mem.has(k)) return this.mem.get(k);
    try {
      const c = await caches.open(CACHE);
      const hit = await c.match(k);
      if (!hit) return null;
      const bytes = await hit.arrayBuffer();
      const rate = Number(hit.headers.get('x-rate'));
      const buf = rate ? this.toBuffer(new Float32Array(bytes), rate) : await this.context().decodeAudioData(bytes);
      this.mem.set(k, buf);
      return buf;
    } catch { return null; }
  }

  toBuffer(samples, rate) {
    const buf = this.context().createBuffer(1, samples.length, rate);
    buf.copyToChannel(samples, 0);
    return buf;
  }

  // Text -> AudioBuffer with the natural voice (cached).
  async buffer(text) {
    const hit = await this.cached(text);
    if (hit) return hit;
    if (this.chatter) {
      const key = this.cacheKey(text);
      const wav = await this.chatterSay(text);
      try { const c = await caches.open(CACHE); await c.put(key, new Response(wav.slice(0), { headers: { 'Content-Type': 'audio/wav' } })); } catch { /* cache optional */ }
      const buf = await this.context().decodeAudioData(wav);
      this.mem.set(key, buf);
      return buf;
    }
    const { audio, rate } = await this.ask({ type: 'say', text, voice: this.cfg.voiceId });
    this.ready = true;
    try {
      const c = await caches.open(CACHE);
      await c.put(this.cacheKey(text), new Response(audio.slice().buffer, { headers: { 'Content-Type': 'application/octet-stream', 'x-rate': String(rate) } }));
    } catch { /* cache optional */ }
    const buf = this.toBuffer(audio, rate);
    this.mem.set(this.cacheKey(text), buf);
    return buf;
  }

  // Plays a decoded buffer. Resolves when it finishes (or is stopped).
  playBuffer(buf, { gain = 1 } = {}) {
    const ctx = this.context();
    try { this.playing?.stop(); } catch { /* */ }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain(); g.gain.value = gain;
    src.connect(g); g.connect(ctx.destination);
    if (this.tap) g.connect(this.tap);
    this.playing = src;
    return new Promise((done) => {
      src.onended = () => { if (this.playing === src) this.playing = null; done(); };
      src.start();
    });
  }

  // ---------------------------------------------------------------- device voice
  deviceVoices() {
    if (!('speechSynthesis' in window)) return [];
    const score = (v) => (/en-GB/i.test(v.lang) ? 2 : /^en/i.test(v.lang) ? 1 : -20)
      + (/natural|neural|premium|enhanced|siri/i.test(v.name) ? 6 : 0) + (/google/i.test(v.name) ? 3 : 0);
    return speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang)).sort((a, b) => score(b) - score(a));
  }
  deviceSay(text, gen) {
    if (!('speechSynthesis' in window)) return Promise.resolve();
    return new Promise((done) => {
      if (gen !== this.gen) { done(); return; }
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      const vs = this.deviceVoices();
      const v = vs.find((x) => x.name === this.cfg.deviceVoice) || vs[0];
      if (v) { u.voice = v; u.lang = v.lang; } else u.lang = 'en-GB';
      u.rate = 1.02;
      let finished = false;
      const end = () => { if (!finished) { finished = true; done(); } };
      u.onend = u.onerror = end;
      speechSynthesis.speak(u);
      setTimeout(end, 1500 + text.length * 90);  // iOS sometimes never fires onend
    });
  }

  // ---------------------------------------------------------------- speaking
  // live: short coach calls. The natural voice plays when that line is already cached;
  // otherwise the phone voice speaks now (no lag) and the natural version is made for next time.
  async say(text, { live = false } = {}) {
    this.stop();
    const gen = this.gen;
    if (!this.ai) return this.deviceSay(text, gen);
    if (live) {
      const hit = await this.cached(text);
      if (gen !== this.gen) return;
      if (hit) return this.playBuffer(hit);
      this.buffer(text).catch((e) => { this.lastError = e.message; });
      return this.deviceSay(text, gen);
    }
    try {
      const buf = await this.buffer(text);
      if (gen !== this.gen) return;
      return this.playBuffer(buf);
    } catch (e) {
      this.lastError = e.message;
      return this.deviceSay(text, gen);
    }
  }

  // Makes and caches a list of lines ahead of time. onStep(done, total).
  async prepare(lines, onStep) {
    const list = [...new Set(lines)];
    let done = 0;
    for (const t of list) {           // one at a time: the worker runs them in order anyway
      await this.buffer(t);
      onStep?.(++done, list.length);
    }
  }
}

export const voice = new Voice();

// ---------------------------------------------------------------- settings panel
// phrases: the coach's fixed lines, so they can be made in one go.
export function buildVoicePanel({ phrases = [] } = {}) {
  const root = document.createElement('div');
  root.className = 'voice-panel';
  root.innerHTML = `
    <div class="seg" role="radiogroup" aria-label="Voice" data-k="prov">
      <button type="button" role="radio" data-value="chatterbox">Chatterbox (PC)</button>
      <button type="button" role="radio" data-value="natural">Natural (AI)</button>
      <button type="button" role="radio" data-value="device">Phone</button>
    </div>
    <div data-k="cb">
      <p class="sheet-note">Resemble AI's Chatterbox running on your PC's graphics card: studio-quality, free and unlimited,
      with voice cloning. Works while the voice server is running on that PC (start-chatterbox.bat).</p>
      <p class="sheet-note" data-k="cbstate">Checking the voice server…</p>
      <label class="field"><span>Voice</span><select data-k="cbvoice"></select></label>
      <label class="field"><span>Model</span><select data-k="cbmodel">
        <option value="turbo">Turbo · fast</option>
        <option value="standard">Standard · more expressive</option></select></label>
      <label class="field" data-k="cbenergyrow"><span>Energy <b data-k="cbenergyv"></b></span>
        <input type="range" min="0.25" max="1.2" step="0.05" data-k="cbenergy"></label>
      <div class="row-btns"><button type="button" class="mini" data-k="cbtest">Test voice</button>
        <button type="button" class="mini" data-k="cbprep">Prepare coach calls</button>
        <button type="button" class="mini" data-k="cbrefresh">Refresh voices</button></div>
      <p class="sheet-note small">Clone a voice: put a clean 10–20 second WAV of one person talking in
      <b>chatterbox-server\\voices</b> on your PC, then tap Refresh voices. Only use voices you have permission to use.</p>
    </div>
    <div data-k="nat">
      <p class="sheet-note">A free, open-source AI voice (Kokoro) that sounds like a real commentator. It runs on this
      phone: no account, no key, nothing sent anywhere. The first time it downloads about 90 MB (use Wi-Fi), then works offline.</p>
      <label class="field"><span>Voice</span><select data-k="voice"></select></label>
      <div class="bar" data-k="dlbar" hidden><i></i></div>
      <div class="row-btns"><button type="button" class="mini" data-k="test">Test voice</button>
        <button type="button" class="mini" data-k="prep">Prepare coach calls</button></div>
      <p class="sheet-note small">Prepare coach calls makes every live call ahead of time so the coach answers instantly
      in the natural voice. Until then, new calls use the phone voice once while the natural one is made.</p>
    </div>
    <div data-k="dev">
      <p class="sheet-note">Uses a voice built into this device. Instant and offline, but more robotic. On iPhone, download an
      Enhanced or Premium voice in Settings › Accessibility › Spoken Content › Voices for a better sound.</p>
      <label class="field"><span>Voice</span><select data-k="dvoice"></select></label>
      <div class="row-btns"><button type="button" class="mini" data-k="dtest">Test voice</button></div>
    </div>
    <p class="status" data-k="status" role="status"></p>`;
  const $ = (k) => root.querySelector(`[data-k="${k}"]`);
  const status = (msg, err = false) => { $('status').textContent = msg; $('status').classList.toggle('error', err); };

  const sync = () => {
    $('prov').querySelectorAll('button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === voice.cfg.provider)));
    $('cb').hidden = !voice.chatter;
    $('nat').hidden = !voice.natural;
    $('dev').hidden = voice.ai;
    $('cbmodel').value = voice.cfg.cbModel;
    $('cbenergy').value = voice.cfg.cbEnergy;
    $('cbenergyv').textContent = Number(voice.cfg.cbEnergy).toFixed(2);
    $('cbenergyrow').hidden = voice.cfg.cbModel !== 'standard';
    if (voice.chatter) checkServer();
  };
  const checkServer = async () => {
    $('cbstate').textContent = 'Checking the voice server…';
    try {
      const h = await voice.chatterHealth();
      const list = await voice.chatterVoices();
      $('cbvoice').replaceChildren(...list.map((v) => Object.assign(document.createElement('option'), { value: v, textContent: v === 'default' ? 'Default (Chatterbox)' : v })));
      if (!list.includes(voice.cfg.cbVoice)) voice.set({ cbVoice: 'default' });
      $('cbvoice').value = voice.cfg.cbVoice;
      $('cbstate').textContent = `● Connected · ${h.gpu || h.device}${h.loaded.length ? '' : ' · the first line loads the model (~20 s)'}`;
      $('cbstate').classList.remove('error');
    } catch {
      $('cbstate').textContent = 'Not connected. On your PC, double-click start-chatterbox.bat in the chatterbox-server folder, then tap Refresh voices.';
      $('cbstate').classList.add('error');
    }
  };
  $('cbvoice').addEventListener('change', () => voice.set({ cbVoice: $('cbvoice').value }));
  $('cbmodel').addEventListener('change', () => { voice.set({ cbModel: $('cbmodel').value }); sync(); });
  $('cbenergy').addEventListener('input', () => { voice.set({ cbEnergy: Number($('cbenergy').value) }); $('cbenergyv').textContent = Number($('cbenergy').value).toFixed(2); });
  $('cbrefresh').addEventListener('click', checkServer);
  $('voice').replaceChildren(...NATURAL_VOICES.map((v) => Object.assign(document.createElement('option'), { value: v.id, textContent: v.name })));
  $('voice').value = voice.cfg.voiceId;
  const fillDevice = () => {
    const vs = voice.deviceVoices();
    $('dvoice').replaceChildren(...vs.map((v) => Object.assign(document.createElement('option'), { value: v.name, textContent: `${v.name} (${v.lang})` })));
    $('dvoice').value = voice.cfg.deviceVoice || vs[0]?.name || '';
  };
  voice.onProgress = (f) => {
    $('dlbar').hidden = f >= 1;
    $('dlbar').firstChild.style.width = `${Math.round(f * 100)}%`;
    status(f >= 1 ? 'Voice downloaded. Speaking…' : `Downloading the voice… ${Math.round(f * 100)}%`);
  };

  $('prov').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    voice.set({ provider: b.dataset.value }); sync(); status('');
  });
  $('voice').addEventListener('change', () => voice.set({ voiceId: $('voice').value }));
  $('dvoice').addEventListener('change', () => voice.set({ deviceVoice: $('dvoice').value }));
  const test = async () => {
    voice.unlock(); voice.lastError = '';
    if (voice.natural && !voice.ready) status('Starting the natural voice… (first time downloads ~90 MB)');
    if (voice.chatter) status('Asking Chatterbox on your PC…');
    await voice.say('Nice and sharp. Jab, cross, then get out on an angle. Keep that rear hand home.');
    $('dlbar').hidden = true;
    status(voice.lastError ? `${voice.lastError} Using the phone voice instead.` : (voice.ai ? 'That\'s the AI voice. It\'s ready.' : ''), !!voice.lastError);
  };
  $('test').addEventListener('click', test);
  $('dtest').addEventListener('click', test);
  $('cbtest').addEventListener('click', test);
  $('cbprep').addEventListener('click', () => $('prep').click());
  $('prep').addEventListener('click', async () => {
    voice.unlock();
    try {
      await voice.prepare(phrases, (d, n) => status(`Preparing coach calls ${d}/${n}…`));
      status(`Coach calls ready. They now play instantly in the ${voice.chatter ? `"${voice.cfg.cbVoice}"` : 'AI'} voice.`);
    } catch (e) { status(e.message, true); }
  });

  sync(); fillDevice();
  if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = fillDevice;
  return root;
}
