// Runs the Kokoro neural voice off the main thread so the UI and camera never stutter.
// Kokoro-82M is a free, open-source (Apache-2.0) text-to-speech model. It runs entirely on
// this device; the text is never sent anywhere. The model (~90 MB) downloads once from
// Hugging Face and is then kept in the browser cache.

const LIB = 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm';
const MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX';

let tts = null, loading = null;

function load() {
  if (tts) return Promise.resolve(tts);
  if (!loading) {
    loading = (async () => {
      const { KokoroTTS } = await import(LIB);
      tts = await KokoroTTS.from_pretrained(MODEL, {
        dtype: 'q8', device: 'wasm',
        progress_callback: (p) => {
          if (p.status === 'progress' && p.total) postMessage({ type: 'progress', loaded: p.loaded, total: p.total, file: p.file });
        },
      });
      return tts;
    })().catch((e) => { loading = null; throw e; });
  }
  return loading;
}

onmessage = async (e) => {
  const { id, type, text, voice } = e.data;
  try {
    await load();
    if (type === 'load') { postMessage({ id, ok: true }); return; }
    const a = await tts.generate(text, { voice });
    const audio = a.audio instanceof Float32Array ? a.audio : Float32Array.from(a.audio);
    postMessage({ id, audio, rate: a.sampling_rate }, [audio.buffer]);
  } catch (err) {
    postMessage({ id, error: String(err?.message || err) });
  }
};
