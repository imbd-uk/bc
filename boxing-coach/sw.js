// Offline support. App code is fetched fresh when online (so updates show up),
// large model/runtime files are served from cache. Never touches camera data.
const CACHE = 'boxing-coach-v12';
const FILES = [
  './', './index.html', './styles.css', './app.js', './engine.js', './personal.js', './plexus.js', './lstm.js', './lessons.js', './breakdown.js',
  './filmstudy.js', './voice.js', './voice-worker.js', './people.js',
  './manifest.webmanifest', './icons/boxer-180.png', './icons/boxer-192.png', './icons/boxer-512.png',
  './models/punch_lstm.bin', './models/pose_landmarker_full.task', './models/fight_punch_mlp.json',
  './vendor/vision_bundle.mjs',
  './vendor/wasm/vision_wasm_internal.js', './vendor/wasm/vision_wasm_internal.wasm',
  './vendor/wasm/vision_wasm_nosimd_internal.js', './vendor/wasm/vision_wasm_nosimd_internal.wasm',
];
const BIG = /\/(models|vendor)\//;

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;
  if (BIG.test(url.pathname)) {
    e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request)));
    return;
  }
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
        return res;
      })
      .catch(() => caches.match(e.request, { ignoreSearch: true })),
  );
});
