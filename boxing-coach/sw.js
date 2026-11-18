// Caches the app so it opens offline after the first visit. Never touches camera data.
const CACHE = 'boxing-coach-v1';
const FILES = [
  './', './index.html', './styles.css', './app.js', './lstm.js', './manifest.webmanifest',
  './icons/icon-180.png', './icons/icon-192.png', './icons/icon-512.png',
  './models/punch_lstm.bin', './models/pose_landmarker_full.task',
  './vendor/vision_bundle.mjs',
  './vendor/wasm/vision_wasm_internal.js', './vendor/wasm/vision_wasm_internal.wasm',
  './vendor/wasm/vision_wasm_nosimd_internal.js', './vendor/wasm/vision_wasm_nosimd_internal.wasm',
];

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
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request)));
});
