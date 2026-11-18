# Boxing Coach

A phone web app for shadow boxing. It watches you through the camera, recognises
jab, cross, lead and rear hooks, and lead and rear uppercuts, calls them out loud,
and counts punches and names your combos.
It has an optional round timer (2 or 3 minute rounds with 1 minute rest).

Everything runs on the phone. The camera feed is never uploaded or saved.

## Put it on your iPhone (free, about 10 minutes)

iPhone only allows camera access for pages served over `https://`, so the app needs to
be hosted somewhere. GitHub Pages is free and only hosts the app's code. Your
video never goes there.

1. Make a free account at github.com (on a computer is easiest).
2. Click **+ › New repository**. Name it `boxing-coach`, set it to **Public**, and click **Create repository**.
3. On the new repository page, click **uploading an existing file**.
4. Open this `boxing-coach` folder on your PC, select **everything inside it** (including the
   `icons`, `models` and `vendor` folders), and drag it onto the GitHub page. Wait for
   the upload to finish, then click **Commit changes**.
5. Go to **Settings › Pages**. Under *Build and deployment*, set Source to **Deploy from a branch**,
   Branch to **main** and folder to **/ (root)**, then click **Save**.
6. After a minute or two, your app will be live at
   `https://YOUR-GITHUB-USERNAME.github.io/boxing-coach/`
7. On your iPhone, open that link in **Safari**, tap **Share › Add to Home Screen**.
   It now opens full screen like an app and works offline after the first load.
8. The first time you tap **Start training**, allow camera access.

If the camera is blocked: **Settings › Safari › Camera › Allow**, then reload.

## Updating the app on GitHub

Open your repository, go into the `boxing-coach` folder, click **Add file › Upload files**,
drag in everything from this folder, and click **Commit changes**. Files with the same
name are replaced. Reopen the app on your phone (twice if it still shows the old version).

## Using it

The app has four pages, switched with the tab bar at the bottom:

- **Analyse**: start a shadow-boxing session (front/back camera, round timer, voice calls),
  or score a recorded video.
- **Teach**: train the AI on your own style. It first reads your stance from your guard
  (orthodox or southpaw, which you can correct), then asks for 10 of each of the six punches,
  then 5 reps of each combo (jab-cross, double jab-cross, jab-cross-lead hook,
  jab-rear uppercut-lead hook, cross-lead hook-cross, lead uppercut-rear hook). It learns
  them on the phone and shows a self-test score. Redo any punch or combo it gets wrong.
  Once each punch has 5+ examples, Analyse uses **your model**, which also uses which hand
  moved (lead or rear) and your taught combos to call fast punches, and says the combo name.
- **Fights**: upload a clip of two fighters (e.g. a pro fight). First it shows **who's who**:
  everyone is boxed with its guess (Fighter A, Fighter B, Referee) and you tap to correct it.
  The referee is recognised by his shirt and bare hands (fighters are shirtless with gloves), and
  fighters are tracked by appearance (trunks, gloves, skin), so swapping sides or camera cuts don't
  swap them. Small arm movements, guard adjustments and the referee's gestures don't count as
  punches: a punch has to travel towards an opponent who is in shot. It also detects **knockdowns**
  (and what caused them), when the **referee steps in**, **feints** (a short, sharp move at the opponent
  that comes straight back, counted when the opponent reacts or a punch follows) and **level changes**
  (a quick drop in height, measured so camera tilts and zooms don't fake it), and links them to the
  punch they set up ("feints the jab, then throws a cross", "changes levels and digs to the body").
  Type each fighter's name in the who's-who step and it's used everywhere.
  **Thorough** mode (default) reads the clip at ¼ speed, then a second pass re-decides who's who for
  every frame using the whole clip, so one-frame swaps and flickers are removed. Review plays at normal
  speed with skeletons gliding between analysed frames. Each punch is marked **landed / blocked /
  missed** (glove on head/body, the head snapping back, or meeting the guard) and each exchange gets a
  **winner**. **Tap any punch to correct it** (type, outcome, or not a punch): corrections are saved as
  a **fight model** in the browser and, after 8, are applied to every new clip. The model can be
  exported and imported as a JSON file. It tracks both fighters,
  calls each punch (type, lead/rear, head/body), splits the clip into exchanges and writes up
  who led, the setups (jab into the cross, double jab, body-to-head…), the counters and who had
  the last word, plus each fighter's output, punch mix and go-to combination. Tap an
  exchange to replay it at half speed with both skeletons and punch labels drawn on top.
  **AI film study** plays the clip like a coach's video breakdown: it slows into each key moment,
  freezes, spotlights the fighter, draws rings, arrows and range lines, and talks through the
  positioning, setup, counter and exit. **Save as video** records it (with voice-over) to an
  .mp4/.webm file.
- **Voice** (Analyse page):
  - *Chatterbox (PC)*: Resemble AI's open-source Chatterbox running on a PC's NVIDIA GPU via a
    small local server (separate `chatterbox-server` folder, not part of this repo). Studio quality,
    free, unlimited, voice cloning from a 10–20 s sample. Works when the app is opened on that PC.
  - *Natural (AI)*: the free, open-source Kokoro voice running on the phone (no account or key;
    ~90 MB one-time download, then offline).
  - *Phone*: the built-in fallback. "Prepare coach calls" makes every live call ahead of time.
- **Tips**: setup, accuracy and technique tips.
- **Lessons**: stance, footwork and all six punches. Each has an animated 3D boxer you can
  watch from the front, ¾ or side, slowed down, mirrored for southpaws. It also lists key
  points and common mistakes, and has a **Practise** button that starts the camera and counts
  only that punch up to a target.

Prop the phone at chest height, 2–3 m away, so your shoulders, hands and hips are in shot.

## How it works

- **Pose:** Google MediaPipe Pose Landmarker (`models/pose_landmarker_full.task`) runs
  in the browser via WebAssembly and the phone's GPU.
- **Punch recognition:** the trained LSTM from the original project
  (ACM40960/Boxing, MIT licence, see `LICENSE-original.txt`). Its weights were converted to
  `models/punch_lstm.bin` and the network is reimplemented in `lstm.js`. It gives the
  same outputs as the original weights.
- The same rules as the original: 25-frame window, 70% confidence, a wrist-speed gate,
  2 s recovery between punches, and a 3 s combo window. Frames are resampled to 30 fps so
  the model sees the same timing it was trained on, whatever the phone's frame rate.
- `models/block_lstm.bin` (the original block classifier) is included but not used yet,
  as in the original app.
- **Fight punch model** (`models/fight_punch_mlp.json`, Fights tab): a small neural network
  (62 → 96 → 64 → 6) that names each punch found in a fight clip. It was trained on the
  **BoxingVI** skeleton benchmark (Kumar, Baghel, Singh, Badatya, Yadav, Srinivasan and Hegde,
  *BoxingVI: A Multi-Modal Benchmark for Boxing Action Recognition and Localization*,
  arXiv:2511.16524, 2025; github.com/Bikudebug/BoxingVI): about 4,750 labelled punches from nine
  of its videos. The file holds network weights only; none of the BoxingVI videos, skeletons or
  labels are included in this repository. The BoxingVI repository does not state a licence.
  On boxers it hadn't seen it was right **68%** of the time across the six punch types
  (cross ~97%, jab ~80%, lead hook ~60%, rear uppercut 60–89%, lead uppercut ~45%, rear hook poor),
  against 32.5% for the hand-written rules it replaces. Corrections you make in the app are
  layered on top of it.

## Files

| Path | What it is |
|---|---|
| `index.html`, `styles.css`, `app.js` | Pages, sessions and Teach mode |
| `engine.js` | Pose → 30 fps stream → punch detection |
| `personal.js` | Your model: stance, taught punches and combos, k-nearest-neighbour classifier, self-test |
| `filmstudy.js` | AI film study: commentary script, paused highlights, voice-over, video export |
| `voice.js`, `voice-worker.js` | Voice: Kokoro neural TTS in a worker (cached) + phone-voice fallback, settings panel |
| `people.js` | Who's who: appearance sampling, referee detection, appearance-based tracking, knockdown check |
| `breakdown.js` | Fight breakdown: two-fighter tracking, punch calls, exchanges and the write-up |
| `lessons.js` | Lessons pages and the animated boxer (keyframed rig drawn on a canvas) |
| `plexus.js` | Animated background |
| `lstm.js` | Starter punch classifier (original project's network) |
| `models/` | Pose model, converted classifier weights, and the BoxingVI-trained fight punch model |
| `vendor/` | MediaPipe runtime (bundled so nothing loads from other sites) |
| `sw.js`, `manifest.webmanifest`, `icons/` | Home-screen install and offline support |
