# Boxing Coach

A phone web app for shadow boxing. It watches you through the camera, recognises
jab, cross, hook and uppercut, calls them out loud, and counts punches and combos.
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

## Using it

- Prop the phone at chest height, 2–3 m away, so your shoulders, hands and hips are in shot.
- **Front camera** lets you watch yourself. The **back camera** gives better quality.
- **Analyse a recorded video** lets you film a clip (or pick one) and have it scored.

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

## Files

| Path | What it is |
|---|---|
| `index.html`, `styles.css`, `app.js` | The app |
| `lstm.js` | Punch classifier network |
| `models/` | Pose model and converted classifier weights |
| `vendor/` | MediaPipe runtime (bundled so nothing loads from other sites) |
| `sw.js`, `manifest.webmanifest`, `icons/` | Home-screen install and offline support |
