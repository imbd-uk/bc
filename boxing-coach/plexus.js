// Animated background: a slowly rolling wireframe "plexus" surface with drifting dust.
// Drawn on a canvas, capped at ~30 fps, paused when hidden.

export function startPlexus(canvas) {
  const ctx = canvas.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const COLS = 15, ROWS = 7;
  let w = 0, h = 0, dpr = 1, raf = 0, running = true, lastDraw = 0;

  // Surface vertices with a little random jitter so it reads as a network, not a grid.
  const verts = [];
  const rnd = mulberry(7);
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      verts.push({
        u: (c + (rnd() - 0.5) * 0.7) / (COLS - 1),
        v: (r + (rnd() - 0.5) * 0.7) / (ROWS - 1),
        ph: rnd() * Math.PI * 2,
        glow: rnd() < 0.18,
      });
    }
  }
  const edges = [];
  const id = (r, c) => r * COLS + c;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (c < COLS - 1) edges.push([id(r, c), id(r, c + 1)]);
      if (r < ROWS - 1) edges.push([id(r, c), id(r + 1, c)]);
      if (r < ROWS - 1 && c < COLS - 1 && rnd() < 0.55) edges.push(rnd() < 0.5 ? [id(r, c), id(r + 1, c + 1)] : [id(r, c + 1), id(r + 1, c)]);
    }
  }
  const dust = Array.from({ length: 90 }, () => ({ x: rnd(), y: rnd(), z: 0.3 + rnd() * 0.7, s: rnd() }));

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    draw(performance.now());
  }

  function project(vx, time) {
    // Surface spans wider than the screen; depth runs into the distance.
    const x3 = (vx.u - 0.5) * 2.6;
    const z3 = 0.6 + vx.v * 2.2;
    const y3 = 0.18 * Math.sin(vx.u * 5.2 + time * 0.35 + vx.v * 1.6)
             + 0.10 * Math.sin(vx.u * 9.0 - time * 0.22 + vx.ph)
             - 0.06 * vx.v;
    const f = Math.min(w, h * 1.4) * 0.9;
    return { x: w / 2 + (x3 / z3) * f, y: h * 0.62 + ((y3 + 0.25) / z3) * f * 0.9, d: z3 };
  }

  function draw(now) {
    const time = reduce ? 4 : now / 1000;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // dust
    for (const p of dust) {
      const x = ((p.x + time * 0.004 * p.z) % 1) * w;
      const y = (p.y * 0.9 + 0.05) * h + Math.sin(time * 0.3 + p.s * 6) * 4;
      ctx.fillStyle = `rgba(255,255,255,${0.08 + p.z * 0.25})`;
      ctx.fillRect(x, y, p.z * 1.6, p.z * 1.6);
    }

    const P = verts.map((v) => project(v, time));
    ctx.lineWidth = 0.7;
    for (const [a, b] of edges) {
      const pa = P[a], pb = P[b];
      const fade = Math.max(0, 1 - (pa.d + pb.d) / 2 / 3.0);
      ctx.strokeStyle = `rgba(255,255,255,${0.05 + fade * 0.32})`;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    }
    P.forEach((p, i) => {
      const fade = Math.max(0, 1 - p.d / 3.0);
      const r = (verts[i].glow ? 2.2 : 1.3) * (0.6 + fade);
      if (verts[i].glow) {
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 6);
        g.addColorStop(0, `rgba(255,255,255,${0.35 * fade})`);
        g.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, r * 6, 0, Math.PI * 2); ctx.fill();
      }
      ctx.fillStyle = `rgba(255,255,255,${0.25 + fade * 0.7})`;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
    });
  }

  function frame(now) {
    if (!running) return;
    if (now - lastDraw > 33) { lastDraw = now; draw(now); }
    raf = requestAnimationFrame(frame);
  }

  addEventListener('resize', resize);
  resize();
  if (!reduce) raf = requestAnimationFrame(frame);

  return {
    pause() { running = false; cancelAnimationFrame(raf); },
    resume() { if (running || reduce) return; running = true; raf = requestAnimationFrame(frame); },
  };
}

function mulberry(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
