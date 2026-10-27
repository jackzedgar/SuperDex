// Hero background: bundles of "imagined rollouts" that leave scattered start
// states and converge on a slowly drifting goal pose. Animation is driven by
// wall-clock time, so it looks the same at any frame rate.
(function () {
  const canvas = document.getElementById('rollout-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const palette = ['#4dabf7', '#66d9e8', '#9775fa', '#ffa94d', '#74c0fc', '#63e6be'];
  const SAMPLES = 48;
  let w = 0, h = 0, rollouts = [];

  function goal(t) {
    return {
      x: w * (0.8 + 0.05 * Math.cos(t * 0.00021)),
      y: h * (0.55 + 0.15 * Math.sin(t * 0.00029)),
    };
  }

  function spawn(t, stagger) {
    const sx = -0.05 * w + Math.random() * 0.5 * w;
    const sy = Math.random() * h;
    return {
      sx, sy,
      c1x: sx + 150 + Math.random() * 250, c1y: sy + (Math.random() - 0.5) * 420,
      c2dx: -(120 + Math.random() * 260), c2dy: (Math.random() - 0.5) * 360,
      ex: (Math.random() - 0.5) * 40, ey: (Math.random() - 0.5) * 40,
      born: t - (stagger ? Math.random() * 6000 : 0),
      dur: 3500 + Math.random() * 3000,
      color: palette[Math.floor(Math.random() * palette.length)],
      width: 0.8 + Math.random() * 1.4,
    };
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth;
    h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.max(28, Math.min(90, (w * h) / 14000)));
    const now = performance.now();
    rollouts = Array.from({ length: n }, () => spawn(now, true));
  }

  // Cubic Bezier from the start state to (jittered) goal.
  function point(r, g, u) {
    const ex = g.x + r.ex, ey = g.y + r.ey;
    const c2x = ex + r.c2dx, c2y = ey + r.c2dy;
    const v = 1 - u;
    return [
      v * v * v * r.sx + 3 * v * v * u * r.c1x + 3 * v * u * u * c2x + u * u * u * ex,
      v * v * v * r.sy + 3 * v * v * u * r.c1y + 3 * v * u * u * c2y + u * u * u * ey,
    ];
  }

  function draw(t) {
    ctx.clearRect(0, 0, w, h);
    const g = goal(t);
    ctx.lineCap = 'round';
    for (let k = 0; k < rollouts.length; k++) {
      let r = rollouts[k];
      let p = reduceMotion ? 1 : (t - r.born) / r.dur;
      if (p < 0) continue;
      if (p > 1.35) { r = rollouts[k] = spawn(t, false); p = 0; }
      const head = Math.min(1, p);
      const fadeOut = p > 1 ? 1 - (p - 1) / 0.35 : 1;
      const n = Math.max(2, Math.ceil(head * SAMPLES));
      ctx.strokeStyle = r.color;
      ctx.lineWidth = r.width;
      let prev = point(r, g, 0);
      for (let i = 1; i <= n; i++) {
        const cur = point(r, g, (i / n) * head);
        // Brighter toward the head of the rollout.
        ctx.globalAlpha = (0.08 + 0.62 * Math.pow(i / n, 2)) * fadeOut;
        ctx.beginPath();
        ctx.moveTo(prev[0], prev[1]);
        ctx.lineTo(cur[0], cur[1]);
        ctx.stroke();
        prev = cur;
      }
      if (p < 1) {
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = r.color;
        ctx.beginPath();
        ctx.arc(prev[0], prev[1], r.width + 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    // Goal pose marker.
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(g.x, g.y, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(g.x, g.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.fill();
  }

  function frame(t) {
    draw(t);
    requestAnimationFrame(frame);
  }

  window.addEventListener('resize', () => { resize(); if (reduceMotion) draw(performance.now()); });
  resize();
  if (reduceMotion) {
    draw(performance.now());
  } else {
    requestAnimationFrame(frame);
  }
})();

// Tracking-kernel chart: single Gaussian vs. the dual-scale kernel.
(function () {
  const svg = document.getElementById('kernel-chart');
  if (!svg) return;
  const x0 = 46, y0 = 14, pw = 380, ph = 150, emax = 10;
  const X = (e) => x0 + (e / emax) * pw;
  const Y = (k) => y0 + (1 - k) * ph;
  const path = (f) => {
    let d = '';
    for (let i = 0; i <= 200; i++) {
      const e = (i / 200) * emax;
      d += (i ? ' L' : 'M') + X(e).toFixed(1) + ',' + Y(f(e)).toFixed(1);
    }
    return d;
  };
  const single = (e) => Math.exp(-e * e);
  const dual = (e) => 0.7 * Math.exp(-e * e) + 0.3 * Math.exp(-(e * e) / 25);
  const ns = 'http://www.w3.org/2000/svg';
  const add = (tag, attrs, text) => {
    const el = document.createElementNS(ns, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (text) el.textContent = text;
    svg.appendChild(el);
    return el;
  };
  for (let e = 0; e <= emax; e += 2) {
    add('line', { x1: X(e), x2: X(e), y1: y0, y2: y0 + ph, stroke: '#eef0f3' });
    add('text', { x: X(e), y: y0 + ph + 16, 'text-anchor': 'middle' }, String(e));
  }
  for (const k of [0, 0.5, 1]) {
    add('line', { x1: x0, x2: x0 + pw, y1: Y(k), y2: Y(k), stroke: '#eef0f3' });
    add('text', { x: x0 - 8, y: Y(k) + 4, 'text-anchor': 'end' }, k.toFixed(1));
  }
  add('line', { x1: x0, x2: x0 + pw, y1: y0 + ph, y2: y0 + ph, stroke: '#9aa1ab' });
  add('line', { x1: x0, x2: x0, y1: y0, y2: y0 + ph, stroke: '#9aa1ab' });
  add('path', { d: path(single), fill: 'none', stroke: '#adb5bd', 'stroke-width': 2, 'stroke-dasharray': '5 4' });
  add('path', { d: path(dual), fill: 'none', stroke: '#7048e8', 'stroke-width': 2.5 });
  add('text', { x: x0 + pw / 2, y: y0 + ph + 34, 'text-anchor': 'middle' }, 'tracking error  e / σ');
  // Legend in the empty upper-right corner.
  add('line', { x1: X(3.6), x2: X(4.4), y1: Y(0.93), y2: Y(0.93), stroke: '#7048e8', 'stroke-width': 2.5 });
  add('text', { x: X(4.6), y: Y(0.93) + 4, fill: '#7048e8' }, 'dual-scale κ: slope persists to ~10σ');
  add('line', { x1: X(3.6), x2: X(4.4), y1: Y(0.79), y2: Y(0.79), stroke: '#adb5bd', 'stroke-width': 2, 'stroke-dasharray': '5 4' });
  add('text', { x: X(4.6), y: Y(0.79) + 4 }, 'single Gaussian: flat beyond ~3σ');
})();

// KaTeX rendering for $...$ and \[...\].
document.addEventListener('DOMContentLoaded', function () {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
    });
  }
});
