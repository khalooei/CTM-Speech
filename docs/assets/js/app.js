/* F-CTM project page: hero animation, interactive restoration demo, results explorer. No dependencies. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CODEC_NAMES = { dac: "DAC", mimi: "Mimi", speechtokenizer: "SpeechTokenizer" };
  const RANK_NAMES = { intervention: "Oracle (ours)", random: "Random", rvq_early: "RVQ-early", displacement: "Codeword displacement", energy: "Perturbation energy" };
  const RANK_STYLE = { intervention: ["--series-1", false], rvq_early: ["--series-2", false], displacement: ["--series-3", false], energy: ["--series-4", false], random: ["--null", true] };
  const COND_LABEL = (c) => c.replace("pgd_ctc_eps", "PGD ε=").replace("gaussian_snr", "Gaussian ").replace("esc50_snr", "ESC-50 ").replace("esc10_snr", "ESC-10 ")
    .replace("pink_snr", "pink ").replace("brown_snr", "brown ").replace("bitcrush_10bit", "10-bit crush").replace("opus_", "Opus ").replace(/(\d+)k$/, "$1 kbit/s").replace(/(Gaussian|ESC-50|ESC-10|pink|brown) (\d+)$/, "$1 $2 dB");
  const pct = (v, d = 0) => `${(100 * v).toFixed(d)}%`;
  const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : (+v).toFixed(d));
  const redraws = [];

  /* ---------------- theme ---------------- */
  const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } } };
  const saved = store.get("fctm-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-btn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    store.set("fctm-theme", next);
    redraws.forEach((f) => f());
  });

  /* ---------------- hero: a token matrix where most cells flip but few matter ---------------- */
  (function hero() {
    const cv = $("#hero-canvas"), ctx = cv.getContext("2d");
    let W, H, cols, rows, cell = 18, state = [], t0 = performance.now();
    function init() {
      const dpr = devicePixelRatio || 1;
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cols = Math.ceil(W / cell); rows = Math.ceil(H / cell);
      state = Array.from({ length: cols * rows }, () => ({ flip: Math.random(), crit: Math.random() < 0.035, ph: Math.random() * 6.28 }));
    }
    function frame(now) {
      const t = (now - t0) / 1000;
      const grid = css("--grid"), orange = css("--series-2"), blue = css("--series-1");
      ctx.clearRect(0, 0, W, H);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const s = state[r * cols + c];
        const wave = 0.5 + 0.5 * Math.sin(t * 0.9 + s.ph + c * 0.07);
        let col = grid, a = 0.9;
        if (s.crit) { col = blue; a = 0.35 + 0.65 * wave; }
        else if (s.flip < 0.62 * wave + 0.15) { col = orange; a = 0.18; }
        ctx.globalAlpha = a; ctx.fillStyle = col;
        ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4);
      }
      ctx.globalAlpha = 1;
      if (!reduceMotion) requestAnimationFrame(frame);
    }
    init(); addEventListener("resize", init); requestAnimationFrame(frame);
    redraws.push(() => reduceMotion && requestAnimationFrame(frame));
  })();

  let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => redraws.forEach((f) => f()), 150); });

  /* ---------------- reveal + counters ---------------- */
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    e.target.classList.add("in");
    e.target.querySelectorAll("[data-count]").forEach(countUp);
    io.unobserve(e.target);
  }), { threshold: 0.15 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
  function countUp(el) {
    if (reduceMotion || el.dataset.done) return;
    el.dataset.done = 1;
    const a = +el.dataset.count, b = el.dataset.to ? +el.dataset.to : null, d = +(el.dataset.dec || 0), sep = el.dataset.sep;
    const show = (v) => (sep ? Math.round(v).toLocaleString("en-US") : v.toFixed(d));
    const start = performance.now(), dur = 1300;
    (function step(now) {
      const p = Math.min(1, (now - start) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = b == null ? show(a * e) : `${show(a * e)}–${show(b * e)}`;
      if (p < 1) requestAnimationFrame(step);
    })(start);
  }

  $("#copy-bib").addEventListener("click", (ev) => {
    const txt = $("#bibtex").firstChild.textContent;
    const done = (msg) => { ev.target.textContent = msg; setTimeout(() => (ev.target.textContent = "Copy"), 1400); };
    try { navigator.clipboard.writeText(txt).then(() => done("Copied"), () => done("Select & copy")); } catch { done("Select & copy"); }
  });

  /* ---------------- shared tooltip ---------------- */
  const tip = $("#tip");
  function showTip(evt, html) {
    tip.innerHTML = html; tip.classList.add("on");
    const pad = 14, r = tip.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.position = "fixed"; tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  }
  const hideTip = () => tip.classList.remove("on");

  /* ---------------- small SVG chart kit ---------------- */
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); parent && parent.appendChild(n); return n; };
  // densify a curve so straight segments stay straight in data space under a non-linear x scale
  const dense = (xs, ys) => { const out = []; for (let i = 0; i < xs.length; i++) { if (i) { const n = Math.min(24, Math.ceil((xs[i] - xs[i - 1]) * 200)); for (let j = 1; j < n; j++) { const f = j / n; out.push([xs[i - 1] + f * (xs[i] - xs[i - 1]), ys[i - 1] + f * (ys[i] - ys[i - 1])]); } } out.push([xs[i], ys[i]]); } return out; };
  const interp = (xs, ys, x) => {
    if (x <= xs[0]) return ys[0];
    for (let i = 1; i < xs.length; i++) if (x <= xs[i]) { const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] || 1); return ys[i - 1] + t * (ys[i] - ys[i - 1]); }
    return ys[ys.length - 1];
  };

  // x uses a square-root scale by default: recovery curves change fastest at small budgets.
  function lineChart(host, { series, xDom = [0, 1], yDom = [-0.1, 1.05], xLabel, yLabel, marker = null, refY = null, height = 300, xTicks = [0, .01, .05, .1, .25, .5, 1], yTicks = [0, .25, .5, .75, 1], sqrtX = true }) {
    host.innerHTML = "";
    const W = Math.max(300, host.clientWidth || 600), H = height, m = { l: 54, r: 14, t: 10, b: 44 };
    const fx = (x) => { const u = (x - xDom[0]) / (xDom[1] - xDom[0]); return sqrtX ? Math.sqrt(Math.max(0, u)) : u; };
    const ifx = (f) => xDom[0] + (sqrtX ? f * f : f) * (xDom[1] - xDom[0]);
    const sx = (x) => m.l + fx(x) * (W - m.l - m.r);
    const sy = (y) => H - m.b - (y - yDom[0]) / (yDom[1] - yDom[0]) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": yLabel }, host);
    const ax = el("g", { class: "axis" }, svg);
    if (yDom[0] < -0.3) { const st = yDom[0] < -2 ? 1 : 0.5; yTicks = []; for (let v = Math.ceil(yDom[0] / st) * st; v <= 1.0001; v += st) yTicks.push(+v.toFixed(2)); }
    yTicks.forEach((v) => { el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, ax); el("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end" }, ax).textContent = v; });
    xTicks.forEach((v) => { el("line", { class: "gridline", x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, opacity: 0.5 }, ax); el("text", { x: sx(v), y: H - m.b + 18, "text-anchor": "middle" }, ax).textContent = pct(v); });
    el("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: sy(Math.max(0, yDom[0])), y2: sy(Math.max(0, yDom[0])) }, ax);
    el("text", { class: "axis-title", x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = `${xLabel} (√ scale)`;
    el("text", { class: "axis-title", x: 12, y: (H - m.b) / 2, transform: `rotate(-90 12 ${(H - m.b) / 2})`, "text-anchor": "middle" }, svg).textContent = yLabel;
    if (refY != null) el("line", { x1: m.l, x2: W - m.r, y1: sy(refY), y2: sy(refY), stroke: css("--text-muted"), "stroke-dasharray": "2 4", "stroke-width": 1 }, svg);
    const clipY = (y) => Math.max(yDom[0], Math.min(yDom[1], y));
    series.filter((s) => s.visible !== false).forEach((s) => {
      if (s.lo && s.hi) {
        const up = dense(s.x, s.hi).map(([x, y]) => `${sx(x)},${sy(clipY(y))}`), dn = dense(s.x, s.lo).map(([x, y]) => `${sx(x)},${sy(clipY(y))}`).reverse();
        el("polygon", { points: up.concat(dn).join(" "), fill: s.color, opacity: 0.13 }, svg);
      }
      el("polyline", { points: dense(s.x, s.y).map(([x, y]) => `${sx(x)},${sy(clipY(y))}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.dash ? "6 5" : "none" }, svg);
    });
    let mk = null;
    if (marker != null) mk = el("line", { x1: sx(marker), x2: sx(marker), y1: m.t, y2: H - m.b, stroke: css("--text-secondary"), "stroke-width": 1.2 }, svg);
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--text-muted"), "stroke-width": 1, opacity: 0 }, svg);
    const dots = series.filter((s) => s.visible !== false).map((s) => el("circle", { r: 4.5, fill: s.color, stroke: css("--surface-1"), "stroke-width": 2, opacity: 0 }, svg));
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (evt) => {
      const r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) / r.width * W;
      const x = ifx(Math.max(0, Math.min(1, (px - m.l) / (W - m.l - m.r))));
      cross.setAttribute("x1", sx(x)); cross.setAttribute("x2", sx(x)); cross.setAttribute("opacity", 1);
      const vis = series.filter((s) => s.visible !== false);
      let rows = "";
      vis.forEach((s, i) => { const y = interp(s.x, s.y, x); dots[i].setAttribute("cx", sx(x)); dots[i].setAttribute("cy", sy(clipY(y))); dots[i].setAttribute("opacity", 1);
        rows += `<div class="tr"><span><i class="k" style="background:${s.color}"></i>${s.name}</span><b>${fmt(y)}</b></div>`; });
      showTip(evt, `<div class="tt">${pct(x, x < 0.1 ? 1 : 0)} of changed cells restored</div>${rows}`);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); dots.forEach((d) => d.setAttribute("opacity", 0)); hideTip(); });
    return { setMarker: (x) => mk && (mk.setAttribute("x1", sx(x)), mk.setAttribute("x2", sx(x))) };
  }

  function legend(host, items, onToggle) {
    host.innerHTML = "";
    items.forEach((it) => {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("aria-pressed", it.visible !== false);
      b.innerHTML = it.shape ? `${it.shape}${it.name}` : `<span class="ln${it.dash ? " dash" : ""}" style="border-color:${it.color}"></span>${it.name}`;
      if (onToggle) b.addEventListener("click", () => { it.visible = it.visible === false; b.setAttribute("aria-pressed", it.visible !== false); onToggle(); });
      else b.style.cursor = "default";
      host.appendChild(b);
    });
  }

  function table(host, head, rows) {
    host.innerHTML = `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  /* ---------------- word-level diff for transcripts ---------------- */
  function diffWords(ref, hyp) {
    const a = ref.trim().split(/\s+/).filter(Boolean), b = hyp.trim().split(/\s+/).filter(Boolean);
    const D = Array.from({ length: a.length + 1 }, (_, i) => Array.from({ length: b.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++)
      D[i][j] = Math.min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    const out = []; let i = a.length, j = b.length;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && D[i][j] === D[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)) { out.push(a[i - 1] === b[j - 1] ? [b[j - 1], ""] : [b[j - 1], "sub"]); i--; j--; }
      else if (j > 0 && D[i][j] === D[i][j - 1] + 1) { out.push([b[j - 1], "ins"]); j--; }
      else { out.push([a[i - 1], "del"]); i--; }
    }
    return out.reverse().map(([w, c]) => (c ? `<span class="${c}" title="${c === "sub" ? "substitution" : c === "ins" ? "insertion" : "deleted reference word"}">${w.toLowerCase()}</span>` : w.toLowerCase())).join(" ");
  }

  /* ---------------- data ---------------- */
  Promise.all([fetch("assets/data/cases.json").then((r) => r.json()), fetch("assets/data/summary.json").then((r) => r.json())])
    .then(([cases, summary]) => { initDemo(cases); initResults(summary); })
    .catch((e) => {
      $("#case-panel").innerHTML = `<div class="loading">Could not load the demo data (${e.message}). If you opened this file directly, serve the folder instead: <code>python -m http.server -d docs</code></div>`;
    });

  /* ================= interactive demo ================= */
  function initDemo(meta) {
    const cases = meta.cases, budgets = meta.budgets;
    const codecs = [...new Set(cases.map((c) => c.codec))], conds = [...new Set(cases.map((c) => c.condition))];
    let codec = codecs.includes("mimi") ? "mimi" : codecs[0], cond = conds[0], order = "oracle";
    const segC = $("#codec-seg"), segK = $("#cond-seg");
    codecs.forEach((c) => segC.appendChild(segBtn(CODEC_NAMES[c] || c, () => { codec = c; sync(); render(); }, c)));
    conds.forEach((c) => segK.appendChild(segBtn(cases.find((x) => x.condition === c).family, () => { cond = c; sync(); render(); }, c)));
    function segBtn(label, fn, key) { const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.dataset.key = key; b.addEventListener("click", fn); return b; }
    function sync() { segC.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.key === codec)); segK.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.key === cond)); }
    sync();

    let state = null;
    function render() {
      stopSweep();
      const c = cases.find((x) => x.codec === codec && x.condition === cond);
      const panel = $("#case-panel");
      if (!c) { panel.innerHTML = `<div class="loading">No case for this combination.</div>`; return; }
      panel.innerHTML = `
        <div class="case-head">
          <div><div class="kicker" style="margin-bottom:6px">Reference transcript</div><div class="ref">“${c.text.toLowerCase()}”</div></div>
          <div class="chips"><span class="chip">${CODEC_NAMES[c.codec]} · Q=${c.Q} × T=${c.T}</span><span class="chip">${c.family}</span><span class="chip">TCR ${fmt(c.tcr)}</span><span class="chip">|𝒟| = ${c.n_changed.toLocaleString("en-US")}</span><span class="chip">utt ${c.utterance_id}</span></div>
        </div>
        <div class="inputs"><label>Clean input <audio controls preload="none" src="assets/${c.files.input_clean}"></audio></label><label>Corrupted input <audio controls preload="none" src="assets/${c.files.input_corrupt}"></audio></label></div>
        <div class="grid-wrap">
          <div class="picker" style="justify-content:space-between"><div class="seg" id="order-seg"><button type="button" data-o="oracle">Grid: oracle order</button><button type="button" data-o="random">Grid: random order</button></div>
          <div class="grid-legend"><span><i class="sw" style="background:var(--grid)"></i>unchanged</span><span><i class="sw" style="background:var(--series-2)"></i>flipped (corrupted)</span><span><i class="sw" style="background:var(--series-1)"></i>restored to clean</span><span><i class="sw" style="background:linear-gradient(90deg,transparent,var(--seq-hi))"></i>frame influence</span></div></div>
          <canvas id="token-canvas" aria-label="Token matrix of RVQ stages by frames; flipped cells are restored in ranking order as the budget grows"></canvas>
          <div class="axis-note"><span>stage 0 (coarse) at top · t = 0</span><span>time → ${c.duration_s} s</span></div>
        </div>
        <div class="budget">
          <button class="play" id="sweep" type="button" aria-label="Play the restoration sweep"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4v16l13-8z"/></svg></button>
          <div><input type="range" id="budget" min="0" max="${budgets.length - 1}" step="1" value="3" aria-label="Restoration budget">
            <div class="ticks">${budgets.map((b) => `<span>${pct(b)}</span>`).join("")}</div></div>
          <div class="budget-val" id="budget-val">5%<small>k = 0 cells</small></div>
        </div>
        <div class="compare">
          ${["oracle", "random"].map((o) => `
          <div class="lane ${o}" id="lane-${o}">
            <h3>${o === "oracle" ? "Oracle ranking (F-CTM)" : "Random ranking (null)"}<span>${o === "oracle" ? "frames → cells, wav2vec2-base" : "same budget, seeded"}</span></h3>
            <div class="recbar"><i style="width:0"></i></div>
            <div class="recrow"><span>recovered share of the loss gap</span><b class="rec">–</b></div>
            <div class="recrow"><span>WER (held-out HuBERT-L)</span><b class="wer">–</b></div>
            <div class="hyp"></div>
            <audio controls preload="none"></audio>
          </div>`).join("")}
        </div>
        <div class="case-curve"><p class="cap muted" style="font-size:.84rem;margin:0 0 6px">Curves: this row as released with the paper (≈30 prefix points, ranking of the original run). The lanes above were recomputed for this page on different hardware; when the loss gap is small, single points are noisy, which is why the paper reports medians over rows and a per-row resolution test.</p><div class="legend" id="case-legend"></div><div class="chart" id="case-chart"></div></div>`;

      const pos = {}; ["oracle", "random"].forEach((o) => { pos[o] = c.cells[`rank_${o}`]; });
      state = { c, k: 0, target: 0, bi: 3, restoredAt: new Float64Array(c.n_changed).fill(-1) };
      const finfl = c.frame_influence, fmax = Math.max(1e-9, ...Object.values(finfl).filter((v) => v > 0));
      const cv = $("#token-canvas"), ctx = cv.getContext("2d");
      const changedAt = new Int32Array(c.Q * c.T).fill(-1);
      c.cells.q.forEach((q, i) => (changedAt[q * c.T + c.cells.t[i]] = i));

      function draw(now = performance.now()) {
        const dpr = devicePixelRatio || 1, W = cv.clientWidth;
        const rowH = Math.max(7, Math.min(18, Math.floor(260 / c.Q))), strip = 12, H = c.Q * rowH + strip + 4;
        if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = W * dpr; cv.height = H * dpr; cv.style.height = `${H}px`; }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);
        const cw = W / c.T, gap = cw > 3 ? 1 : 0;
        const grid = css("--grid"), orange = css("--series-2"), blue = css("--series-1"), hi = css("--seq-hi");
        const rk = pos[order], kk = state.k;
        for (let q = 0; q < c.Q; q++) for (let t = 0; t < c.T; t++) {
          const i = changedAt[q * c.T + t]; let col = grid, a = 1;
          if (i >= 0) {
            if (rk[i] < kk) {
              col = blue;
              if (state.restoredAt[i] < 0) state.restoredAt[i] = now;
              const age = (now - state.restoredAt[i]) / 600; a = age < 1 ? 1 : 0.85;
            } else { col = orange; a = 0.85; state.restoredAt[i] = -1; }
          }
          ctx.globalAlpha = a; ctx.fillStyle = col;
          ctx.fillRect(t * cw, q * rowH, Math.max(1, cw - gap), rowH - 1);
          if (i >= 0 && rk[i] < kk && now - state.restoredAt[i] < 600 && !reduceMotion) {
            ctx.globalAlpha = 1 - (now - state.restoredAt[i]) / 600; ctx.fillStyle = "#fff";
            ctx.fillRect(t * cw, q * rowH, Math.max(1, cw - gap), rowH - 1);
          }
        }
        ctx.globalAlpha = 1;
        const y0 = c.Q * rowH + 4;
        for (let t = 0; t < c.T; t++) {
          const v = finfl[t] || 0;
          ctx.globalAlpha = v > 0 ? 0.15 + 0.85 * Math.min(1, v / fmax) : 0.06; ctx.fillStyle = hi;
          ctx.fillRect(t * cw, y0, Math.max(1, cw - gap), strip);
        }
        ctx.globalAlpha = 1;
      }
      let raf = null;
      function animate() {
        cancelAnimationFrame(raf);
        const step = () => {
          const d = state.target - state.k;
          state.k = Math.abs(d) < 0.5 ? state.target : state.k + d * (reduceMotion ? 1 : 0.12);
          draw();
          if (state.k !== state.target || performance.now() - lastSet < 700) raf = requestAnimationFrame(step);
        };
        raf = requestAnimationFrame(step);
      }
      let lastSet = 0;
      cv.addEventListener("mousemove", (evt) => {
        const r = cv.getBoundingClientRect(), rowH = Math.max(7, Math.min(18, Math.floor(260 / c.Q)));
        const t = Math.floor((evt.clientX - r.left) / r.width * c.T), q = Math.floor((evt.clientY - r.top) / rowH);
        if (q < 0 || q >= c.Q || t < 0 || t >= c.T) return hideTip();
        const i = changedAt[q * c.T + t], sec = (t / c.frame_rate).toFixed(2);
        const extra = i >= 0 ? `<div class="tr"><span>oracle rank</span><b>${pos.oracle[i] + 1} / ${c.n_changed}</b></div><div class="tr"><span>random rank</span><b>${pos.random[i] + 1}</b></div>` : `<div class="tr"><span>token</span><b>unchanged</b></div>`;
        showTip(evt, `<div class="tt">stage ${q} · frame ${t} · ${sec} s</div>${extra}<div class="tr"><span>frame influence</span><b>${fmt(finfl[t] ?? 0, 3)}</b></div>`);
      });
      cv.addEventListener("mouseleave", hideTip);

      const orderSeg = $("#order-seg");
      const syncOrder = () => orderSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.o === order));
      orderSeg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { order = b.dataset.o; syncOrder(); state.restoredAt.fill(-1); animate(); }));
      syncOrder();

      // per-case curves (from the released row, same scorer)
      const items = Object.keys(c.curves).map((rk) => ({ key: rk, name: `${RANK_NAMES[rk] || rk} · AURC ${fmt(c.curves[rk].aurc)}`, color: css(RANK_STYLE[rk][0]), dash: RANK_STYLE[rk][1], visible: rk === "intervention" || rk === "random" }));
      let caseChart = null;
      const drawCase = () => {
        items.forEach((it) => (it.color = css(RANK_STYLE[it.key][0])));
        caseChart = lineChart($("#case-chart"), {
          series: items.map((it) => ({ ...it, name: RANK_NAMES[it.key], x: c.curves[it.key].x, y: c.curves[it.key].y })),
          xLabel: "fraction of changed cells restored", yLabel: "recovery (held-out)", marker: budgets[state.bi], refY: 0.9, height: 260,
          yDom: [Math.min(-0.1, ...Object.values(c.curves).flatMap((v) => v.y)), 1.08],
        });
      };
      legend($("#case-legend"), items, drawCase);
      drawCase();

      const slider = $("#budget");
      function setBudget(bi, { play = false } = {}) {
        state.bi = bi; slider.value = bi;
        const b = budgets[bi];
        state.target = Math.round(b * c.n_changed); lastSet = performance.now(); animate();
        $("#budget-val").innerHTML = `${pct(b)}<small>k = ${state.target.toLocaleString("en-US")} cells</small>`;
        ["oracle", "random"].forEach((o) => {
          const s = c.steps[o][bi], lane = $(`#lane-${o}`);
          lane.querySelector(".recbar i").style.width = `${Math.max(0, Math.min(1, s.recovery)) * 100}%`;
          lane.querySelector(".rec").textContent = fmt(s.recovery) + (s.recovery < -0.05 ? " · worse than corrupted" : s.recovery > 1.05 ? " · better than clean codes" : "");
          lane.querySelector(".wer").textContent = pct(s.wer);
          lane.querySelector(".hyp").innerHTML = s.hyp ? diffWords(c.text, s.hyp) : "<i>(empty transcript)</i>";
          const au = lane.querySelector("audio");
          const wasPlaying = !au.paused;
          au.src = `assets/${s.audio}`;
          if (play || wasPlaying) au.play().catch(() => {});
        });
        caseChart && caseChart.setMarker(b);
      }
      slider.addEventListener("input", () => setBudget(+slider.value));
      setBudget(3);
      redraws.push(() => { if (state && state.c === c) { draw(); drawCase(); } });
      addEventListener("resize", () => draw());

      // sweep: play the oracle hybrid at every budget in turn
      $("#sweep").addEventListener("click", () => (sweep ? stopSweep() : startSweep()));
      function startSweep() {
        let bi = 0;
        const au = $("#lane-oracle audio");
        sweep = true; $("#sweep").innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>`;
        const next = () => { if (!sweep) return; if (bi >= budgets.length) return stopSweep(); setBudget(bi++); au.onended = next; au.play().catch(stopSweep); };
        next();
      }
    }
    let sweep = false;
    function stopSweep() {
      sweep = false;
      const btn = $("#sweep"); if (btn) btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4v16l13-8z"/></svg>`;
      const au = $("#lane-oracle audio"); if (au) au.onended = null;
    }
    render();
  }

  /* ================= results explorer ================= */
  function initResults(S) {
    const codecs = Object.keys(S.codecs);
    let codec = "dac";
    const seg = $("#res-codec-seg");
    codecs.forEach((c) => { const b = document.createElement("button"); b.type = "button"; b.textContent = `${CODEC_NAMES[c]} (n=${S.codecs[c].n})`; b.dataset.key = c;
      b.addEventListener("click", () => { codec = c; seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x.dataset.key === codec)); drawAll(); }); seg.appendChild(b); });
    seg.querySelector("button").setAttribute("aria-pressed", "true");
    const order = ["intervention", "rvq_early", "displacement", "energy", "random"];
    const vis = { intervention: true, rvq_early: true, displacement: true, energy: false, random: true };

    function drawCurves() {
      const C = S.codecs[codec].curves;
      const items = order.filter((k) => C[k]).map((k) => ({ key: k, name: RANK_NAMES[k], color: css(RANK_STYLE[k][0]), dash: RANK_STYLE[k][1], visible: vis[k] }));
      legend($("#curve-legend"), items, () => { items.forEach((it) => (vis[it.key] = it.visible !== false)); drawCurves(); });
      lineChart($("#curve-chart"), {
        series: items.map((it) => ({ ...it, x: S.grid, y: C[it.key].median, lo: it.key === "intervention" || it.key === "random" ? C[it.key].q25 : null, hi: it.key === "intervention" || it.key === "random" ? C[it.key].q75 : null })),
        xLabel: "fraction of changed cells restored", yLabel: "median recovery (held-out)", refY: 0.9, yDom: [-0.2, 1.05], yTicks: [0, .25, .5, .75, 1],
      });
      $("#curve-cap").textContent = `${CODEC_NAMES[codec]}: median over ${S.codecs[codec].n} gated rows resolved by HuBERT-large; shaded = interquartile range; dotted = α = 0.9.`;
      const at = [0.01, 0.05, 0.1, 0.25, 0.5];
      table($("#curve-table"), ["Ranking", ...at.map((a) => pct(a))], order.filter((k) => C[k]).map((k) => [RANK_NAMES[k], ...at.map((a) => fmt(interp(S.grid, C[k].median, a)))]));
    }

    function drawStage() {
      const st = S.stage[codec], host = $("#stage-chart"); host.innerHTML = "";
      const Q = st.alone.length, W = Math.max(300, host.clientWidth || 460), H = 300, m = { l: 44, r: 10, t: 10, b: 42 };
      const lo = Math.min(-0.2, ...st.alone), hi = 1.05;
      const sy = (y) => H - m.b - (y - lo) / (hi - lo) * (H - m.t - m.b), bw = (W - m.l - m.r) / Q;
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "per-stage restoration" }, host);
      const ax = el("g", { class: "axis" }, svg);
      [0, 0.25, 0.5, 0.75, 1].forEach((v) => { el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, ax); el("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end" }, ax).textContent = v; });
      el("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0) }, ax);
      el("text", { class: "axis-title", x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = "RVQ stage q";
      const blue = css("--series-1"), gray = css("--null");
      for (let q = 0; q < Q; q++) {
        const x0 = m.l + q * bw, w = Math.max(3, bw / 2 - 3);
        el("text", { x: x0 + bw / 2, y: H - m.b + 16, "text-anchor": "middle", class: "lbl" }, svg).textContent = q;
        const bars = [[st.flip_share[q], gray, x0 + 2], [st.alone[q], blue, x0 + bw / 2 + 1]];
        bars.forEach(([v, col, x]) => { const y1 = sy(Math.max(0, v)), y2 = sy(Math.min(0, v)); el("rect", { x, y: y1, width: w, height: Math.max(1, y2 - y1), rx: 2, fill: col }, svg); });
        const hit = el("rect", { x: x0, y: m.t, width: bw, height: H - m.t - m.b, fill: "transparent" }, svg);
        hit.addEventListener("mousemove", (evt) => showTip(evt, `<div class="tt">${CODEC_NAMES[codec]} · stage ${q}</div><div class="tr"><span><i class="k" style="background:${gray}"></i>share of flips</span><b>${pct(st.flip_share[q], 1)}</b></div><div class="tr"><span><i class="k" style="background:${blue}"></i>recovery, stage alone</span><b>${fmt(st.alone[q])}</b></div>`));
        hit.addEventListener("mouseleave", hideTip);
      }
      legend($("#stage-legend"), [{ name: "share of flipped cells", shape: `<i class="sw" style="background:${gray}"></i>` }, { name: "median recovery, stage restored alone", shape: `<i class="sw" style="background:${blue}"></i>` }]);
      table($("#stage-table"), ["Stage", "Share of flips", "Recovery (alone)"], st.alone.map((v, q) => [q, pct(st.flip_share[q], 1), fmt(v)]));
    }

    function drawScatter() {
      const host = $("#scatter-chart"); host.innerHTML = "";
      const W = Math.max(320, host.clientWidth || 1000), H = 320, m = { l: 56, r: 20, t: 12, b: 44 }, xd = [0.4, 1.0], yd = [0, 1];
      const sx = (x) => m.l + (x - xd[0]) / (xd[1] - xd[0]) * (W - m.l - m.r), sy = (y) => H - m.b - (y - yd[0]) / (yd[1] - yd[0]) * (H - m.t - m.b);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "TCR versus damage rate" }, host);
      const ax = el("g", { class: "axis" }, svg);
      [0, .25, .5, .75, 1].forEach((v) => { el("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, ax); el("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end" }, ax).textContent = pct(v); });
      [.4, .5, .6, .7, .8, .9, 1].forEach((v) => el("text", { x: sx(v), y: H - m.b + 18, "text-anchor": "middle" }, ax).textContent = v.toFixed(1));
      el("text", { class: "axis-title", x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = "mean token change rate (TCR)";
      el("text", { class: "axis-title", x: 12, y: (H - m.b) / 2, transform: `rotate(-90 12 ${(H - m.b) / 2})`, "text-anchor": "middle" }, svg).textContent = "damage rate";
      const col = { dac: css("--series-1"), mimi: css("--series-2"), speechtokenizer: css("--series-3") }, surf = css("--surface-1");
      const shape = (c, x, y, g) => c === "dac" ? el("circle", { cx: x, cy: y, r: 6.5 }, g) : c === "mimi" ? el("rect", { x: x - 6, y: y - 6, width: 12, height: 12, rx: 2 }, g) : el("path", { d: `M${x} ${y - 7.5}L${x + 7} ${y + 5.5}L${x - 7} ${y + 5.5}Z` }, g);
      S.scatter.forEach((p) => {
        const g = el("g", { fill: col[p.codec], stroke: surf, "stroke-width": 2 }, svg);
        const s = shape(p.codec, sx(p.tcr), sy(p.damage), g);
        s.style.cursor = "pointer";
        s.addEventListener("mousemove", (evt) => showTip(evt, `<div class="tt">${CODEC_NAMES[p.codec]} · ${COND_LABEL(p.condition)}</div><div class="tr"><span>mean TCR</span><b>${fmt(p.tcr)}</b></div><div class="tr"><span>damage rate</span><b>${pct(p.damage)}</b></div>`));
        s.addEventListener("mouseleave", hideTip);
      });
      const sh = { dac: (c) => `<svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="${c}"/></svg>`, mimi: (c) => `<svg width="12" height="12"><rect x="1" y="1" width="10" height="10" rx="2" fill="${c}"/></svg>`, speechtokenizer: (c) => `<svg width="12" height="12"><path d="M6 0.5L11.5 11H0.5Z" fill="${c}"/></svg>` };
      legend($("#scatter-legend"), Object.keys(col).map((c) => ({ name: CODEC_NAMES[c], shape: sh[c](col[c]) })));
      table($("#scatter-table"), ["Tokenizer", "Condition", "Mean TCR", "Damage rate"], S.scatter.map((p) => [CODEC_NAMES[p.codec], COND_LABEL(p.condition), fmt(p.tcr), pct(p.damage)]));
    }
    function drawAll() { drawCurves(); drawStage(); drawScatter(); }
    drawAll();
    redraws.push(drawAll);
  }
})();
