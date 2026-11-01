/* app.js -- UI only. All simulation / filtering lives in engine.js */
const { P, NP, createEpisode } = Engine;
const C = { tw: "#8796a8", be: "#35d0b5", or: "#f0b24a", sh: "#ff6a5e", grid: "#1b2632", ax: "#6c7b8b" };
const $ = (id) => document.getElementById(id);
let E, timer = null, seed = 1000 + ((Math.random() * 9000) | 0);

function frame(cv, { y0, y1, fmt, log, ticks }) {
  const d = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * d) || cv.height !== Math.round(h * d)) { cv.width = Math.round(w * d); cv.height = Math.round(h * d); }
  const c = cv.getContext("2d"); c.setTransform(d, 0, 0, d, 0, 0); c.clearRect(0, 0, w, h);
  const L = 50, R = 10, T = 8, B = 20, X = (i) => L + (i / P.N) * (w - L - R);
  const Y = (v) => T + (1 - (log ? (Math.log(v) - Math.log(y0)) / (Math.log(y1) - Math.log(y0)) : (v - y0) / (y1 - y0))) * (h - T - B);
  c.font = "10.5px ui-monospace,Consolas,monospace"; c.textBaseline = "middle";
  (ticks || [0, 1, 2, 3, 4].map((i) => y0 + ((y1 - y0) * i) / 4)).forEach((v) => {
    c.strokeStyle = C.grid; c.lineWidth = 1; c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke();
    c.fillStyle = C.ax; c.textAlign = "right"; c.fillText(fmt(v), L - 6, Y(v)); });
  c.textAlign = "center"; [0, 10, 20, 30, 40, 50].forEach((i) => { c.fillStyle = C.ax; c.fillText(i, X(i), h - 8); });
  E.shocks.forEach((s) => { c.strokeStyle = C.sh; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(X(s.t), T); c.lineTo(X(s.t), h - B); c.stroke(); c.setLineDash([]);
    c.fillStyle = C.sh; c.textAlign = "left"; c.fillText("⚡×" + s.m, X(s.t) + 4, T + 8); });
  return { c, X, Y };
}
function line(F, arr, col, wd = 1.6) {
  if (!arr.length) return; const { c, X, Y } = F; c.strokeStyle = col; c.lineWidth = wd; c.lineJoin = "round"; c.beginPath();
  arr.forEach((v, i) => (i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)))); c.stroke();
}
const rng = (arrs, pad = 0.08) => { const f = arrs.flat(); let a = Math.min(...f), b = Math.max(...f); if (a === b) { a -= 1; b += 1; } const p = (b - a) * pad; return [a - p, b + p]; };

function render() {
  const A = E.ag, H = E.H, cl = H.cloud;
  { // particle cloud (log axis): prior guesses, weighted posterior, truth, estimate
    const F = frame($("c-cloud"), { y0: 6e-5, y1: 1e-3, log: true, ticks: [1e-4, 2e-4, 5e-4, 1e-3], fmt: (v) => (v >= 1e-3 ? "1e-3" : (v * 1e4).toFixed(0) + "e-4") });
    const stepW = (F.X(1) - F.X(0)) * 0.8;
    cl.forEach((b, t) => {
      F.c.fillStyle = "rgba(150,165,182,.42)"; b.prior.forEach((v, i) => F.c.fillRect(F.X(t) + (((i * 0.618) % 1) - 0.5) * stepW, F.Y(v), 2.2, 2.2));
      F.c.fillStyle = "rgba(53,208,181,.55)"; b.post.forEach((v, i) => { F.c.beginPath(); F.c.arc(F.X(t) + (((i * 0.382 + 0.1) % 1) - 0.5) * stepW, F.Y(v), 1.7, 0, 6.3); F.c.fill(); });
    });
    line(F, H.a, C.or, 1.8); line(F, H.ah, C.be, 1.4);
  }
  { const [a, b] = rng([A.twap.S_h, A.belief.S_h, A.oracle.S_h]); const F = frame($("c-price"), { y0: a, y1: b, fmt: (v) => v.toFixed(2) });
    line(F, A.twap.S_h, C.tw); line(F, A.oracle.S_h, C.or, 1.2); line(F, A.belief.S_h, C.be); }
  { const F = frame($("c-inv"), { y0: 0, y1: 100, fmt: (v) => v + "%" }); const pc = (x) => x.map((q) => (q / P.q0) * 100);
    line(F, pc(A.twap.q_h), C.tw); line(F, pc(A.oracle.q_h), C.or, 1.2); line(F, pc(A.belief.q_h), C.be); }
  { const mx = Math.max(800, ...A.belief.v_h, ...A.oracle.v_h) * 1.08; const F = frame($("c-rate"), { y0: 0, y1: mx, fmt: (v) => Math.round(v) });
    line(F, A.twap.v_h, C.tw); line(F, A.oracle.v_h, C.or, 1.2); line(F, A.belief.v_h, C.be); }
  { const s = H.sd.map((v) => v * 1e6); const F = frame($("c-std"), { y0: 0, y1: Math.max(2, ...s) * 1.12, fmt: (v) => v.toFixed(1) }); line(F, s, C.be); }
  // KPI cards
  for (const k of ["twap", "belief", "oracle"]) {
    const a = A[k]; $("is-" + k).textContent = "$" + Math.round(a.cost).toLocaleString();
    $("s-" + k).textContent = `${Math.round(P.q0 - a.q).toLocaleString()} sold · ${((a.q / P.q0) * 100).toFixed(0)}% left`;
    if (k !== "twap") { const d = a.cost - A.twap.cost, el = $("d-" + k), box = el.parentElement;
      if (E.done) { if (Math.abs(d) < 1) { el.textContent = "≈ same as TWAP · FINAL"; box.classList.remove("win", "lose"); } else { el.textContent = (d < 0 ? "▼ " : "▲ ") + "$" + Math.abs(Math.round(d)).toLocaleString() + " vs TWAP · FINAL"; box.classList.toggle("win", d < 0); box.classList.toggle("lose", d > 0); } }
      else { el.textContent = "in progress · unequal shares sold"; box.classList.remove("win", "lose"); } }
    $("is-" + k).parentElement.classList.toggle("final", E.done);
  }
  const i = H.ah.length - 1;
  if (i >= 0) { $("f-est").textContent = `${(H.ah[i] * 1e4).toFixed(2)} ± ${(H.sd[i] * 1e4).toFixed(3)}e-4`; $("f-true").textContent = (H.a[i] * 1e4).toFixed(2) + "e-4";
    const e = H.ess[i]; $("f-ess").textContent = Math.round(e).toLocaleString(); const hp = $("f-hp"); const ok = e > 0.1 * NP;
    hp.textContent = ok ? "healthy" : "low ESS"; hp.className = "pill " + (ok ? "ok" : "low");
    hp.title = "Low ESS is expected for a step or two right after a shock; persistent low ESS means the filter is failing."; }
  $("progi").style.width = (E.t / P.N) * 100 + "%"; $("stepv").textContent = `step ${E.t} / ${P.N}`;
}
function tick() { E.step(); render(); if (E.done) { stop(); $("run").textContent = "✓ Finished"; $("run").disabled = true; $("shock").disabled = true; } }
function stop() { clearInterval(timer); timer = null; if (E && !E.done) $("run").textContent = "▶ Run"; }
function start() { if (timer || E.done) return; $("run").textContent = "❚❚ Pause"; timer = setInterval(tick, +$("speed").value); }
function reset() { stop(); seed = 1000 + ((Math.random() * 9000) | 0); E = createEpisode(seed); $("run").disabled = false; $("shock").disabled = false; $("run").textContent = "▶ Run"; render(); }
$("run").onclick = () => (timer ? stop() : start());
$("reset").onclick = reset;
$("speed").onchange = () => { if (timer) { stop(); start(); } };
$("mult").oninput = (e) => ($("multv").textContent = "×" + e.target.value);
$("shock").onclick = () => { E.shock(+$("mult").value); if (!timer && !E.done) tick(); };
addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); $("run").click(); } if (e.key === "s" || e.key === "S") $("shock").click(); });
addEventListener("resize", () => render());
reset();
