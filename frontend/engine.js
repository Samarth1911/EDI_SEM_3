/* engine.js -- JS port of the Python reference (simulator.py, observation_model.py,
   particle_filter.py, heuristic_policy.py). Python is the reference; parity_check.js
   compares this against the Python experiment numbers. */
(function (g) {
  const P = { S0: 10, q0: 20000, N: 50, sigma: 0.02, kT: 1.25e-5, lk: 3, sk: 1.875e-4,
              aT: 1.5e-4, la: 3, sa: 4.5928e-4, rho: 0.9 };
  const O = { kr: 1.25e-5, ar: 1.5e-4, sb: 0.05, sn: 0.0018, vb: 5000, vf: 50, vn: 120, rb: 0.02, rn: 0.0008 };
  const NP = 5000, tau = 1 / P.N, NV = 80;   // particles, dt, particles drawn for display

  function rng(seed) {                       // mulberry32 + Marsaglia polar gaussian
    let a = seed >>> 0, spare = null;
    const u = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const gs = () => { if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      const m = Math.sqrt(-2 * Math.log(r) / r); spare = y * m; return x * m; };
    return { u, g: gs };
  }
  const cir = (x, th, l, s, dB) => { const xp = Math.max(x, 0); return Math.max(x + l * (th - xp) * tau + s * Math.sqrt(xp) * dB, 1e-10); };
  const observe = (k, a, R) => { const ar = a / O.ar, kr = k / O.kr;
    return { sp: Math.max(O.sb * ar + R.g() * O.sn, 1e-5),
             vo: Math.max(O.vb / Math.max(ar, 1e-3) + R.g() * O.vn, O.vf),
             rv: Math.max(O.rb * kr + R.g() * O.rn, 1e-5) }; };
  function sys(w, m, R) {                    // systematic resampling
    const out = new Int32Array(m), u0 = R.u(); let c = w[0], j = 0;
    for (let i = 0; i < m; i++) { const p = (u0 + i) / m; while (p > c && j < w.length - 1) { j++; c += w[j]; } out[i] = j; }
    return out;
  }

  class PF {
    constructor(R) {
      this.R = R; const n = NP; this.k = new Float64Array(n); this.a = new Float64Array(n); this.w = new Float64Array(n).fill(1 / n);
      for (let i = 0; i < n; i++) { this.k[i] = Math.max(P.kT + R.g() * 4.6875e-7, 1e-6); this.a[i] = Math.max(P.aT + R.g() * 3.75e-6, 1e-6); }
      this.jk = 0.15 * Math.sqrt(P.sk ** 2 * P.kT / (2 * P.lk)); this.ja = 0.15 * Math.sqrt(P.sa ** 2 * P.aT / (2 * P.la));
    }
    step(o) {
      const n = NP, R = this.R, w = this.w; let k = this.k, a = this.a;
      const prior = new Float32Array(NV); for (let i = 0; i < NV; i++) prior[i] = a[(R.u() * n) | 0];
      const lg = new Float64Array(n); let mx = -Infinity;
      for (let i = 0; i < n; i++) {                                       // reweight (3 Gaussian proxy channels)
        const ar = a[i] / O.ar, kr = k[i] / O.kr;
        const l = -0.5 * (((o.sp - O.sb * ar) / O.sn) ** 2 + ((o.vo - Math.max(O.vb / Math.max(ar, 1e-3), O.vf)) / O.vn) ** 2 + ((o.rv - O.rb * kr) / O.rn) ** 2);
        lg[i] = l; if (l > mx) mx = l;
      }
      let s = 0; for (let i = 0; i < n; i++) { w[i] *= Math.exp(lg[i] - mx); s += w[i]; }
      if (!(s > 0)) w.fill(1 / n); else for (let i = 0; i < n; i++) w[i] /= s;
      let ma = 0, mk = 0, s2 = 0; for (let i = 0; i < n; i++) { ma += w[i] * a[i]; mk += w[i] * k[i]; s2 += w[i] * w[i]; }
      let va = 0; for (let i = 0; i < n; i++) va += w[i] * (a[i] - ma) ** 2;
      const ess = 1 / s2, di = sys(w, NV, R), post = new Float32Array(NV); for (let i = 0; i < NV; i++) post[i] = a[di[i]];
      const out = { aMean: ma, aStd: Math.sqrt(Math.max(va, 0)), kMean: mk, ess, prior, post };   // posterior recorded PRE-resample
      if (ess < 0.5 * n) {                                                // resample + roughen
        const idx = sys(w, n, R), k2 = new Float64Array(n), a2 = new Float64Array(n);
        for (let i = 0; i < n; i++) { k2[i] = Math.max(k[idx[i]] + R.g() * this.jk, 1e-8); a2[i] = Math.max(a[idx[i]] + R.g() * this.ja, 1e-8); }
        this.k = k = k2; this.a = a = a2; w.fill(1 / n);
      }
      const st = Math.sqrt(tau), c2 = Math.sqrt(1 - P.rho ** 2), sk = P.sk * 1.5, sa = P.sa * 1.5, lm = Math.log(0.5), lr = Math.log(12);
      for (let i = 0; i < n; i++) {                                       // propagate: CIR diffusion + alpha jump component
        const z1 = R.g() * st, z2 = R.g() * st, d2 = P.rho * z1 + c2 * z2, kp = Math.max(k[i], 0), ap = Math.max(a[i], 0);
        k[i] = Math.max(k[i] + P.lk * (P.kT - kp) * tau + sk * Math.sqrt(kp) * z1, 1e-8);
        let an = Math.max(a[i] + P.la * (P.aT - ap) * tau + sa * Math.sqrt(ap) * d2, 1e-8);
        if (R.u() < 0.08) an *= Math.exp(lm + R.u() * lr);                // p=0.08, m~LogUniform(0.5,6)
        a[i] = an;
      }
      return out;
    }
  }

  // belief heuristic: gamma=1, taper=1, s in [0.2,2], z_unc=0 (uncertainty term did not help -- see README)
  function heur(t, q, ah) {
    if (q <= 0) return 0; if (t >= P.N - 1) return q;
    const s = Math.min(2, Math.max(0.2, P.aT / Math.max(ah, 1e-12))), tp = (P.N - 1 - t) / P.N;
    return Math.min((q / (P.N - t)) * (1 + (s - 1) * tp), q);
  }

  function createEpisode(seed) {
    const Rm = rng(seed), Ro = rng(seed + 10000), pf = new PF(rng(seed + 20000));
    const mk = () => ({ S: P.S0, q: P.q0, cost: 0, S_h: [P.S0], q_h: [P.q0], v_h: [], c_h: [0] });
    const E = { t: 0, k: P.kT, a: P.aT, pending: null, shocks: [], done: false,
                H: { a: [], ah: [], sd: [], ess: [], cloud: [] }, ag: { twap: mk(), belief: mk(), oracle: mk() } };
    E.shock = (m) => { if (!E.done) E.pending = m; };
    E.step = () => {
      if (E.done) return; const t = E.t;
      if (E.pending) { E.a *= E.pending; E.shocks.push({ t, m: E.pending }); E.pending = null; }
      const b = pf.step(observe(E.k, E.a, Ro));                           // belief for time t, from noisy proxies only
      E.H.a.push(E.a); E.H.ah.push(b.aMean); E.H.sd.push(b.aStd); E.H.ess.push(b.ess); E.H.cloud.push(b);
      const dW = Rm.g() * Math.sqrt(tau);
      for (const name of ["twap", "belief", "oracle"]) {
        const A = E.ag[name];
        let v = name === "twap" ? P.q0 / P.N : name === "belief" ? heur(t, A.q, b.aMean) : heur(t, A.q, E.a);
        v = Math.min(Math.max(v, 0), A.q);
        const Sx = A.S - E.a * v; A.cost += P.S0 * v - Sx * v; A.S += -E.k * v + P.sigma * dW; A.q -= v;
        A.S_h.push(A.S); A.q_h.push(A.q); A.v_h.push(v); A.c_h.push(A.cost);
      }
      const st = Math.sqrt(tau), z1 = Rm.g() * st, z2 = Rm.g() * st;      // exogenous impact paths (independent of trading)
      E.k = cir(E.k, P.kT, P.lk, P.sk, z1); E.a = cir(E.a, P.aT, P.la, P.sa, P.rho * z1 + Math.sqrt(1 - P.rho ** 2) * z2);
      if (++E.t >= P.N) E.done = true;
    };
    return E;
  }
  g.Engine = { P, O, NP, createEpisode };
  if (typeof module !== "undefined") module.exports = g.Engine;
})(typeof window !== "undefined" ? window : globalThis);
