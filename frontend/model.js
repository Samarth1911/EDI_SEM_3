/* model.js -- Stage 5 data layer (no DOM). REVISED; NOT RUN against the real backend by its author.
 * Turns the C5 step responses (interfaces.STEP_RESPONSE_SCHEMA) into per-episode history arrays.
 * Everything here is defensive: the backend's optional keys (particles, per-policy action/mult, events) are read if present.
 *
 * REVISION NOTES (vs the first version)
 *  - validateStep: now also checks len(belief.regime_probs) == config.n_regimes (the Python validator does; the port missed it)
 *    and that regime_probs are finite numbers.
 *  - apply(): no longer throws on non-object input; ignores responses whose t is not an integer; rejects a response from a
 *    DIFFERENT episode_id instead of silently dropping it as a "duplicate" (returns episodeMismatch:true; caller should reset());
 *    records a warning when steps are skipped (t > last t + 1).
 *  - A policy that first appears after step 0 is back-filled with NaN so every per-policy array has the same length as the
 *    history arrays (before, its arrays were shorter and every index was misaligned).
 *  - fmtUSD prints negatives as -$1,234 (was $-1,234); esc(null/undefined) returns "".
 *  - ADDED: soldOf(name) and costPerShare(name) -- a running average shortfall per share sold, because cum_cost alone is
 *    NOT comparable across policies mid-episode (they have sold different amounts). Also not a final-cost comparison.
 * NOT in the Stage 0 schema (the backend must add it, otherwise the particle cloud stays empty): the optional `particles`
 * block read by normalizeCloud, and the optional per-policy `mult` / `action` keys.
 */
(function (root) {
  "use strict";

  const REQUIRED = ["schema_version", "episode_id", "seed", "t", "N", "done", "config", "truth",
    "observation", "belief", "policies", "events"];
  const isNum = (x) => typeof x === "number" && isFinite(x);

  /** JS port of interfaces.validate_step_response (+ the optional particles block). Returns a list of problems. */
  function validateStep(r) {
    if (!r || typeof r !== "object") return ["response is not an object"];
    const e = [];
    for (const k of REQUIRED) if (!(k in r)) e.push("missing key: " + k);
    if (e.length) return e;
    if (!Number.isInteger(r.t) || r.t < 0) e.push("t must be a non-negative integer");
    if (!Number.isInteger(r.N) || r.N < 1) e.push("N must be a positive integer");
    if (typeof r.done !== "boolean") e.push("done must be boolean");
    for (const k of ["obs_kind", "n_regimes", "regime_names", "q0", "S0"]) if (!(k in (r.config || {}))) e.push("config." + k);
    for (const k of ["S", "kappa", "alpha", "regime"]) if (!(k in (r.truth || {}))) e.push("truth." + k);
    for (const k of ["spread", "volume", "realvol"]) if (!isNum((r.observation || {})[k])) e.push("observation." + k + " is not a finite number");
    const b = r.belief || {};
    for (const k of ["t", "regime_probs", "alpha_mean", "alpha_std", "kappa_mean", "kappa_std", "ess"]) if (!(k in b)) e.push("belief." + k);
    if (Array.isArray(b.regime_probs)) {
      if (!b.regime_probs.every(isNum)) e.push("belief.regime_probs contains a non-finite value");
      else {
        const s = b.regime_probs.reduce((a, x) => a + x, 0);
        if (Math.abs(s - 1) > 1e-4) e.push("belief.regime_probs does not sum to 1 (" + s + ")");
      }
      const K = (r.config || {}).n_regimes;
      if (K != null && b.regime_probs.length !== K) e.push("len(belief.regime_probs) != config.n_regimes");
    }
    for (const [n, p] of Object.entries(r.policies || {}))
      for (const k of ["q", "v", "S", "S_exec", "cum_cost"]) if (!isNum((p || {})[k])) e.push("policies." + n + "." + k + " is not a finite number");
    if (!Array.isArray(r.events)) e.push("events must be an array");
    else r.events.forEach((ev, i) => { if (!ev || typeof ev.type !== "string" || !Number.isInteger(ev.t)) e.push("events[" + i + "] needs type and integer t"); });
    if ("particles" in r && r.particles != null) {
      const p = r.particles;
      if (!Array.isArray(p.regimes)) e.push("particles.regimes must be an array");
      else p.regimes.forEach((g, j) => {
        if (!g || !Array.isArray(g.alpha) || !Array.isArray(g.kappa) || !Array.isArray(g.w)) e.push("particles.regimes[" + j + "] needs alpha, kappa, w arrays");
        else if (g.alpha.length !== g.kappa.length || g.alpha.length !== g.w.length) e.push("particles.regimes[" + j + "] arrays differ in length");
      });
    }
    return e;
  }

  /** Group / oracle flag from the policy name (heuristic: names come from the backend, see STAGE0_INTERFACES.md 10.3). */
  function classifyPolicy(name) {
    const usesTruth = /oracle/.test(name);
    let group = "benchmark", label = "benchmark";
    if (name === "twap") { group = "baseline"; label = "baseline"; }
    else if (/^ens_/.test(name)) { group = "ensemble"; label = "DDQN ensemble"; }
    else if (/^ddqn_/.test(name)) { group = "agent"; label = "DDQN agent"; }
    return { group, label, usesTruth };
  }
  const GROUP_ORDER = { baseline: 0, benchmark: 1, agent: 2, ensemble: 3 };

  /** Normalise the particle block: per regime numeric arrays and weights relative to the largest weight. */
  function normalizeCloud(p) {
    if (!p || !Array.isArray(p.regimes)) return null;
    return {
      t: p.t, nPer: p.n_per_regime,
      regimes: p.regimes.map((g, j) => {
        const w = (g.w || []).map(Number);
        let wmax = 0;
        for (const x of w) if (x > wmax) wmax = x;
        if (!(wmax > 0)) wmax = 1;
        return { regime: g.regime != null ? g.regime : j, prob: Number(g.prob), nTotal: g.n_total,
          alpha: (g.alpha || []).map(Number), kappa: (g.kappa || []).map(Number), wRel: w.map((x) => x / wmax) };
      })
    };
  }

  function createStore() {
    const s = {
      desc: null, cfg: null, N: 50, t: -1, done: false, episodeId: null,
      steps: [], events: [], errors: [], warnings: [], cloud: [],
      H: null, P: {}, _seen: new Set(), _names: []
    };
    const blankH = () => ({ t: [], S_truth: [], alpha: [], kappa: [], regime: [], spread: [], volume: [], realvol: [],
      a_mean: [], a_std: [], k_mean: [], k_std: [], ess: [], nPart: [], pr: [], per: [] });
    const lastOf = (arr) => (arr.length ? arr[arr.length - 1] : NaN);

    function reset(desc) {
      s.desc = desc || null; s.cfg = (desc && desc.config) || null; s.N = (desc && desc.N) || 50;
      s.episodeId = (desc && desc.episode_id) || null;
      s.t = -1; s.done = false; s.steps = []; s.events = []; s.errors = []; s.warnings = []; s.cloud = [];
      s.H = blankH(); s.P = {}; s._seen = new Set(); s._names = [];
    }
    reset(null);

    function apply(resp) {
      const errors = validateStep(resp);
      if (!resp || typeof resp !== "object") return { added: false, errors };
      if (errors.length && !("t" in resp)) return { added: false, errors };
      if (!Number.isInteger(resp.t)) return { added: false, errors };
      if (s.episodeId == null) s.episodeId = resp.episode_id != null ? resp.episode_id : null;
      else if (resp.episode_id != null && resp.episode_id !== s.episodeId)
        return { added: false, errors, episodeMismatch: true };            // caller should reset() for a new episode
      if (resp.t <= s.t) return { added: false, errors, duplicate: true };   // a retried step: ignore
      const gap = s.t >= 0 && resp.t > s.t + 1;
      if (gap) s.warnings.push({ t: resp.t, warning: "step gap: expected t=" + (s.t + 1) + ", got t=" + resp.t });
      if (errors.length) s.errors.push({ t: resp.t, errors });
      s.cfg = resp.config || s.cfg; s.N = resp.N || s.N; s.done = !!resp.done; s.t = resp.t;
      s.steps.push(resp);
      const n = s.steps.length;                                            // length every history array must have now
      const H = s.H, b = resp.belief || {}, tr = resp.truth || {}, ob = resp.observation || {};
      H.t.push(resp.t); H.S_truth.push(tr.S); H.alpha.push(tr.alpha); H.kappa.push(tr.kappa); H.regime.push(tr.regime);
      H.spread.push(ob.spread); H.volume.push(ob.volume); H.realvol.push(ob.realvol);
      H.a_mean.push(b.alpha_mean); H.a_std.push(b.alpha_std); H.k_mean.push(b.kappa_mean); H.k_std.push(b.kappa_std);
      H.ess.push(b.ess); H.nPart.push(b.n_particles != null ? b.n_particles : null);
      H.pr.push(Array.isArray(b.regime_probs) ? b.regime_probs.slice() : []);
      H.per.push(Array.isArray(b.per_regime) ? b.per_regime.map((g) => Object.assign({}, g)) : []);
      s.cloud.push(normalizeCloud(resp.particles));
      const pol = resp.policies || {};
      for (const name of Object.keys(pol)) {
        if (!s.P[name]) {                                                  // new policy: back-fill the steps it missed
          const nan = () => new Array(n - 1).fill(NaN);
          s.P[name] = { q: nan(), v: nan(), S: nan(), Sx: nan(), cost: nan(), mult: nan(), action: nan() };
          s._names.push(name);
        }
      }
      for (const name of s._names) {
        const p = pol[name], A = s.P[name];
        A.q.push(p ? p.q : NaN); A.v.push(p ? p.v : NaN); A.S.push(p ? p.S : NaN); A.Sx.push(p ? p.S_exec : NaN);
        A.cost.push(p ? p.cum_cost : NaN);
        const m = p && (p.mult != null ? p.mult : p.multiplier);
        A.mult.push(m != null ? Number(m) : NaN); A.action.push(p && p.action != null ? Number(p.action) : NaN);
      }
      for (const ev of resp.events || []) {
        const key = ev.type + "|" + ev.t + "|" + JSON.stringify(ev);
        if (!s._seen.has(key)) { s._seen.add(key); s.events.push(ev); }
      }
      return { added: true, errors, gap };
    }

    const last = () => (s.steps.length ? s.steps[s.steps.length - 1] : null);
    const names = () => s._names.slice();
    /** Cost difference vs TWAP. Only meaningful at the end (cum_cost counts shares sold so far), so null while running. */
    function deltaVsTwap(name) {
      if (!s.done || !s.P.twap || !s.P[name]) return null;
      const a = s.P[name].cost, b = s.P.twap.cost;
      return a[a.length - 1] - b[b.length - 1];
    }
    /** Shares sold so far = sum of the policy's per-step v (independent of whether the backend's q is before/after the trade). */
    function soldOf(name) {
      const A = s.P[name];
      if (!A) return NaN;
      let t = 0;
      for (const x of A.v) if (isNum(x)) t += x;
      return t;
    }
    /** Running average shortfall per share sold, $/share. NOT a final-cost comparison and NOT comparable across policies
     *  that have sold different amounts; it is a live indicator only. NaN before the first sale. */
    function costPerShare(name) {
      const A = s.P[name];
      if (!A || !A.cost.length) return NaN;
      const sold = soldOf(name), c = lastOf(A.cost);
      return sold > 0 && isNum(c) ? c / sold : NaN;
    }
    return { view: s, reset, apply, last, names, deltaVsTwap, soldOf, costPerShare, lastOf };
  }

  /** Default policies to plot: TWAP, then a spread of benchmark / agent / ensemble rows, at most 5. */
  function defaultPlotSet(names) {
    const out = [];
    const add = (n) => { if (n && names.includes(n) && !out.includes(n)) out.push(n); };
    add("twap"); add("ce_filtered"); add("heuristic_belief");
    add(names.find((n) => /^ens_filt_full/.test(n)));
    add(names.find((n) => /^ens_oracle/.test(n)));
    add(names.find((n) => /^ddqn_filt_full/.test(n)));
    for (const n of names) { if (out.length >= 3) break; add(n); }
    return out.slice(0, 5);
  }

  function fmtUSD(v) {
    if (!isNum(v)) return "—";
    const r = Math.round(v);
    return (r < 0 ? "-$" : "$") + Math.abs(r).toLocaleString("en-US");
  }
  function fmtSci(v, d) { return isNum(v) ? v.toExponential(d == null ? 2 : d) : "—"; }
  function esc(x) {
    if (x == null) return "";
    return String(x).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  const api = { validateStep, classifyPolicy, GROUP_ORDER, normalizeCloud, createStore, defaultPlotSet, fmtUSD, fmtSci, esc, isNum };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else { root.EDI = root.EDI || {}; root.EDI.model = api; }
})(typeof self !== "undefined" ? self : this);