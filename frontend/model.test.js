/* model.test.js -- checks for model.js. NOT RUN by its author. Usage (from the folder holding model.js):  node model.test.js */
"use strict";
const assert = require("assert");
const M = require("./model.js");

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log("[PASS] " + name); }
  catch (e) { fail++; console.log("[FAIL] " + name + "\n       " + e.message); }
}

function mk(t, o) {
  o = o || {};
  const N = o.N || 5, pol = {};
  for (const n of (o.pols || ["twap"]))
    pol[n] = { q: 100 - 10 * t, v: 10, S: 10, S_exec: 9.9, cum_cost: o.cost && o.cost[n] != null ? o.cost[n] : t + 1 };
  const r = {
    schema_version: "1", episode_id: o.ep || "e1", seed: 1, t, N, done: !!o.done,
    config: { obs_kind: "gaussian", n_regimes: 2, regime_names: ["calm", "stressed"], q0: 100, S0: 10 },
    truth: { S: 10, kappa: 1e-5, alpha: 1e-4, regime: 0 },
    observation: { spread: 0.05, volume: 5000, realvol: 0.02 },
    belief: { t, regime_probs: [0.7, 0.3], alpha_mean: 1e-4, alpha_std: 1e-5, kappa_mean: 1e-5, kappa_std: 1e-6, ess: 100 },
    policies: pol, events: o.events || []
  };
  if (o.particles) r.particles = o.particles;
  return r;
}

test("valid response passes validateStep", () => assert.deepStrictEqual(M.validateStep(mk(0)), []));

test("missing key is reported", () => {
  const r = mk(0); delete r.belief;
  assert.ok(M.validateStep(r).some((x) => x.includes("belief")));
});

test("len(regime_probs) != n_regimes is reported", () => {
  const r = mk(0); r.belief.regime_probs = [1.0];
  assert.ok(M.validateStep(r).some((x) => x.includes("n_regimes")));
});

test("regime_probs not summing to 1 is reported", () => {
  const r = mk(0); r.belief.regime_probs = [0.7, 0.7];
  assert.ok(M.validateStep(r).some((x) => x.includes("sum to 1")));
});

test("particle arrays of different length are reported", () => {
  const r = mk(0, { particles: { regimes: [{ alpha: [1, 2], kappa: [1], w: [1, 1] }] } });
  assert.ok(M.validateStep(r).some((x) => x.includes("differ in length")));
});

test("apply never throws on junk input", () => {
  const st = M.createStore();
  for (const junk of [null, undefined, "abc", 5, [], {}, { t: "x" }]) {
    const res = st.apply(junk);
    assert.strictEqual(res.added, false);
  }
  assert.strictEqual(st.view.steps.length, 0);
});

test("duplicate step is ignored, order preserved", () => {
  const st = M.createStore();
  assert.strictEqual(st.apply(mk(0)).added, true);
  assert.strictEqual(st.apply(mk(1)).added, true);
  const dup = st.apply(mk(1));
  assert.strictEqual(dup.added, false); assert.strictEqual(dup.duplicate, true);
  assert.deepStrictEqual(st.view.H.t, [0, 1]);
});

test("a different episode_id is rejected without mutating the store", () => {
  const st = M.createStore();
  st.apply(mk(0, { ep: "e1" }));
  const r = st.apply(mk(1, { ep: "e2" }));
  assert.strictEqual(r.added, false); assert.strictEqual(r.episodeMismatch, true);
  assert.strictEqual(st.view.steps.length, 1);
  st.reset({ episode_id: "e2", config: mk(0).config, N: 5 });
  assert.strictEqual(st.apply(mk(0, { ep: "e2" })).added, true);
});

test("step gap is recorded as a warning but the step is kept", () => {
  const st = M.createStore();
  st.apply(mk(0));
  const r = st.apply(mk(2));
  assert.strictEqual(r.added, true); assert.strictEqual(r.gap, true);
  assert.strictEqual(st.view.warnings.length, 1);
});

test("a policy appearing late is back-filled so all arrays stay aligned", () => {
  const st = M.createStore();
  st.apply(mk(0, { pols: ["twap"] }));
  st.apply(mk(1, { pols: ["twap", "late"] }));
  st.apply(mk(2, { pols: ["twap", "late"] }));
  const n = st.view.H.t.length;
  for (const nm of st.names()) for (const k of ["q", "v", "S", "Sx", "cost", "mult", "action"])
    assert.strictEqual(st.view.P[nm][k].length, n, nm + "." + k);
  assert.ok(isNaN(st.view.P.late.v[0]));
  assert.strictEqual(st.view.P.late.v[1], 10);
});

test("a policy that disappears is padded with NaN", () => {
  const st = M.createStore();
  st.apply(mk(0, { pols: ["twap", "x"] }));
  st.apply(mk(1, { pols: ["twap"] }));
  assert.strictEqual(st.view.P.x.v.length, 2);
  assert.ok(isNaN(st.view.P.x.v[1]));
});

test("deltaVsTwap is null while running and exact at the end", () => {
  const st = M.createStore();
  st.apply(mk(0, { pols: ["twap", "ens_a"], cost: { twap: 5, ens_a: 4 } }));
  assert.strictEqual(st.deltaVsTwap("ens_a"), null);
  st.apply(mk(1, { pols: ["twap", "ens_a"], cost: { twap: 100, ens_a: 90 }, done: true }));
  assert.strictEqual(st.deltaVsTwap("ens_a"), -10);
});

test("costPerShare = last cum_cost / sum of v; NaN before any sale", () => {
  const st = M.createStore();
  assert.ok(isNaN(st.costPerShare("twap")));
  st.apply(mk(0)); st.apply(mk(1));                    // cum_cost 1 then 2; v = 10 each
  assert.strictEqual(st.soldOf("twap"), 20);
  assert.ok(Math.abs(st.costPerShare("twap") - 0.1) < 1e-12);
});

test("events are de-duplicated", () => {
  const st = M.createStore();
  const ev = { type: "shock", t: 0, mult: 3 };
  st.apply(mk(0, { events: [ev] }));
  st.apply(mk(1, { events: [ev] }));
  assert.strictEqual(st.view.events.length, 1);
});

test("normalizeCloud scales weights so the largest is 1; null-safe", () => {
  const c = M.normalizeCloud({ t: 3, n_per_regime: 3, regimes: [{ regime: 0, prob: 0.5, alpha: [1, 2, 3], kappa: [1, 1, 1], w: [2, 4, 1] }] });
  assert.deepStrictEqual(c.regimes[0].wRel, [0.5, 1, 0.25]);
  assert.strictEqual(M.normalizeCloud(null), null);
  const z = M.normalizeCloud({ regimes: [{ alpha: [1], kappa: [1], w: [0] }] });
  assert.deepStrictEqual(z.regimes[0].wRel, [0]);
});

test("fmtUSD sign and non-finite handling", () => {
  assert.strictEqual(M.fmtUSD(1234.4), "$1,234");
  assert.strictEqual(M.fmtUSD(-1234.4), "-$1,234");
  assert.strictEqual(M.fmtUSD(NaN), "—");
});

test("esc handles null/undefined and escapes HTML", () => {
  assert.strictEqual(M.esc(null), ""); assert.strictEqual(M.esc(undefined), "");
  assert.strictEqual(M.esc('<a href="x">&</a>'), "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;");
});

test("defaultPlotSet returns 1..5 known names, TWAP first when present", () => {
  const names = ["ens_filt_full_bonus", "twap", "ce_filtered", "heuristic_belief", "ens_oracle", "bl1_x", "other"];
  const s = M.defaultPlotSet(names);
  assert.ok(s.length >= 1 && s.length <= 5);
  assert.strictEqual(s[0], "twap");
  assert.ok(s.every((n) => names.includes(n)));
  assert.strictEqual(new Set(s).size, s.length);
});

test("classifyPolicy", () => {
  assert.strictEqual(M.classifyPolicy("twap").group, "baseline");
  assert.strictEqual(M.classifyPolicy("ens_oracle").usesTruth, true);
  assert.strictEqual(M.classifyPolicy("ens_filt_full").group, "ensemble");
  assert.strictEqual(M.classifyPolicy("ce_filtered").usesTruth, false);
});

console.log("\n" + pass + " passed, " + fail + " failed");
process.exit(fail ? 1 : 0);