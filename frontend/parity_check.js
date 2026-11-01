// node frontend/parity_check.js  -- compares the JS port to the Python experiment (validation/experiments/run_policy_comparison.py)
// Python (300 eps, shock step U[8,40], x U[2,5]): TWAP 4431 | belief-TWAP paired -86 | calm TWAP 3659, paired ~0
const E = require("./engine.js"); const n = 60; let ok = true;
function run(shocked) {
  const r = { twap: [], belief: [], oracle: [] };
  for (let i = 0; i < n; i++) {
    const ep = E.createEpisode(900 + i), rs = (i * 2654435761 % 4294967296) / 4294967296, step = 8 + Math.floor(rs * 33), m = 2 + ((i * 40503 % 1000) / 1000) * 3;
    for (let t = 0; t < E.P.N; t++) { if (shocked && t === step) ep.shock(m); ep.step(); }
    for (const k in r) r[k].push(ep.ag[k].cost);
  } return r;
}
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
for (const [name, sh, tw, bd] of [["shock", true, 4431, -86], ["calm", false, 3659, 0]]) {
  const r = run(sh), dB = mean(r.belief.map((v, i) => v - r.twap[i])), dO = mean(r.oracle.map((v, i) => v - r.twap[i]));
  const pass = Math.abs(mean(r.twap) - tw) < 200 && Math.abs(dB - bd) < 25 && Math.abs(dB - dO) < 10; ok = ok && pass;
  console.log(`${name}: TWAP ${mean(r.twap).toFixed(0)} (py ~${tw}) | belief-TWAP ${dB.toFixed(1)} (py ~${bd}) | oracle-TWAP ${dO.toFixed(1)}  ${pass ? "PASS" : "FAIL"}`);
}
console.log(ok ? "PARITY OK" : "PARITY FAILED"); process.exit(ok ? 0 : 1);
