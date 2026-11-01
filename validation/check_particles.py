"""
check_particles.py -- Stage 4C amendment: opt-in particle cloud in the step response. NOT RUN by its author.
Run from EDI/validation (venv active) AFTER `python apply_particle_patch.py` in EDI/backend:
    python check_particles.py                                  (500 particles per regime; about a minute)

  Q1 no side effects: with particle_sample = 200 the whole episode (belief, v paths, IS; shock injected at step 5) equals the
     particle_sample = 0 episode exactly. The sampling draws no filter random numbers.
  Q2 structure: per step, K regimes x 200 finite, positive alpha/kappa, w >= 0, n_total = 500, t matches, response passes
     validate_step_response and json.dumps(allow_nan=False).
  Q3 exactness: with particle_sample = n (the whole cloud) the weighted mean AND std of alpha and kappa and the mean weight equal the
     belief's per-regime values (so the exposed cloud IS the posterior the belief was computed from, pre-resample), and prob equals regime_probs.
  Q4 determinism: two runs give identical samples; the sample at step t does not depend on earlier particle_sample draws.
  Q5 off by default: no "particles" key when particle_sample = 0.
  Q6 errors: particle_sample > imm_particles or < 0 rejected.
  Q7 HTTP (needs fastapi + httpx; skipped otherwise): "particle_sample" accepted, absent by default, 422 above the filter size.
Also prints the JSON size of one step with 200 particles per regime.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ["IMM_PARTICLES"] = "500"
import numpy as np

import interfaces as I
from live_session import LiveEpisode

SEED, NPART, POL = 1_000_000, 500, ("twap", "ce_filtered", "heuristic_belief")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def run(sample, shock_at=None, seed=SEED):
    ep = LiveEpisode(seed, imm_particles=NPART, policies=POL, particle_sample=sample)
    resp = []
    while not ep.done:
        resp.append(ep.step(inject_shock=3.0) if ep.t == shock_at else ep.step())
    return ep, resp


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


print("\n== no side effects")
e0, r0 = run(0, shock_at=5)
e1, r1 = run(200, shock_at=5)
same_b = all(json.dumps(a["belief"], sort_keys=True) == json.dumps(b["belief"], sort_keys=True) for a, b in zip(r0, r1))
dv = max(abs(a["policies"][n]["v"] - b["policies"][n]["v"]) for a, b in zip(r0, r1) for n in POL)
dI = max(abs(e0.logs()[n].IS - e1.logs()[n].IS) for n in POL)
check("Q1a belief identical with and without sampling", same_b)
check("Q1b v path and IS identical", dv == 0.0 and dI == 0.0, f"(max |dv| {dv:.2e}, max |dIS| {dI:.2e})")
check("Q5 no 'particles' key when particle_sample = 0", all("particles" not in r for r in r0))

print("\n== structure")
K = e1.K
ok_struct, ok_val, ok_json = True, True, True
for r in r1:
    p = r["particles"]
    ok_struct &= (p["t"] == r["t"] and p["n_per_regime"] == 200 and len(p["regimes"]) == K)
    for j, g in enumerate(p["regimes"]):
        ok_struct &= (g["regime"] == j and g["n_total"] == NPART and len(g["alpha"]) == len(g["kappa"]) == len(g["w"]) == 200)
        a_, k_, w_ = np.array(g["alpha"]), np.array(g["kappa"]), np.array(g["w"])
        ok_val &= bool(np.all(np.isfinite(a_)) and np.all(np.isfinite(k_)) and np.all(np.isfinite(w_))
                       and np.all(a_ > 0) and np.all(k_ > 0) and np.all(w_ >= 0))
    try:
        json.dumps(r, allow_nan=False)
    except ValueError:
        ok_json = False
check("Q2a structure (t, counts, K regimes)", ok_struct)
check("Q2b finite, alpha/kappa > 0, w >= 0", ok_val)
check("Q2c responses pass validate_step_response", not [e for r in r1 for e in I.validate_step_response(r)])
check("Q2d JSON-serialisable without NaN", ok_json)
print(f"  (one step with 200 particles per regime: {len(json.dumps(r1[10]))} bytes of JSON; without: {len(json.dumps(r0[10]))})")

print("\n== exactness (sample = whole cloud)")
ef, rf = run(NPART)
worst = dict(am=0.0, asd=0.0, km=0.0, ksd=0.0, w=0.0, p=0.0)
for r in rf:
    b = r["belief"]
    for j, g in enumerate(r["particles"]["regimes"]):
        a_, k_, w_ = np.array(g["alpha"]), np.array(g["kappa"]), np.array(g["w"])
        pr = b["per_regime"][j]
        wn = w_ / w_.sum()
        am, km = float((wn * a_).sum()), float((wn * k_).sum())
        asd, ksd = float(np.sqrt((wn * (a_ - am) ** 2).sum())), float(np.sqrt((wn * (k_ - km) ** 2).sum()))
        worst["am"] = max(worst["am"], abs(am - pr["alpha_mean"]) / pr["alpha_mean"])
        worst["km"] = max(worst["km"], abs(km - pr["kappa_mean"]) / pr["kappa_mean"])
        worst["asd"] = max(worst["asd"], abs(asd - pr["alpha_std"]) / max(pr["alpha_std"], 1e-30))
        worst["ksd"] = max(worst["ksd"], abs(ksd - pr["kappa_std"]) / max(pr["kappa_std"], 1e-30))
        worst["w"] = max(worst["w"], abs(w_.sum() / NPART - 1.0))
        worst["p"] = max(worst["p"], abs(g["prob"] - b["regime_probs"][j]))
check("Q3a weighted mean of alpha, kappa == belief per-regime mean (rel)", max(worst["am"], worst["km"]) < 1e-9,
      f"(alpha {worst['am']:.1e}, kappa {worst['km']:.1e})")
check("Q3b weighted std of alpha, kappa == belief per-regime std (rel)", max(worst["asd"], worst["ksd"]) < 1e-6,
      f"(alpha {worst['asd']:.1e}, kappa {worst['ksd']:.1e})")
check("Q3c mean weight is 1 within each regime", worst["w"] < 1e-9, f"({worst['w']:.1e})")
check("Q3d regime prob == belief.regime_probs", worst["p"] < 1e-12, f"({worst['p']:.1e})")

print("\n== determinism")
_, r2 = run(200, shock_at=5)
check("Q4 identical samples on a repeated run", all(json.dumps(a["particles"], sort_keys=True) == json.dumps(b["particles"], sort_keys=True)
                                                     for a, b in zip(r1, r2)))

print("\n== errors")
check("Q6a particle_sample above the filter size rejected", raises(lambda: LiveEpisode(SEED, imm_particles=NPART, policies=POL, particle_sample=NPART + 1)))
check("Q6b negative particle_sample rejected", raises(lambda: LiveEpisode(SEED, imm_particles=NPART, policies=POL, particle_sample=-1)))

print("\n== HTTP")
try:
    from fastapi.testclient import TestClient
    import api_server
    c = TestClient(api_server.app)
    body = {"seed": SEED, "policies": ["twap", "ce_filtered"]}
    d = c.post("/api/episode", json={**body, "particle_sample": 50}).json()
    s = c.post(f"/api/episode/{d['episode_id']}/step", json={})
    pj = s.json().get("particles")
    check("Q7a particle_sample=50 over HTTP gives 50 per regime", s.status_code == 200 and pj is not None and pj["n_per_regime"] == 50
          and all(len(g["alpha"]) == 50 for g in pj["regimes"]))
    d0 = c.post("/api/episode", json=body).json()
    s0 = c.post(f"/api/episode/{d0['episode_id']}/step", json={})
    check("Q7b no particles by default", s0.status_code == 200 and "particles" not in s0.json())
    check("Q7c above the filter size -> 422", c.post("/api/episode", json={**body, "particle_sample": NPART + 1}).status_code == 422)
    check("Q7d above 2000 -> 422", c.post("/api/episode", json={**body, "particle_sample": 2001}).status_code == 422)
except ImportError as ex:
    print(f"  [SKIP] HTTP checks need fastapi and httpx ({ex})")

print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)