"""
check_live_session.py -- Stage 4C checks. NOT RUN by its author. Run from EDI/validation (venv active):
    python check_live_session.py            (500 particles, 2 observation kinds, about a minute or two; no web server needed)

Checks (each prints PASS/FAIL; exit code 1 if any FAIL):
  P  live == batch: for each obs kind, an unmodified live episode reproduces run_stage4_benchmarks (pre-pass + play) for
     EVERY hosted policy: IS and the whole v path equal to 1e-9. Proves the online filter + per-policy generators match the
     cached-belief benchmark path.
  C  contract: every step response passes validate_step_response, belief regime_probs sum to 1, done only at the last step,
     EpisodeLogs pass validate_episode_log, cum_cost at the end equals IS.
  B  oracle barrier: B1 non-oracle policies receive truth=None, oracle ones receive it;
     B2 non-oracle policies receive a belief, oracle ones receive belief=None.
  S  shock: inject_shock=3 at step 10 -> alpha before step 10 unchanged, alpha[10] = 3 x baseline, TWAP IS equals a direct
     simulator run with shock_step=10.
  R  forced regime: {regime 1, duration 5} at step 10 -> regimes 10..14 are 1, earlier steps unchanged, TWAP IS equals a direct
     run with forced_regime_path.
  F  rollback: a policy that raises after earlier policies already committed (on a plain step, a shock step and a forced-regime
     step) leaves the episode untouched; retrying reproduces the unfailed episode (IS and v path to 1e-9).
  E  errors: second shock, out-of-range values and stepping a finished episode are rejected.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np

from live_session import LiveEpisode, DEFAULT_POLICIES, PRESETS
import interfaces as I
from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams, REGIME_RNG_TAG, force_regime_window
from observation_model import ObsParams
from imm_particle_filter import IMMParticleFilter, IMMParams
from run_stage4_benchmarks import prepass, play, make_policies
from heuristic_policy import HeuristicParams

SEED, PART = 1_000_000, 500
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def run_all(ep):
    resp = []
    while not ep.done:
        resp.append(ep.step())
    return resp


def twap_direct(sp, rp, seed, **kw):
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    g = sim.run_stepwise(seed=seed, **kw)
    s = next(g)
    while True:
        v = 0.0 if s["q"] <= 0 else s["q"] / (sp.N - int(s["t"]))
        try:
            s = g.send(v)
        except StopIteration as stop:
            return stop.value


for kind in ("gaussian", "spread_hawkes"):
    print(f"\n== obs_kind = {kind}")
    live = LiveEpisode(SEED, obs_kind=kind, imm_particles=PART)
    # B: wrap act() to record what truth / belief each policy receives
    seen, seen_b = {}, {}
    for p in live.policies:
        orig = p.act
        def wrapped(x, _o=orig, _n=p.name):
            seen.setdefault(_n, []).append(x.truth is not None)
            seen_b.setdefault(_n, []).append(x.belief is not None)
            return _o(x)
        p.act = wrapped
    resp = run_all(live)

    # C
    errs = [e for r in resp for e in I.validate_step_response(r)]
    check("C1 every step response passes validate_step_response", not errs, str(errs[:2]))
    check("C2 t runs 0..N-1 and done only at the end", [r["t"] for r in resp] == list(range(live.N))
          and [r["done"] for r in resp] == [False] * (live.N - 1) + [True])
    lg = live.logs()
    bad = {n: I.validate_episode_log(l) for n, l in lg.items() if I.validate_episode_log(l)}
    check("C3 EpisodeLogs pass validate_episode_log", not bad, str(bad)[:200])
    gap = max(abs(resp[-1]["policies"][n]["cum_cost"] - lg[n].IS) for n in lg)
    check("C4 final cum_cost == IS", gap < 1e-6, f"(max gap {gap:.2e})")

    # B
    ok = all((all(seen[p.name]) if p.uses_oracle_truth else not any(seen[p.name])) for p in live.policies)
    check("B1 oracle policies get truth, others get truth=None", ok, str({p.name: p.uses_oracle_truth for p in live.policies}))
    ok = all((not any(seen_b[p.name])) if p.uses_oracle_truth else all(seen_b[p.name]) for p in live.policies)
    check("B2 non-oracle policies get a belief, oracle ones get belief=None", ok)

    # P: batch reference
    sp, op = SimParams(), ObsParams()
    rp = RegimeParams.default(); rp.init_probs = rp.stationary_distribution()
    imm = IMMParticleFilter(sp, op, rp, IMMParams(n_particles=PART), seed=0)
    ep0, bel = prepass(SEED, sp, rp, op, kind, imm)
    fns = make_policies(sp, rp, HeuristicParams(), {})
    wcfg = live.world
    worst_IS, worst_v = 0.0, 0.0
    for n in DEFAULT_POLICIES:
        ref = play(SEED, fns[n], n, sp, rp, ep0, bel, wcfg)
        worst_IS = max(worst_IS, abs(ref.IS - lg[n].IS))
        worst_v = max(worst_v, max(abs(a.v - b.v) for a, b in zip(ref.steps, lg[n].steps)))
    check("P1 live == batch benchmarks, IS (all hosted policies)", worst_IS < 1e-9, f"(max |dIS| {worst_IS:.2e})")
    check("P2 live == batch benchmarks, v path", worst_v < 1e-9, f"(max |dv| {worst_v:.2e})")

# ---- S / R on the gaussian world --------------------------------------------------------------
print("\n== shock and forced regime (gaussian)")
sp = SimParams(); rp = PRESETS["default"](); rp.init_probs = rp.stationary_distribution()
base = LiveEpisode(SEED, imm_particles=PART); rb = run_all(base)
base_alpha = np.array([r["truth"]["alpha"] for r in rb])
base_reg = np.array([r["truth"]["regime"] for r in rb])

e = LiveEpisode(SEED, imm_particles=PART); rs = []
while not e.done:
    rs.append(e.step(inject_shock=3.0) if e.t == 10 else e.step())
al = np.array([r["truth"]["alpha"] for r in rs])
check("S1 steps before 10 unchanged", np.array_equal(al[:10], base_alpha[:10]))
check("S2 alpha[10] = 3 x baseline", abs(al[10] / base_alpha[10] - 3.0) < 1e-9, f"(ratio {al[10] / base_alpha[10]:.6f})")
check("S3 event reported", any(ev["type"] == "shock_injected" and ev["t"] == 10 for ev in rs[10]["events"]))
d = twap_direct(sp, rp, SEED, shock_step=10, shock_jump_alpha_mult=3.0)
check("S4 live TWAP IS == direct run with shock_step=10", abs(e.logs()["twap"].IS - d.IS) < 1e-9,
      f"(live {e.logs()['twap'].IS:.4f} direct {d.IS:.4f})")

e = LiveEpisode(SEED, imm_particles=PART); rr = []
while not e.done:
    rr.append(e.step(force_regime={"regime": 1, "duration": 5}) if e.t == 10 else e.step())
rg = np.array([r["truth"]["regime"] for r in rr])
check("R1 regimes 10..14 forced to 1", bool(np.all(rg[10:15] == 1)))
check("R2 regimes before 10 unchanged", np.array_equal(rg[:10], base_reg[:10]))
path = RegimeSwitchingSimulator(sp, rp, seed=SEED).sample_regime_path(sp.N, np.random.default_rng([SEED, REGIME_RNG_TAG]))
d = twap_direct(sp, rp, SEED, forced_regime_path=force_regime_window(path, 10, 5, 1))
check("R3 live TWAP IS == direct run with forced path", abs(e.logs()["twap"].IS - d.IS) < 1e-9,
      f"(live {e.logs()['twap'].IS:.4f} direct {d.IS:.4f})")

# ---- F: partial-failure rollback ------------------------------------------------------------
print("\n== partial-failure rollback")
def faulty_run(request):
    e = LiveEpisode(SEED, imm_particles=PART)
    victim = e.policies[-1]                           # last policy: earlier ones have already committed when it raises
    orig, armed = victim.act, {"on": False}
    def flaky(x):
        if armed["on"]:
            armed["on"] = False
            raise RuntimeError("injected fault")
        return orig(x)
    victim.act = flaky
    st = None
    while not e.done:
        if e.t == 2 and st is None:
            armed["on"] = True
            try:
                e.step(**request)
            except RuntimeError:
                pass
            st = (e.t, len(e.obs_rows), len(e.beliefs), [len(h["v"]) for h in e.hist.values()],
                  e.shock_step, e.regime_forced)
        e.step()                                      # the retry (plain step) and the rest of the episode
    return e, st

lb = base.logs()
for label, req in (("plain", {}), ("shock", {"inject_shock": 3.0}),
                   ("regime", {"force_regime": {"regime": 1, "duration": 5}})):
    e, st = faulty_run(req)
    check(f"F1 [{label}] failed step left state untouched", st == (2, 2, 2, [2] * len(e.policies), None, False), str(st))
    lg2 = e.logs()
    dIS = max(abs(lg2[n].IS - lb[n].IS) for n in lb)
    dv = max(max(abs(a.v - b.v) for a, b in zip(lg2[n].steps, lb[n].steps)) for n in lb)
    check(f"F2 [{label}] retry reproduces the unfailed episode", dIS < 1e-9 and dv < 1e-9,
          f"(|dIS| {dIS:.2e} |dv| {dv:.2e})")

# ---- E --------------------------------------------------------------------------------------------
print("\n== error handling")
e = LiveEpisode(SEED, imm_particles=PART)
def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False
check("E1 shock outside [2,5] rejected", raises(lambda: e.step(inject_shock=9.0), ValueError))
check("E2 regime id out of range rejected", raises(lambda: e.step(force_regime={"regime": 7, "duration": 3}), ValueError))
check("E3 duration < 1 rejected", raises(lambda: e.step(force_regime={"regime": 1, "duration": 0}), ValueError))
check("E4 rejected requests did not advance the episode", e.t == 0)
e.step(inject_shock=2.0)
check("E5 second shock rejected", raises(lambda: e.step(inject_shock=3.0), ValueError))
run_all(e)
check("E6 stepping a finished episode rejected", raises(lambda: e.step(), RuntimeError))

print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)