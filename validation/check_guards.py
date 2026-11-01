"""
check_guards.py -- Stage 4C Tier 1 guard checks. NOT RUN by its author. Run from EDI/validation (venv active):
    python check_guards.py                  (500 particles; about a minute or two; no web server needed)

  U  unit: GuardConfig validation, stress-regime resolution, mark-to-market cost, constraint arithmetic.
  G1 breaker: a deliberately aggressive policy (3x TWAP rate every step) is switched to TWAP once its mark-to-market cost
     exceeds TWAP's by X; afterwards it trades exactly q/(N-t); its final IS is lower than the unguarded run's.
  G2 constraint: with stress_prob=0 (always stressed) every guarded action satisfies 0.75 <= v/(q/(N-t)) <= 1.25 except the
     last step, which is full liquidation; every logged constraint is consistent.
  G3 isolation: policies that are not guarded produce the same v path as in an unguarded episode (to 1e-12).
  G4 rollback: a fault injected at EVERY step (after the guarded policy has acted) leaves the guard state untouched; the
     retried episode equals a clean guarded episode (IS and v path to 1e-9).
  G5 errors: guards without 'twap', guarding 'twap', unknown regime ids and regime names with no stress match are rejected.
Only the aggressive policy is guarded in G1-G4 (policies=(BAD,)); guarding every policy would also trip/clip the benchmarks.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np

from live_session import LiveEpisode
from guards import GuardConfig, GuardState, mtm_cost, resolve_stress

SEED, PART = 1_000_000, 500
FAILS = []
BAD = "heuristic_belief"                    # the policy we make aggressive; it is not the last hosted policy


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def make_bad(ep):
    p = next(p for p in ep.policies if p.name == BAD)
    p.act = lambda x: float(min(x.q, 3.0 * x.q / (x.N - x.t)))      # 3x the TWAP rate, forced liquidation on the last step
    return p


def run_all(ep):
    resp = []
    while not ep.done:
        resp.append(ep.step())
    return resp


def vpath(resp, name):
    return np.array([r["policies"][name]["v"] for r in resp])


# ---- U ----------------------------------------------------------------------------------------
print("\n== unit")
check("U1 invalid configs rejected", raises(lambda: GuardConfig(breaker_usd=0.0)) and raises(lambda: GuardConfig(stress_prob=2.0))
      and raises(lambda: GuardConfig(stress_lo=1.5, stress_hi=1.0)))
check("U2 resolve_stress by name / explicit / none",
      resolve_stress(GuardConfig(constrain_in_stress=True), ["calm", "Stressed"], 2) == (1,)
      and resolve_stress(GuardConfig(constrain_in_stress=True, stress_regimes=(0,)), ["a", "b"], 2) == (0,)
      and resolve_stress(GuardConfig(breaker_usd=1.0), ["a", "b"], 2) is None)
check("U3 resolve_stress rejects no match / bad index",
      raises(lambda: resolve_stress(GuardConfig(constrain_in_stress=True), ["a", "b"], 2))
      and raises(lambda: resolve_stress(GuardConfig(constrain_in_stress=True, stress_regimes=(5,)), ["a", "b"], 2)))
check("U4 mtm_cost", abs(mtm_cost(10.0, 100.0, 400.0, 50.0, 9.0) - (1000.0 - (400.0 + 450.0))) < 1e-12)
gs = GuardState(GuardConfig(constrain_in_stress=True, stress_regimes=(1,), stress_prob=0.5), ["x"], (1,))
check("U5 constrain clips to the band", abs(gs.constrain("x", 0, 10, 100.0, 90.0, 0.9) - 12.5) < 1e-12
      and abs(gs.constrain("x", 0, 10, 100.0, 1.0, 0.9) - 7.5) < 1e-12 and gs.constrain("x", 0, 10, 100.0, 10.0, 0.9) == 10.0)
check("U6 no clip when calm or on the last step", gs.constrain("x", 0, 10, 100.0, 90.0, 0.1) == 90.0
      and gs.constrain("x", 9, 10, 100.0, 100.0, 0.9) == 100.0)
check("U7 breaker latches", (lambda g: (g.breaker("x", 3, 6.0), g.breaker("x", 4, 0.0))[1])(
    GuardState(GuardConfig(breaker_usd=5.0), ["x"], None)) is True)

# ---- baseline runs ------------------------------------------------------------------------------
base = LiveEpisode(SEED, imm_particles=PART); rb = run_all(base)                     # no guards
ub = LiveEpisode(SEED, imm_particles=PART); make_bad(ub); ru = run_all(ub)           # aggressive policy, no guards
lb, lu = base.logs(), ub.logs()
D = lu[BAD].IS - lu["twap"].IS

# ---- G1 ---------------------------------------------------------------------------------------
print("\n== G1 circuit breaker")
check("G1a precondition: the aggressive policy ends worse than TWAP", D > 0, f"(IS gap {D:.2f})")
X = max(D / 4.0, 1e-6)
g1 = LiveEpisode(SEED, imm_particles=PART, guards=GuardConfig(breaker_usd=X, policies=(BAD,))); make_bad(g1); r1 = run_all(g1)
rep = g1.guard_report()
tr = rep["tripped"].get(BAD)
check("G1b breaker tripped for the guarded aggressive policy and for no other", tr is not None and list(rep["tripped"]) == [BAD],
      f"(tripped at t={tr}, X={X:.2f})")
if tr is not None:
    ok = all(abs(r1[t]["policies"][BAD]["v"] - r1[t]["policies"][BAD]["q"] / (g1.N - t)) < 1e-9 for t in range(tr, g1.N))
    check("G1c from the trip step on it trades exactly the TWAP rate", ok)
check("G1d guarded IS is lower than unguarded IS", g1.logs()[BAD].IS < lu[BAD].IS,
      f"(guarded {g1.logs()[BAD].IS:.2f} unguarded {lu[BAD].IS:.2f} twap {lu['twap'].IS:.2f})")
check("G1e twap itself untouched", np.array_equal(vpath(r1, "twap"), vpath(rb, "twap")))

# ---- G2 / G3 ----------------------------------------------------------------------------------
print("\n== G2 stress constraint")
g2 = LiveEpisode(SEED, imm_particles=PART,
                 guards=GuardConfig(constrain_in_stress=True, stress_regimes=(1,), stress_prob=0.0, policies=(BAD,)))
make_bad(g2); r2 = run_all(g2)
mult = np.array([r2[t]["policies"][BAD]["v"] / (r2[t]["policies"][BAD]["q"] / (g2.N - t)) if r2[t]["policies"][BAD]["q"] > 0 else 1.0
                 for t in range(g2.N - 1)])
check("G2a every non-final action inside [0.75, 1.25] x TWAP rate", bool(np.all((mult >= 0.75 - 1e-9) & (mult <= 1.25 + 1e-9))),
      f"(min {mult.min():.3f} max {mult.max():.3f})")
check("G2b final step is full liquidation", abs(r2[-1]["policies"][BAD]["v"] - r2[-1]["policies"][BAD]["q"]) < 1e-9)
lg = g2.guard_report()["log"]
check("G2c constraint log is consistent", len(lg) > 0 and all(e["type"] == "action_constrained" and 0.75 - 1e-9 <= e["mult"] <= 1.25 + 1e-9
      and e["v"] != e["v_raw"] for e in lg), f"({len(lg)} events)")

print("\n== G3 isolation")
others = [p.name for p in base.policies if p.name != BAD]
iso = max(float(np.max(np.abs(vpath(r2, n) - vpath(rb, n)))) for n in others)
check("G3 unguarded policies unchanged by the guards", iso < 1e-12, f"(max |dv| {iso:.2e})")

# ---- G4 ---------------------------------------------------------------------------------------
print("\n== G4 rollback with guards active")
cfg = GuardConfig(breaker_usd=X, constrain_in_stress=True, stress_regimes=(1,), stress_prob=0.0, policies=(BAD,))
clean = LiveEpisode(SEED, imm_particles=PART, guards=cfg); make_bad(clean); rc = run_all(clean)
f = LiveEpisode(SEED, imm_particles=PART, guards=cfg); make_bad(f)
victim = f.policies[-1]                              # unguarded, so its act() is always called, after the guarded policy has acted
orig, armed = victim.act, {"on": False}
def flaky(x):
    if armed["on"]:
        armed["on"] = False
        raise RuntimeError("injected fault")
    return orig(x)
victim.act = flaky
untouched, fired_every_step, rf = True, True, []
while not f.done:
    before = f.guard.snapshot()
    armed["on"] = True
    try:
        rf.append(f.step())                          # fault did not fire: the step succeeded
        fired_every_step = False
    except RuntimeError:
        untouched = untouched and f.guard.snapshot() == before
        rf.append(f.step())                          # retry
    armed["on"] = False
check("G4a the injected fault fired at every step", fired_every_step)
check("G4b guard state identical after every failed step", untouched)
dIS = max(abs(f.logs()[n].IS - clean.logs()[n].IS) for n in clean.logs())
dv = max(float(np.max(np.abs(vpath(rf, n) - vpath(rc, n)))) for n in clean.logs())
check("G4c retried episode equals the clean guarded episode", dIS < 1e-9 and dv < 1e-9, f"(|dIS| {dIS:.2e} |dv| {dv:.2e})")

# ---- G5 ---------------------------------------------------------------------------------------
print("\n== G5 errors")
check("G5a guards without twap rejected", raises(lambda: LiveEpisode(SEED, policies=("ce_oracle",), imm_particles=PART,
                                                                       guards=GuardConfig(breaker_usd=1.0))))
check("G5b guarding twap rejected", raises(lambda: LiveEpisode(SEED, imm_particles=PART,
                                                                 guards=GuardConfig(breaker_usd=1.0, policies=("twap",)))))
check("G5c regime id out of range rejected", raises(lambda: LiveEpisode(SEED, imm_particles=PART,
      guards=GuardConfig(constrain_in_stress=True, stress_regimes=(9,)))))
check("G5d inactive config is a no-op", LiveEpisode(SEED, imm_particles=PART, guards=GuardConfig()).guard is None)

print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)