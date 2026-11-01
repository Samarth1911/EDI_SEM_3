"""
score_guarded.py -- Stage 4C Tier 1 guards applied to the Stage 3 ensembles.  NO TRAINING.  NOT RUN by its author.
Run from EDI/validation (venv active):
    python score_guarded.py --arm oracle                (fast: no filter)
    python score_guarded.py --arm filt_full_bonus       (reruns the live filter for every episode: slow)

EXPLORATORY and SEPARATE from the Stage 3 results (nothing here changes STAGE3_HANDOFF / STAGE3_REVISIT_HANDOFF numbers).
Policy under test = argmax of mean Q over the arm's 5 validation-selected members (as in score_ensemble.py), wrapped by:
  breaker : once the cumulative cost gap to TWAP (over COMPLETED steps) exceeds X dollars, the agent is switched off and
            trades the TWAP action (multiplier 1.0) until the end (latched).
  band    : when P(stress) >= prob, the chosen multiplier is clipped to [lo, hi] (default 0.75..1.25) and snapped to the grid.
            P(stress) = last entry of the agent observation (regime probabilities are its last K entries for the arms that have
            them; for the ORACLE arm those entries are the one-hot TRUE regime, i.e. perfect detection). Verified at run time.
  The final step is never constrained (forced liquidation).

PROTOCOL (set before any guarded number was seen)
  1. Tuning worlds 1,500,000+ (--n-tune, default 300; disjoint from training, validation 800000+, test 900000+, fresh 1,000,000+).
     Configs: breaker X in {50,100,200,400} $, band prob in {0.5, 0.8}, and every breaker x band combination.
  2. Selection: among configs whose mean noise-free diff vs TWAP is within --tol (5 $) of the unguarded ensemble, take the one with
     the lowest CVaR95 (full IS) vs TWAP; it only counts if that CVaR95 is lower than the unguarded ensemble's by >= --min-gain (10 $).
     Otherwise: "no config qualifies".
  3. The ONE selected config is scored once on TEST (900000+) and FRESH (1,000,000+) worlds, paired against the unguarded ensemble and TWAP.
     --report-all also prints every other config on those blocks (not selected: do not quote them as results).

CAVEATS
  * The breaker's gap here is the NOISE-FREE decomposition gap (uses the true alpha, kappa of each step: evaluation-side), i.e. a
    stand-in for the live guard, which sees realised prices (noisy) and will trip differently. X calibrated here does not transfer 1:1.
  * CVaR95 rests on 5% of worlds (10 of 200, 25 of 500): wide uncertainty; a bootstrap se is printed.
  * Selection used the CVaR criterion on tuning worlds; test/fresh are untouched by it.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
from ddqn import DDQNAgent
from ddqn_train import run_episodes, EVAL_BASE

ARMS = {
    "oracle": dict(arm="oracle"), "raw_proxy": dict(arm="raw_proxy"), "filt_full": dict(arm="filtered"),
    "filt_mean_only": dict(arm="filtered", no_unc=True), "filt_no_regime": dict(arm="filtered", no_regime=True),
    "filt_single": dict(arm="filtered", filt="single"), "filt_full_bonus": dict(arm="filtered"),
    "filt_mean_bonus": dict(arm="filtered", no_unc=True),
}
BAND_ARMS = {"oracle", "filt_full", "filt_mean_only", "filt_full_bonus", "filt_mean_bonus"}   # regime probs are the last obs entries

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="filt_full_bonus", choices=list(ARMS))
ap.add_argument("--tag", default=None)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--n-tune", type=int, default=300)
ap.add_argument("--n-test", type=int, default=200)
ap.add_argument("--n-fresh", type=int, default=500)
ap.add_argument("--imm-particles", type=int, default=2000)
ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "outputs", "stage3c", "ckpt"))
ap.add_argument("--breaker-usd", default="50,100,200,400")
ap.add_argument("--stress-probs", default="0.5,0.8")
ap.add_argument("--stress-lo", type=float, default=0.75)
ap.add_argument("--stress-hi", type=float, default=1.25)
ap.add_argument("--tol", type=float, default=5.0)
ap.add_argument("--min-gain", type=float, default=10.0)
ap.add_argument("--no-band", action="store_true")
ap.add_argument("--report-all", action="store_true")
a = ap.parse_args()

tag = a.tag if a.tag is not None else ("_abl_b1" if a.arm.endswith("_bonus") else "_abl")
spec = ARMS[a.arm]
single = spec.get("filt") == "single"
env = make_env(EnvConfig(obs_mode=spec["arm"], use_uncertainty=not spec.get("no_unc", False),
                         use_regime=not spec.get("no_regime", False), n_regimes=1 if single else 2, reward_scale=0.1),
               obs_kind="gaussian", filter_regimes="single" if single else "imm", seed=0,
               imm_params=IMMParams(n_particles=a.imm_particles), stationary_init=True)
mults = [float(m) for m in env.cfg.action_multipliers]
N, q0 = int(env.N), float(env.p.q0)
v_tw = q0 / N
assert abs(mults[env.twap_action] - 1.0) < 1e-12, "twap_action is not the 1.0 multiplier"
allowed = [i for i, m in enumerate(mults) if a.stress_lo - 1e-12 <= m <= a.stress_hi + 1e-12]
assert allowed, "no grid multiplier inside the stress band"


def snap(i: int) -> int:
    m = min(max(mults[i], a.stress_lo), a.stress_hi)
    return min(allowed, key=lambda j: abs(mults[j] - m))


agents = []
for s in range(a.seeds):
    p = os.path.join(a.ckpt_dir, f"{a.arm}_base1{tag}_s{s}_valsel.npz")
    ag = DDQNAgent.load(p)
    assert ag.obs_dim == env.obs_dim and ag.n_actions == env.n_actions, f"{p}: agent/env mismatch"
    agents.append(ag)
print(f"arm={a.arm} tag={tag}: {len(agents)} val-sel agents; obs_dim={env.obs_dim} n_actions={env.n_actions} N={N}")


class Ensemble:
    def __init__(self, ags):
        self.ags = ags

    def greedy(self, obs):
        return int(np.argmax(np.mean([g.q_values(obs)[0] for g in self.ags], axis=0)))


ens = Ensemble(agents)


def run_guarded(seed, cfg):
    """One episode of the ensemble under cfg = dict(X=None|float, band=None|prob). Returns per-episode numbers."""
    obs, _ = env.reset(seed=int(seed))
    q, t, done = q0, 0, False
    tripped, trip_t, n_con, cum_gap, cum_ag = False, None, 0, 0.0, 0.0
    while not done:
        act = ens.greedy(obs)
        if not tripped and cfg["X"] is not None and cum_gap > cfg["X"]:
            tripped, trip_t = True, t
        if tripped:
            act = env.twap_action
        elif cfg["band"] is not None and t < N - 1 and float(obs[-1]) >= cfg["band"]:
            a2 = snap(act)
            n_con += int(a2 != act)
            act = a2
        v = q if t == N - 1 else min(mults[act] * q / (N - t), q)
        obs, _, done, _, _ = env.step(act)
        row = env.log[-1]
        al, ka = float(row["true_alpha"]), float(row["true_kappa"])
        c_ag = al * v * v + ka * v * (q - v)                              # drift-only decomposition, this step
        c_tw = al * v_tw * v_tw + ka * v_tw * (q0 - (t + 1) * v_tw)       # what TWAP pays in the same step
        cum_ag += c_ag
        cum_gap += c_ag - c_tw
        q -= v
        t += 1
    e = env.get_episode()
    return dict(IS=float(e["IS"]), nf=float(e["sum_decomp"]), my_nf=cum_ag,
                q_left=float(e["q0"] - sum(x["v"] for x in e["steps"])), tripped=tripped, trip_t=trip_t, n_con=n_con)


FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


# ---- self-checks (stop before the long runs if any fails) --------------------------------------------------------------
print("\n== self-checks")
for s in (EVAL_BASE, EVAL_BASE + 1, EVAL_BASE + 2):
    tw = run_episodes(env, [s], None)
    en = run_episodes(env, [s], ens)
    g0 = run_guarded(s, dict(X=None, band=None))
    check(f"G0 [{s}] inert wrapper == unguarded ensemble", abs(g0["IS"] - float(en["IS"][0])) < 1e-9,
          f"(dIS {abs(g0['IS'] - float(en['IS'][0])):.2e})")
    check(f"D  [{s}] recomputed decomposition == env sum_decomp", abs(g0["my_nf"] - g0["nf"]) < 1e-6 * max(1.0, abs(g0["nf"])),
          f"(mine {g0['my_nf']:.4f} env {g0['nf']:.4f})")
    check(f"Q  [{s}] fully liquidated", abs(g0["q_left"]) < 1e-6 * q0, f"(q_left {g0['q_left']:.3e})")
    gt = run_guarded(s, dict(X=-1.0, band=None))
    check(f"T  [{s}] breaker from step 0 == TWAP", gt["trip_t"] == 0 and abs(gt["IS"] - float(tw["IS"][0])) < 1e-9,
          f"(trip_t {gt['trip_t']}, dIS {abs(gt['IS'] - float(tw['IS'][0])):.2e})")

obs, _ = env.reset(seed=EVAL_BASE)
tail_ok, done = True, False
while not done:
    tail = np.asarray(obs[-2:], dtype=float)
    tail_ok = tail_ok and bool(np.all((tail >= -1e-9) & (tail <= 1 + 1e-9)) and abs(tail.sum() - 1.0) < 1e-6)
    obs, _, done, _, _ = env.step(ens.greedy(obs))
band_ok = tail_ok and (a.arm in BAND_ARMS) and not single and not a.no_band
print(f"  band configs {'ENABLED' if band_ok else 'SKIPPED'} (regime-probability layout detected: {tail_ok}; arm supports it: {a.arm in BAND_ARMS})"
      + ("   NOTE: oracle arm -> P(stress) is the TRUE regime (perfect detection)" if a.arm == "oracle" and band_ok else ""))
if FAILS:
    print("\nSELF-CHECKS FAILED:", ", ".join(FAILS), "-- not running the study. Paste this output back.")
    sys.exit(1)

# ---- helpers ------------------------------------------------------------------------------------------------------------
se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))


def cvar(x, alpha=0.05):
    k = max(1, int(np.ceil(alpha * len(x))))
    return float(np.sort(x)[-k:].mean())


def boot_cvar_delta(x, y, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(reps):
        j = rng.integers(0, n, n)
        vals.append(cvar(x[j]) - cvar(y[j]))
    return cvar(x) - cvar(y), float(np.std(vals, ddof=1))


def score_block(seeds, cfgs):
    seeds = list(seeds)
    tw = run_episodes(env, seeds, None)
    en = run_episodes(env, seeds, ens)
    out = {"twap": dict(IS=tw["IS"], nf=tw["IS_decomp"]), "ens": dict(IS=en["IS"], nf=en["IS_decomp"]), "seeds": np.array(seeds)}
    for c in cfgs:
        rs = [run_guarded(s, c) for s in seeds]
        out[c["name"]] = dict(IS=np.array([r["IS"] for r in rs]), nf=np.array([r["nf"] for r in rs]),
                              trip=np.array([r["tripped"] for r in rs]), trip_t=np.array([r["trip_t"] if r["tripped"] else -1 for r in rs]),
                              ncon=np.array([r["n_con"] for r in rs]))
    return out


def print_block(title, res, names):
    tw, en = res["twap"], res["ens"]
    n = len(tw["IS"])
    print(f"\n===== {title}: {n} worlds; TWAP mean IS {tw['IS'].mean():.1f} =====")
    d = en["nf"] - tw["nf"]
    print(f"  unguarded ensemble : nf vs TWAP {d.mean():+7.1f} (se {se(d):4.1f})   CVaR95-TWAP {cvar(en['IS']) - cvar(tw['IS']):+7.1f}")
    for nm in names:
        r = res[nm]
        d_t = r["nf"] - tw["nf"]
        d_e = r["nf"] - en["nf"]
        cd, cse = boot_cvar_delta(r["IS"], en["IS"])
        tr = r["trip"]
        print(f"  {nm:<14}: nf vs TWAP {d_t.mean():+7.1f} (se {se(d_t):4.1f})   CVaR95-TWAP {cvar(r['IS']) - cvar(tw['IS']):+7.1f}")
        print(f"  {'':<14}  guarded - unguarded: nf {d_e.mean():+6.1f} (paired se {se(d_e):4.1f}, t {d_e.mean() / max(se(d_e), 1e-12):+.2f})"
              f"   CVaR95 {cd:+6.1f} (boot se {cse:4.1f})")
        print(f"  {'':<14}  breaker tripped in {tr.mean():.1%} of worlds"
              + (f" (mean trip step {r['trip_t'][tr].mean():.1f})" if tr.any() else "")
              + f"; band changed {r['ncon'].sum() / (n * (N - 1)):.1%} of non-final steps")


# ---- tuning -----------------------------------------------------------------------------------------------------------
Xs = [float(x) for x in a.breaker_usd.split(",") if x]
Ps = [float(x) for x in a.stress_probs.split(",") if x] if band_ok else []
cfgs = [dict(name=f"brk{x:g}", X=x, band=None) for x in Xs]
cfgs += [dict(name=f"band{p:g}", X=None, band=p) for p in Ps]
cfgs += [dict(name=f"brk{x:g}+band{p:g}", X=x, band=p) for x in Xs for p in Ps]
names = [c["name"] for c in cfgs]

print(f"\n== tuning on worlds 1,500,000+ ({a.n_tune} worlds, {len(cfgs)} configs)")
tune = score_block(range(1_500_000, 1_500_000 + a.n_tune), cfgs)
print_block("TUNING worlds", tune, names)

tw_t, en_t = tune["twap"], tune["ens"]
m0 = float(np.mean(en_t["nf"] - tw_t["nf"]))
c0 = cvar(en_t["IS"]) - cvar(tw_t["IS"])
best, best_c = None, c0 - a.min_gain
for nm in names:
    r = tune[nm]
    m = float(np.mean(r["nf"] - tw_t["nf"]))
    c = cvar(r["IS"]) - cvar(tw_t["IS"])
    if m <= m0 + a.tol and c <= best_c:
        best, best_c = nm, c
if best is None:
    print(f"\nSELECTION: no config qualifies (needs mean nf within {a.tol:g} $ of unguarded AND CVaR95 at least {a.min_gain:g} $ lower; "
          f"unguarded tuning CVaR95-TWAP {c0:+.1f}).")
else:
    print(f"\nSELECTION: {best} (tuning CVaR95-TWAP {best_c:+.1f} vs unguarded {c0:+.1f}; mean nf within {a.tol:g} $).")

# ---- final blocks -------------------------------------------------------------------------------------------------------
final_names = names if a.report_all else ([best] if best else [])
final_cfgs = [c for c in cfgs if c["name"] in final_names]
store = {"tune_seeds": tune["seeds"], "selected": np.array(best if best else "none")}
for nm in ["twap", "ens"] + names:
    store[f"tune_{nm}_IS"], store[f"tune_{nm}_nf"] = tune[nm]["IS"], tune[nm]["nf"]
for title, w0, n in (("TEST worlds", EVAL_BASE, a.n_test), ("FRESH worlds", 1_000_000, a.n_fresh)):
    res = score_block(range(w0, w0 + n), final_cfgs)
    shown = final_names
    print_block(title + (" (selected config)" if not a.report_all else " (ALL configs; only the selected one is a result)"), res, shown)
    blk = "test" if title.startswith("TEST") else "fresh"
    for nm in ["twap", "ens"] + shown:
        store[f"{blk}_{nm}_IS"], store[f"{blk}_{nm}_nf"] = res[nm]["IS"], res[nm]["nf"]
    store[f"{blk}_seeds"] = res["seeds"]

out = os.path.join(HERE, "outputs", "stage3c", f"guarded_{a.arm}.npz")
np.savez(out, **store)
print("\nsaved per-world arrays to", out)
print("Reminder: exploratory; breaker gap here is the noise-free decomposition (not the live realised-price gap); "
      "Stage 3 numbers are unchanged.")