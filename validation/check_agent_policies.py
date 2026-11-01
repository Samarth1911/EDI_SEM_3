"""
check_agent_policies.py -- Stage 4C: hosted DDQN agents / ensembles.  NOT RUN by its author.  Run from EDI/validation (venv active):
    python check_agent_policies.py                       (500 particles for A/F/D/H/E; saved-score check at 2000 particles)

  X  PolicyInput has the optional obs field; arm_config() equals the EnvConfig that score_ensemble.py builds.
  A  live == environment harness: for each arm, ensemble and single member s0, on one world, the live observation vectors
     (PolicyInput.obs), the v path and the IS equal what BeliefMDPEnv produces (max |diff| printed; expect 0).
  F  information barriers: oracle-arm agents get truth and no belief, filtered agents get the belief and no truth, raw_proxy
     agents get neither; obs has the arm's dimension.
  D  contract: step responses pass validate_step_response and carry action/mult for the agents.
  P  saved scores: live ensemble noise-free cost equals outputs/stage3c/ens_nf_<arm>.npz (test worlds) to 1e-6 (2000 particles).
  H  band guard on an ensemble: effective multiplier == clip(raw multiplier, 0.75, 1.25) when P(stress) >= 0.5, else raw.
  E  errors: wrong family / obs_kind, arms without Hawkes agents (bonus), unknown arm (filt_single), member index out of range are
     rejected; a Hawkes single agent steps; a Hawkes ensemble steps (skipped until its 5 checkpoints exist).
  Hawkes: A/F/D (and P, if outputs/stage3c/ens_nf_<arm>_hawkes.npz exists) also run for the spread_hawkes agents that are hosted.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np

import interfaces as I
from live_session import LiveEpisode
import agent_policies as AP
from guards import GuardConfig
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="oracle,raw_proxy,filt_full,filt_no_regime,filt_full_bonus")
ap.add_argument("--particles", type=int, default=500)
ap.add_argument("--saved-arms", default="oracle,filt_full_bonus")
ap.add_argument("--hawkes-arms", default="oracle,raw_proxy,filt_full,filt_mean_only")
ap.add_argument("--saved-hawkes-arms", default="oracle,filt_full")
ap.add_argument("--saved-particles", type=int, default=2000)
ap.add_argument("--n-saved", type=int, default=3)
ap.add_argument("--seed", type=int, default=900000)
a = ap.parse_args()
ARMS_RUN = [x for x in a.arms.split(",") if x]
SAVED_ARMS = [x for x in a.saved_arms.split(",") if x]
HAWKES_RUN = [x for x in a.hawkes_arms.split(",") if x]
SAVED_HAWKES = [x for x in a.saved_hawkes_arms.split(",") if x]
RUNS = [("gaussian", x) for x in ARMS_RUN] + [("spread_hawkes", x) for x in HAWKES_RUN]
FAILS = []


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


# original per-arm specs, copied from score_ensemble.py, to cross-check arm_config
SPEC = {"oracle": dict(arm="oracle"), "raw_proxy": dict(arm="raw_proxy"), "filt_full": dict(arm="filtered"),
        "filt_mean_only": dict(arm="filtered", no_unc=True), "filt_no_regime": dict(arm="filtered", no_regime=True),
        "filt_full_bonus": dict(arm="filtered"), "filt_mean_bonus": dict(arm="filtered", no_unc=True)}


def spec_cfg(arm):
    s = SPEC[arm]
    return EnvConfig(obs_mode=s["arm"], use_uncertainty=not s.get("no_unc", False), use_regime=not s.get("no_regime", False),
                     n_regimes=2, reward_scale=0.1)


def env_for(arm, particles, kind="gaussian"):
    return make_env(spec_cfg(arm), obs_kind=kind, filter_regimes="imm", seed=0,
                    imm_params=IMMParams(n_particles=particles), stationary_init=True)


def env_run(env, seed, members):
    obs, _ = env.reset(seed=int(seed))
    O, done = [], False
    while not done:
        O.append(np.array(obs, copy=True))
        act = int(np.argmax(np.mean([m.q_values(obs)[0] for m in members], axis=0)))
        obs, _, done, _, _ = env.step(act)
    e = env.get_episode()
    return np.array(O), np.array([r["v"] for r in e["steps"]]), float(e["IS"])


def live_run(name, seed, particles, **kw):
    ep = LiveEpisode(int(seed), imm_particles=particles, policies=("twap", name), **kw)
    pol = ep.policies[1]
    rec, orig = [], pol.act
    def wrapped(x, _o=orig):
        rec.append((None if x.obs is None else np.array(x.obs, copy=True), x.belief is not None, x.truth is not None))
        return _o(x)
    pol.act = wrapped
    resp = []
    while not ep.done:
        resp.append(ep.step())
    return ep, pol, rec, resp


def nf_from_log(l):
    return float(sum(s.true_alpha * s.v ** 2 + s.true_kappa * s.v * (s.q_before - s.v) for s in l.steps))


# ---- X ------------------------------------------------------------------------------------------------------------------
print("\n== setup")
check("X1 PolicyInput has the obs field", "obs" in I.PolicyInput.__dataclass_fields__,
      "(add `obs: Optional[np.ndarray] = None` to interfaces.PolicyInput)")
if FAILS:
    sys.exit(1)
check("X2 arm_config == score_ensemble's EnvConfig for every arm", all(AP.arm_config(r) == spec_cfg(r) for r in SPEC))

# ---- A / F / D ------------------------------------------------------------------------------------------------------------
for kind, arm in RUNS:
    print(f"\n== arm {arm} [{kind}] (world {a.seed}, {a.particles} particles)")
    env = env_for(arm, a.particles, kind)
    for name in (f"ens_{arm}", f"ddqn_{arm}_s0"):
        if name not in set(AP.agent_names(kind)):
            print(f"  [SKIP] {name} is not hosted for {kind} (checkpoints missing)")
            continue
        pol0 = AP.make_agent_policy(name, kind)
        O, vE, isE = env_run(env, a.seed, pol0.members)
        ep, pol, rec, resp = live_run(name, a.seed, a.particles, obs_kind=kind)
        N = ep.N
        Ol = np.array([r[0] for r in rec])
        vL = np.array([r["policies"][name]["v"] for r in resp])
        isL = ep.logs()[name].IS
        d_obs = float(np.max(np.abs(O[:N] - Ol)))
        d_v = float(np.max(np.abs(vE - vL)))
        check(f"A1 [{name}] live obs == env obs", O[:N].shape == Ol.shape and d_obs == 0.0, f"(max |d| {d_obs:.2e}, shape {Ol.shape})")
        check(f"A2 [{name}] live v path == env v path", d_v < 1e-9, f"(max |dv| {d_v:.2e})")
        check(f"A3 [{name}] live IS == env IS", abs(isE - isL) < 1e-9, f"(env {isE:.4f} live {isL:.4f})")
        want_truth, want_belief = pol.cfg.obs_mode == "oracle", pol.cfg.obs_mode == "filtered"
        ok = all(r[2] == want_truth and r[1] == want_belief for r in rec)
        check(f"F1 [{name}] truth/belief barrier for obs_mode={pol.cfg.obs_mode}", ok)
        errs = [e for r in resp for e in I.validate_step_response(r)]
        check(f"D1 [{name}] step responses pass the contract", not errs, str(errs[:2]))
        ok = all("action" in r["policies"][name] and 0 <= r["policies"][name]["action"] < len(pol.mults) for r in resp)
        check(f"D2 [{name}] action/mult reported", ok)

# ---- P ----------------------------------------------------------------------------------------------------------------------
for kind, arm in [("gaussian", x) for x in SAVED_ARMS] + [("spread_hawkes", x) for x in SAVED_HAWKES]:
    path = os.path.join(HERE, "outputs", "stage3c", f"ens_nf_{arm}{'' if kind == 'gaussian' else '_hawkes'}.npz")
    print(f"\n== saved scores {kind}/{arm} ({a.saved_particles} particles)")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        continue
    if f"ens_{arm}" not in set(AP.agent_names(kind)):
        print(f"  [SKIP] no ensemble hosted for {kind}/{arm}")
        continue
    z = np.load(path)
    seeds, ens_nf = z["test_seeds"][:a.n_saved], z["test_ens"][:a.n_saved]
    got = []
    for s in seeds:
        ep, pol, rec, resp = live_run(f"ens_{arm}", s, a.saved_particles, obs_kind=kind)
        got.append(nf_from_log(ep.logs()[f"ens_{arm}"]))
    gap = float(np.max(np.abs(np.array(got) - ens_nf)))
    check(f"P1 [{kind}/ens_{arm}] live noise-free cost == saved ens_nf (test worlds {int(seeds[0])}..)", gap < 1e-6, f"(max gap {gap:.2e})")

# ---- H ----------------------------------------------------------------------------------------------------------------------
print("\n== band guard on an ensemble")
name = "ens_filt_full_bonus"
g = GuardConfig(constrain_in_stress=True, stress_regimes=(1,), stress_prob=0.5, policies=(name,))
ep, pol, rec, resp = live_run(name, a.seed, a.particles, guards=g)
bad, n_stressed = 0, 0
for r in resp[:-1]:
    e = r["policies"][name]
    base = e["q"] / (ep.N - r["t"])
    if e["q"] <= 0:
        continue
    eff = e["v"] / base
    stressed = r["belief"]["regime_probs"][1] >= 0.5
    n_stressed += int(stressed)
    want = min(max(e["mult"], 0.75), 1.25) if stressed else e["mult"]
    bad += int(abs(eff - want) > 1e-9)
check("H1 effective multiplier == clip(raw, 0.75, 1.25) when stressed, raw otherwise", bad == 0, f"({bad} violations, {n_stressed} stressed steps)")

# ---- E ----------------------------------------------------------------------------------------------------------------------
print("\n== errors and Hawkes")
mk = lambda name, kind: (lambda: LiveEpisode(1, obs_kind=kind, imm_particles=a.particles, policies=("twap", name)))
check("E1 ensemble of an arm without Hawkes agents rejected on spread_hawkes", raises(mk("ens_filt_full_bonus", "spread_hawkes")))
check("E2 bonus arm on spread_hawkes rejected", raises(mk("ddqn_filt_full_bonus_s0", "spread_hawkes")))
check("E3 filt_single not hosted", raises(mk("ddqn_filt_single_s0", "gaussian")))
check("E4 member index out of range rejected", raises(mk("ddqn_oracle_s7", "gaussian")) and raises(mk("ddqn_oracle_s5", "spread_hawkes")))
check("E5 unknown ensemble rejected", raises(mk("ens_nope", "gaussian")))
try:
    ep, pol, rec, resp = live_run("ddqn_filt_full_s0", a.seed, a.particles) if False else (None, None, None, None)
    eh = LiveEpisode(a.seed, obs_kind="spread_hawkes", imm_particles=a.particles, policies=("twap", "ddqn_filt_full_s0"))
    r0 = eh.step()
    check("E6 spread_hawkes single agent steps", "action" in r0["policies"]["ddqn_filt_full_s0"] and not I.validate_step_response(r0))
except Exception as ex:
    check("E6 spread_hawkes single agent steps", False, f"({type(ex).__name__}: {ex})")

if "ens_oracle" in set(AP.agent_names("spread_hawkes")):
    try:
        eh = LiveEpisode(a.seed, obs_kind="spread_hawkes", imm_particles=a.particles, policies=("twap", "ens_oracle"))
        r0 = eh.step()
        check("E7 spread_hawkes ensemble steps", "action" in r0["policies"]["ens_oracle"] and not I.validate_step_response(r0))
    except Exception as ex:
        check("E7 spread_hawkes ensemble steps", False, f"({type(ex).__name__}: {ex})")
else:
    print("  [SKIP] E7: no spread_hawkes ensemble checkpoints yet")

print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)