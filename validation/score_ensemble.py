"""
score_ensemble.py -- Stage 3 revisit. NO TRAINING. UNTESTED until you run it. Run from EDI/validation (venv active):
    python score_ensemble.py --arm oracle
    python score_ensemble.py --arm filt_full_bonus          (filtered arms run the live particle filter: slower)
    python score_ensemble.py --arm filt_full --obs-kind spread_hawkes   (Hawkes arms: oracle, raw_proxy, filt_full, filt_mean_only;
        needs 5 checkpoints <arm>_base1_hawkes_s0..4_valsel.npz; saves ens_nf_<arm>_hawkes.npz, Gaussian files are never touched)

Question: does averaging the Q-values of the 5 independently trained, validation-selected agents of an arm reduce the
checkpoint/seed noise?  Policy = argmax of the mean Q over the 5 members. No tuning anywhere: all 5 members, mean Q.
Members were selected on the validation worlds (800000+), so test (900000+) and fresh (1,000,000+) worlds are untouched.
Reports, per world block (paired vs TWAP, noise-free = the Stage 3 metric, $; negative = better):
  each member, the mean of the members, the ENSEMBLE, ensemble - mean(members) with a paired se over worlds,
  full-IS CVaR95 vs TWAP, and the mean multiplier by true regime.
References (same worlds): CE grid -90.8 (test, oracle info), CE_filtered_grid -90.2 / CE_oracle_grid -92.2 (fresh).
Pre-registered reading: noise is the limiter if ensemble - mean(members) is below -2 se on BOTH world blocks.
Assumes agent files <arm>_base1<tag>_s<seed>_valsel.npz in outputs/stage3c/ckpt (tag _abl, or _abl_b1 for bonus arms)
and that DDQNAgent has q_values(obs) (it does in ddqn.py as I wrote it; the check below fails loudly otherwise).
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
from ddqn import DDQNAgent
from ddqn_train import run_episodes, paired, EVAL_BASE

ARMS = {
    "oracle": dict(arm="oracle"), "raw_proxy": dict(arm="raw_proxy"), "filt_full": dict(arm="filtered"),
    "filt_mean_only": dict(arm="filtered", no_unc=True), "filt_no_regime": dict(arm="filtered", no_regime=True),
    "filt_single": dict(arm="filtered", filt="single"), "filt_full_bonus": dict(arm="filtered"),
    "filt_mean_bonus": dict(arm="filtered", no_unc=True),
}
HAWKES_ARMS = ("oracle", "raw_proxy", "filt_full", "filt_mean_only")
ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="oracle", choices=list(ARMS))
ap.add_argument("--tag", default=None)
ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--n-test", type=int, default=200)
ap.add_argument("--n-fresh", type=int, default=500)
ap.add_argument("--imm-particles", type=int, default=2000)
ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "outputs", "stage3c", "ckpt"))
a = ap.parse_args()
hawkes = a.obs_kind == "spread_hawkes"
if hawkes and a.arm not in HAWKES_ARMS:
    raise SystemExit(f"no spread_hawkes agents were trained for arm {a.arm}; Hawkes arms: {list(HAWKES_ARMS)}")
tag = a.tag if a.tag is not None else ("_hawkes" if hawkes else ("_abl_b1" if a.arm.endswith("_bonus") else "_abl"))
spec = ARMS[a.arm]
single = spec.get("filt") == "single"
env = make_env(EnvConfig(obs_mode=spec["arm"], use_uncertainty=not spec.get("no_unc", False),
                         use_regime=not spec.get("no_regime", False), n_regimes=1 if single else 2, reward_scale=0.1),
               obs_kind=a.obs_kind, filter_regimes="single" if single else "imm", seed=0,
               imm_params=IMMParams(n_particles=a.imm_particles), stationary_init=True)

agents = []
for s in range(a.seeds):
    p = os.path.join(a.ckpt_dir, f"{a.arm}_base1{tag}_s{s}_valsel.npz")
    ag = DDQNAgent.load(p)
    assert hasattr(ag, "q_values") and ag.obs_dim == env.obs_dim and ag.n_actions == env.n_actions, \
        f"{p}: agent/env mismatch (obs_dim {ag.obs_dim} vs {env.obs_dim}, actions {ag.n_actions} vs {env.n_actions})"
    agents.append(ag)
print(f"arm={a.arm} obs_kind={a.obs_kind} tag={tag}: loaded {len(agents)} val-sel agents; obs_dim={env.obs_dim} n_actions={env.n_actions}")


class Ensemble:
    def __init__(self, ags):
        self.ags = ags

    def greedy(self, obs):
        return int(np.argmax(np.mean([g.q_values(obs)[0] for g in self.ags], axis=0)))


def cvar_diff(res, base):
    k = max(1, int(np.ceil(0.05 * len(res["IS"]))))
    return float(np.sort(res["IS"])[-k:].mean() - np.sort(base["IS"])[-k:].mean())


store = {}
for name, w0, n in (("TEST worlds", EVAL_BASE, a.n_test), ("FRESH worlds", 1_000_000, a.n_fresh)):
    seeds = range(w0, w0 + n)
    twap = run_episodes(env, seeds, None)
    mem = [run_episodes(env, seeds, g) for g in agents]
    ens = run_episodes(env, seeds, Ensemble(agents))
    mem_nf = np.array([r["IS_decomp"] for r in mem])                 # (members, worlds)
    d_mem = mem_nf.mean(axis=0) - twap["IS_decomp"]
    d_ens = ens["IS_decomp"] - twap["IS_decomp"]
    diff = ens["IS_decomp"] - mem_nf.mean(axis=0)
    blk = "test" if name.startswith("TEST") else "fresh"
    store[f"{blk}_seeds"] = np.array(list(seeds)); store[f"{blk}_twap"] = twap["IS_decomp"]
    store[f"{blk}_ens"] = ens["IS_decomp"]; store[f"{blk}_mem"] = mem_nf
    se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v)))
    print(f"\n===== {name}: {n} worlds from {w0}; TWAP mean IS {twap['IS'].mean():.1f} =====")
    for i, r in enumerate(mem):
        p = paired(r, twap, "IS_decomp")
        print(f"  member s{i}: nf {p['diff']:+7.1f} (se {p['se']:4.1f})   CVaR95-TWAP {cvar_diff(r, twap):+7.1f}")
    pm = np.mean([paired(r, twap, "IS_decomp")["diff"] for r in mem])
    pe = paired(ens, twap, "IS_decomp")
    print(f"  mean of members      : nf {pm:+7.1f}   CVaR95-TWAP {np.mean([cvar_diff(r, twap) for r in mem]):+7.1f}")
    print(f"  ENSEMBLE (mean Q)    : nf {pe['diff']:+7.1f} (se {pe['se']:4.1f})   CVaR95-TWAP {cvar_diff(ens, twap):+7.1f}"
          f"   mult by regime {ens['mult_by_regime']}")
    print(f"  ensemble - mean(members): {diff.mean():+6.1f} (paired se {se(diff):.1f}, t {diff.mean() / se(diff):+.2f})")
_out = os.path.join(HERE, "outputs", "stage3c", f"ens_nf_{a.arm}{'_hawkes' if hawkes else ''}.npz")
np.savez(_out, **store)
print("saved per-world arrays to", _out)
if hawkes:
    print("\nReference (first 200 fresh worlds, Stage 4 benchmarks with --obs-kind spread_hawkes): ce_filtered_grid -90.8, ce_oracle (continuous) -99.6; "
          "the Gaussian-world numbers (CE grid -90.8 test) are NOT the Hawkes reference on test/500-fresh worlds.")
else:
    print("\nReference (same worlds): CE on the grid about -90.8 (test) / -90.2..-92.2 (fresh); non-causal ceiling -193.4 (test).")