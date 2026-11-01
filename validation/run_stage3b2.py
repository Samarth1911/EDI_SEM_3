"""
run_stage3b2.py -- train ONE arm with the validation protocol, then evaluate BOTH the last and the
validation-selected agent on the held-out TEST worlds (seeds 900000+i, same as run_stage3b.py, so the TWAP
numbers are directly comparable with your earlier runs).  NOT RUN by its author. Run from EDI/validation.
All defaults are untuned choices; the two that differ from run_stage3b.py are flagged below.
"""
import argparse, csv, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
from ddqn import DDQNConfig
from ddqn_train import run_episodes, paired, EVAL_BASE
from ddqn_train2 import train_v2, restore

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="oracle", choices=["oracle", "raw_proxy", "filtered"])
ap.add_argument("--no-unc", action="store_true")
ap.add_argument("--no-regime", action="store_true")
ap.add_argument("--filter", default="imm", choices=["imm", "single"])
ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
ap.add_argument("--episodes", type=int, default=6000)            # was 1500 (curve was still improving)
ap.add_argument("--eval-episodes", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--imm-particles", type=int, default=2000)
ap.add_argument("--reward-scale", type=float, default=0.1)
ap.add_argument("--eps-decay-frac", type=float, default=0.3)      # was 0.4; leaves more greedy episodes
ap.add_argument("--update-every", type=int, default=4)            # try 2 or 1 for more gradient steps
ap.add_argument("--target-sync", type=int, default=500)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--explore-radius", type=int, default=None, help="local exploration around TWAP; None = uniform")
ap.add_argument("--val-every", type=int, default=200)
ap.add_argument("--n-val", type=int, default=60)
ap.add_argument("--tag", default=None)
a = ap.parse_args()
if a.filter == "single" and a.arm != "filtered":
    sys.exit("--filter single only makes sense with --arm filtered")

tag = a.tag or (f"v2_{a.arm}_unc{int(not a.no_unc)}_reg{int(not a.no_regime)}_{a.filter}_{a.obs_kind}_s{a.seed}"
                f"_r{a.explore_radius}")
env = make_env(EnvConfig(obs_mode=a.arm, use_uncertainty=not a.no_unc, use_regime=not a.no_regime,
                         n_regimes=1 if a.filter == "single" else 2, reward_scale=a.reward_scale),
               obs_kind=a.obs_kind, filter_regimes=a.filter, seed=a.seed,
               imm_params=IMMParams(n_particles=a.imm_particles), stationary_init=True)
print(f"[{tag}] obs_dim={env.obs_dim} n_actions={env.n_actions}")

cfg = DDQNConfig(eps_decay_frac=a.eps_decay_frac, update_every=a.update_every, target_sync=a.target_sync, lr=a.lr)
t0 = time.time()
agent, log, curve, best = train_v2(env, cfg, a.episodes, a.seed, val_every=a.val_every, n_val=a.n_val,
                                   explore_radius=a.explore_radius)
print(f"training+validation: {time.time() - t0:.0f}s   best validation checkpoint at episode {best['ep']} "
      f"(noise-free diff {best['score']:+.1f})")

test_seeds = range(EVAL_BASE, EVAL_BASE + a.eval_episodes)
twap = run_episodes(env, test_seeds, None)
od = os.path.join(HERE, "outputs", "stage3b"); os.makedirs(od, exist_ok=True)
report = {}
for name in ("last", "best_val"):
    if name == "best_val":
        agent.save(os.path.join(od, f"{tag}_last.npz"))
        restore(agent, best["snap"])
        agent.save(os.path.join(od, f"{tag}_bestval.npz"))
    res = run_episodes(env, test_seeds, agent)
    pf, pn = paired(res, twap, "IS"), paired(res, twap, "IS_decomp")
    report[name] = dict(paired_full=pf, paired_noise_free=pn, mult_by_regime=res["mult_by_regime"],
                        max_q_left=float(res["q_left"].max()))
    print(f"\nTEST ({a.eval_episodes} held-out worlds) -- {name}: TWAP {pf['base_mean']:.1f} | agent {pf['mean']:.1f} | "
          f"full diff {pf['diff']:+.1f} (se {pf['se']:.1f}) | noise-free diff {pn['diff']:+.1f} (se {pn['se']:.1f}) | "
          f"CVaR95 {pf['cvar95']:.0f} vs {pf['base_cvar95']:.0f}")
    print(f"   mean multiplier by TRUE regime (mean, n): {res['mult_by_regime']}   max unsold {res['q_left'].max():.2g}")

with open(os.path.join(od, f"{tag}.json"), "w") as f:
    json.dump(dict(tag=tag, args=vars(a), report=report, val_curve=curve, best_val_ep=best["ep"],
                   train_log=log[::20]), f, indent=1)
with open(os.path.join(od, f"{tag}_valcurve.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(curve[0].keys())); w.writeheader(); w.writerows(curve)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["ep"] for r in curve]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(ep, [r["val_diff_nf"] for r in curve], yerr=[r["val_se_nf"] for r in curve], fmt="o-", label="noise-free")
    ax.plot(ep, [r["val_diff_full"] for r in curve], "s--", alpha=0.6, label="full IS")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("training episodes"); ax.set_ylabel("greedy - TWAP, mean IS ($)  [below 0 = better]")
    ax.set_title(f"validation curve: {tag}"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(od, f"{tag}_valcurve.png"), dpi=130)
except Exception as e:
    print("plot skipped:", e)
print("wrote", od)