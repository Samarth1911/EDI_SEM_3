"""Train + evaluate ONE Double-DQN arm vs TWAP on held-out seeds. Run from EDI/validation. UNTESTED until run."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
from ddqn import DDQNConfig
from ddqn_train import train, run_episodes, paired, EVAL_BASE

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="filtered", choices=["oracle", "raw_proxy", "filtered"])
ap.add_argument("--no-unc", action="store_true")
ap.add_argument("--no-regime", action="store_true")
ap.add_argument("--filter", default="imm", choices=["imm", "single"])
ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
ap.add_argument("--episodes", type=int, default=1500)
ap.add_argument("--eval-episodes", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--imm-particles", type=int, default=2000)
ap.add_argument("--reward-scale", type=float, default=0.1)
ap.add_argument("--calm-start", action="store_true", help="disable stationary initial regime (3A default)")
ap.add_argument("--tag", default=None)
a = ap.parse_args()
if a.filter == "single" and a.arm != "filtered":
    sys.exit("--filter single only makes sense with --arm filtered")

tag = a.tag or f"{a.arm}_unc{int(not a.no_unc)}_reg{int(not a.no_regime)}_{a.filter}_{a.obs_kind}_s{a.seed}"
cfg_env = EnvConfig(obs_mode=a.arm, use_uncertainty=not a.no_unc, use_regime=not a.no_regime,
                    n_regimes=1 if a.filter == "single" else 2, reward_scale=a.reward_scale)
env = make_env(cfg_env, obs_kind=a.obs_kind, filter_regimes=a.filter, seed=a.seed,
               imm_params=IMMParams(n_particles=a.imm_particles), stationary_init=not a.calm_start)
print(f"[{tag}] obs_dim={env.obs_dim} n_actions={env.n_actions} imm_particles={a.imm_particles}")

t0 = time.time()
agent, log = train(env, DDQNConfig(), a.episodes, a.seed)
print(f"training: {time.time()-t0:.0f}s")

seeds = range(EVAL_BASE, EVAL_BASE + a.eval_episodes)
twap, res = run_episodes(env, seeds, None), run_episodes(env, seeds, agent)
p_full, p_dec = paired(res, twap, "IS"), paired(res, twap, "IS_decomp")
print(f"\nheld-out episodes: {a.eval_episodes}   max unsold inventory: {res['q_left'].max():.3g}")
print(f"TWAP  mean IS {p_full['base_mean']:.1f}   CVaR95 {p_full['base_cvar95']:.1f}")
print(f"agent mean IS {p_full['mean']:.1f}   CVaR95 {p_full['cvar95']:.1f}")
print(f"paired diff (agent - TWAP), full IS     : {p_full['diff']:+.1f}  (se {p_full['se']:.1f})")
print(f"paired diff (agent - TWAP), noise-free  : {p_dec['diff']:+.1f}  (se {p_dec['se']:.1f})")
print("mean chosen multiplier by TRUE regime (mean, n):", res["mult_by_regime"])

od = os.path.join(HERE, "outputs", "stage3b"); os.makedirs(od, exist_ok=True)
agent.save(os.path.join(od, f"{tag}.npz"))
with open(os.path.join(od, f"{tag}.json"), "w") as f:
    json.dump(dict(tag=tag, args=vars(a), paired_full=p_full, paired_noise_free=p_dec,
                   mult_by_regime=res["mult_by_regime"], max_q_left=float(res["q_left"].max()),
                   train_log=log[::10]), f, indent=1)
print("wrote", od)