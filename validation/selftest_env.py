"""Stage 3A self-test on the REAL stack. Run from EDI/validation:  python selftest_env.py"""
import os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env


def run_twap(env, seed):
    obs, _ = env.reset(seed)
    done = False
    while not done:
        assert np.all(np.isfinite(obs)), "non-finite obs"
        obs, r, done, _, info = env.step(env.twap_action)
    return env.get_episode()


print("== 1. shapes / accounting for all 12 (mode x unc x regime) configs ==")
for mode, unc, reg in itertools.product(("oracle", "raw_proxy", "filtered"), (True, False), (True, False)):
    env = make_env(EnvConfig(obs_mode=mode, use_uncertainty=unc, use_regime=reg), seed=1)
    ep = run_twap(env, seed=11)
    v = np.array([s["v"] for s in ep["steps"]])
    assert abs(v.sum() - ep["q0"]) < 1e-6 * ep["q0"], "inventory not fully sold"
    assert np.allclose(v, ep["q0"] / ep["N"]), "multiplier 1.0 must reproduce TWAP"
    assert abs(ep["IS_from_steps"] - ep["IS"]) < 1e-8 * abs(ep["IS"]), "step IS != Episode.IS"
    print(f"  {mode:9s} unc={unc!s:5} reg={reg!s:5} obs_dim={env.obs_dim:2d} IS={ep['IS']:9.1f}  OK")

print("== 2. 'decomp' cost vs exact IS over 20 seeds (oracle mode, TWAP) ==")
env = make_env(EnvConfig(obs_mode="oracle"), seed=2)
d = []
for s in range(20):
    ep = run_twap(env, seed=100 + s)
    d.append(ep["IS"] - ep["sum_decomp"])
d = np.array(d)
z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print(f"  IS - sum(decomp): mean={d.mean():.2f}  std={d.std(ddof=1):.2f}  z={z:+.2f}  "
      f"({'OK' if abs(z) < 3.5 else 'WARN: decomp not unbiased -> check regime simulator price step'})")

print("== 3. seed reproducibility (filtered) ==")
env = make_env(EnvConfig(obs_mode="filtered"), seed=3)
a, b = run_twap(env, 7), run_twap(env, 7)
same = np.isclose(a["IS"], b["IS"]) and np.isclose(a["steps"][10]["alpha_mean"], b["steps"][10]["alpha_mean"])
print(f"  IS: {a['IS']:.3f} vs {b['IS']:.3f}; alpha_mean[10]: {a['steps'][10]['alpha_mean']:.3e} vs "
      f"{b['steps'][10]['alpha_mean']:.3e} -> {'OK' if same else 'FAIL: reset() does not fully reseed'}")
print("3A self-test finished")