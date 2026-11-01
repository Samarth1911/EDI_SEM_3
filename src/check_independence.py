import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
env = make_env(EnvConfig(obs_mode="filtered", reward_scale=0.1), obs_kind="gaussian",
               filter_regimes="imm", seed=0, imm_params=IMMParams(n_particles=300), stationary_init=True)
def roll(a):
    env.reset(seed=900001); done = False
    while not done:
        _, _, done, _, _ = env.step(a)
    return env.get_episode()["steps"]
A, B = roll(env.twap_action), roll(env.n_actions - 1)
print("actions differ (v):", max(abs(x["v"] - y["v"]) for x, y in zip(A, B)) > 0)
for k in ["true_alpha", "true_kappa", "true_regime", "alpha_mean", "alpha_std", "kappa_mean", "kappa_std"]:
    print(f"{k:12s} max|diff| =", max(abs(x[k] - y[k]) for x, y in zip(A, B)))
print("regime_probs max|diff| =", max(abs(a - b) for x, y in zip(A, B) for a, b in zip(x["regime_probs"], y["regime_probs"])))
