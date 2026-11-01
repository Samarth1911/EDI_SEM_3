"""
stage3c_bonus.py -- Stage 3C (Person 3): uncertainty-driven optimism bonus tied to the filter's posterior std.
NOT RUN by its author. Self-test:  python stage3c_bonus.py

WHAT IT IS
    Optimism in the face of uncertainty about the COST PARAMETERS. The env's "decomp" cost of trading v is
        alpha*v^2 + kappa*v*(q - v).
    If alpha and kappa are only known up to the posterior stds sigma_a, sigma_k, an optimistic agent
    credits action a with the cost it would save if the parameters were one std lower:
        bonus_a = (sigma_a*v_a^2 + sigma_k*v_a*(q - v_a)) * 1e4/(S0*q0) * reward_scale     (reward units)
    It is zero when the filter is certain and grows with the trade size, i.e. it favours large trades
    exactly when the belief says impact might be lower than estimated.

HOW TO USE (selection only -- do NOT put it in the TD target)
    q_sel = q_values(obs) + beta_at(progress, beta) * bonus_vector(env)      # then argmax / epsilon-greedy
    The learned Q stays an estimate of the true return; evaluation uses the plain greedy action.

HONEST CAVEATS
    * The agent's trades do not change the observations or the filter's belief in this simulator, so there is
      NO information to be gained by acting. The bonus therefore is NOT information gathering; it is a
      heuristic to diversify training-time action choices where cost is most uncertain. Whether it helps is
      an empirical question -- that is what ablation (b) measures.
    * beta is untuned (sweep e.g. 0.5, 1, 2). unc_source="pooled" includes between-regime spread in sigma;
      "within" uses the probability-weighted per-regime stds (Stage 2 flags that the pooled std mixes both).
    * On spread_hawkes data the filter was overconfident (Stage 2: 90% coverage 0.54), so sigma is too small there.
"""
from __future__ import annotations

import numpy as np


def beta_at(progress: float, beta: float, anneal_frac: float = 0.8) -> float:
    """Linear anneal to zero over the first `anneal_frac` of training (progress in [0,1])."""
    if anneal_frac <= 0:
        return float(beta)
    return float(beta) * max(0.0, 1.0 - float(progress) / anneal_frac)


def bonus_vector(env, unc_source: str = "pooled", include_kappa: bool = True) -> np.ndarray:
    """(n_actions,) bonus in reward units for the CURRENT state of a BeliefMDPEnv. Zeros if there is no belief
    (oracle / raw_proxy arms). Reads env.belief, env.cfg, env.p, env.N, env._s as in belief_mdp_env.py."""
    A = env.n_actions
    b = getattr(env, "belief", None)
    if b is None:
        return np.zeros(A)
    c, p, s = env.cfg, env.p, env._s
    t, q, N = int(s["t"]), float(s["q"]), int(env.N)
    m = np.asarray(c.action_multipliers, dtype=float)
    v = np.full(A, q) if t >= N - 1 else np.minimum(m * q / (N - t), q)
    if unc_source == "within" and getattr(b, "per_regime", None):
        sa = float(sum(r.prob * r.alpha_std for r in b.per_regime))
        sk = float(sum(r.prob * r.kappa_std for r in b.per_regime))
    elif unc_source in ("pooled", "within"):
        sa, sk = float(b.alpha_std), float(b.kappa_std)
    else:
        raise ValueError("unc_source must be 'pooled' or 'within'")
    cost_unc = sa * v * v + (sk * v * (q - v) if include_kappa else 0.0)
    return cost_unc * 1e4 / (p.S0 * p.q0) * c.reward_scale


def _selftest():
    from types import SimpleNamespace as NS
    cfg = NS(action_multipliers=(0.25, 0.5, 1.0, 2.0), reward_scale=0.1)
    p = NS(S0=10.0, q0=20000.0)
    rb = [NS(prob=0.3, alpha_std=1e-5, kappa_std=1e-6), NS(prob=0.7, alpha_std=3e-5, kappa_std=2e-6)]
    b = NS(alpha_std=4e-5, kappa_std=3e-6, per_regime=rb)
    env = NS(n_actions=4, cfg=cfg, p=p, N=50, _s={"t": 10, "q": 12000.0}, belief=b)
    errs = []
    bv = bonus_vector(env)
    q, base = 12000.0, 12000.0 / 40
    v = np.minimum(np.array([0.25, 0.5, 1.0, 2.0]) * base, q)
    exp = (4e-5 * v ** 2 + 3e-6 * v * (q - v)) * 1e4 / (10.0 * 20000.0) * 0.1
    if not np.allclose(bv, exp, rtol=1e-12):
        errs.append("pooled bonus != hand computation")
    if np.any(bv < 0):
        errs.append("negative bonus")
    w = bonus_vector(env, "within")
    exp_w = ((0.3 * 1e-5 + 0.7 * 3e-5) * v ** 2 + (0.3 * 1e-6 + 0.7 * 2e-6) * v * (q - v)) * 1e4 / 2e5 * 0.1
    if not np.allclose(w, exp_w, rtol=1e-12):
        errs.append("within bonus != hand computation")
    env.belief = None
    if np.any(bonus_vector(env) != 0):
        errs.append("bonus not zero without belief")
    if not (beta_at(0.0, 1.0) == 1.0 and beta_at(0.8, 1.0) == 0.0 and beta_at(1.0, 1.0) == 0.0):
        errs.append("beta schedule")
    print("stage3c_bonus selftest:", "PASS" if not errs else f"FAIL {errs}")


if __name__ == "__main__":
    _selftest()