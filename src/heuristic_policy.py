"""
heuristic_policy.py
--------------------
PRELIMINARY HEURISTIC showing the belief signal carries useful information.
The actual contribution of the project is the Double DQN agent trained on this
belief -- that is NEXT PHASE, not this one.

The rule (deliberately simple, hand-designed, 4 knobs):

    base  = q_remaining / (intervals_left)                 # "remaining-TWAP" rate
    a_eff = alpha_hat + z_unc * alpha_std                  # risk-adjusted impact
    s     = clip( (alpha_ref / a_eff) ** gamma, s_min, s_max )
    s_t   = 1 + (s - 1) * taper(t)                         # deviations fade toward the end
    v_t   = base * s_t          (last interval: sell whatever remains)

Slow down when estimated impact OR uncertainty is high, speed up when both
are low. `taper` exists because a slowed-down schedule must be dumped in the
last interval, and temporary-impact cost is convex in trade size.

Three policies are compared, all through the same harness:
  twap     : q0/N every interval, ignores the market state entirely.
  belief   : the rule above fed (alpha_hat, alpha_std) from the particle
             filter, which only sees the noisy proxies.
  oracle   : the SAME rule fed the TRUE alpha with zero uncertainty.
             NOTE: this is "the rule with perfect information", NOT an
             optimal-control benchmark (that is the Barger-Lorig comparison,
             next phase).
  belief_mean_only : the rule with z_unc=0, i.e. uses the filter's mean but
             ignores its uncertainty (ablation of the project's core claim).

STATUS: milestone-1, preliminary. Knobs are tuned on seeds disjoint from the
evaluation seeds (see validation/experiments/run_policy_comparison.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from simulator import MarketSimulator, SimParams
from observation_model import ObservationModel
from particle_filter import ParticleFilter, PFParams


@dataclass
class HeuristicParams:
    gamma: float = 1.0        # sensitivity of speed to impact level
    z_unc: float = 1.0        # weight on posterior std (0 => ignore uncertainty)
    s_min: float = 0.2        # slowest allowed multiple of the base rate
    s_max: float = 2.0        # fastest allowed multiple
    taper_power: float = 1.0  # deviations scale with ((N-1-t)/N)**taper_power; 0 => no taper


def heuristic_velocity(t, N, q, alpha_hat, alpha_std, alpha_ref, hp: HeuristicParams) -> float:
    if q <= 0:
        return 0.0
    if t >= N - 1:
        return q
    base = q / (N - t)
    a_eff = max(alpha_hat + hp.z_unc * alpha_std, 1e-12)
    s = float(np.clip((alpha_ref / a_eff) ** hp.gamma, hp.s_min, hp.s_max))
    taper = ((N - 1 - t) / N) ** hp.taper_power if hp.taper_power > 0 else 1.0
    s_t = 1.0 + (s - 1.0) * taper
    return min(base * s_t, q)


def run_policy_episode(kind, seed, shock_step=None, shock_mult=3.0, hp=None,
                       sim_params=None, pf_params=None, record=False):
    """One episode under `kind` in {'twap','belief','belief_mean_only','oracle'}.
    Identical `seed` => identical market noise across kinds (common random numbers)."""
    hp = hp or HeuristicParams()
    sim = MarketSimulator(sim_params, seed=seed)
    p = sim.p
    gen = sim.run_stepwise(shock_step=shock_step, shock_jump_alpha_mult=shock_mult, seed=seed)
    use_pf = kind in ("belief", "belief_mean_only")
    if use_pf:
        om = ObservationModel(seed=seed + 10_000)              # own RNG: does not touch sim noise
        pf = ParticleFilter(p, om.p, pf_params, seed=seed + 20_000)
        if kind == "belief_mean_only":
            hp = replace(hp, z_unc=0.0)
    N = p.N
    rec = {"a_hat": [], "a_std": [], "a_true": [], "v": []}

    state = next(gen)
    while True:
        t, q = state["t"], state["q"]
        if kind == "twap":
            v, a_hat, a_std = p.q0 / N, np.nan, np.nan
        elif kind == "oracle":
            a_hat, a_std = state["alpha"], 0.0
            v = heuristic_velocity(t, N, q, a_hat, a_std, p.theta_alpha, hp)
        else:  # belief-based: must NOT read state['kappa'] / state['alpha'] except to GENERATE the noisy proxies
            pf.step(om.observe_step(state["kappa"], state["alpha"]))
            a_hat, a_std = pf.history["alpha_mean"][-1], pf.history["alpha_std"][-1]
            v = heuristic_velocity(t, N, q, a_hat, a_std, p.theta_alpha, hp)
        if record:
            rec["a_hat"].append(a_hat); rec["a_std"].append(a_std)
            rec["a_true"].append(state["alpha"]); rec["v"].append(v)
        try:
            state = gen.send(v)
        except StopIteration as stop:
            ep = stop.value
            break
    return (ep, rec) if record else ep


if __name__ == "__main__":
    import time
    t0 = time.time()
    for kind in ["twap", "belief", "belief_mean_only", "oracle"]:
        ep, rec = run_policy_episode(kind, seed=5, shock_step=25, record=True)
        print(f"{kind:17s} IS={ep.IS:9.1f}  final inv={ep.q[-1]:.2f}  max v={ep.v.max():7.0f}  v[24..28]={np.round(ep.v[24:29]).astype(int)}")
    print(f"({time.time()-t0:.1f}s for 4 episodes)")