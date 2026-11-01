"""
diag_ce_ref.py -- CE-reference script (NOT your older diag_headroom.py, which has different arguments).
NO TRAINING. UNTESTED until you run it. Put in EDI/validation and run, e.g.:
    python diag_ce_ref.py --grid "0.579,0.694,0.833,1.0,1.2,1.44,1.728" --n 200 --seed0 900000
    python diag_ce_ref.py --grid "0.25,0.5,0.75,1.0,1.25,1.5,2.0" --n 200 --seed0 900000   (cross-check vs your -90.8)

Question: how much can ANY policy beat TWAP in this world, and can the env's discrete multiplier grid
express it?  Everything is measured directly on RegimeSwitchingSimulator (same world construction as
env_factory.make_env with stationary_init=True), so it does not depend on belief_mdp_env internals,
except that the multiplier grid is read from EnvConfig().action_multipliers (or pass --grid).

Policies
  twap            q0/N every step.
  heur_oracle     existing heuristic_velocity fed TRUE alpha, z_unc=0 (knobs tuned for single-regime world).
  rhce_oracle     receding-horizon certainty-equivalent, CAUSAL: knows true alpha_t, true regime_t and the true
                  regime model; forecasts E[alpha_s] forward (CIR drift + Markov regime), plans v_s ~ 1/E[alpha_s]
                  over the remaining steps, executes the first step, re-plans. Ignores permanent impact and
                  Jensen effects -> a reasonable reference, NOT optimal.
  rhce_grid       rhce_oracle with its multiplier snapped to the nearest env grid value (log distance).
                  ASSUMES the env multiplier scales q_remaining/(N-t) and the last step sells everything.
  const_min       smallest grid multiplier every step, forced dump at the end (what the collapsed agent seems to do).
  hindsight_nf    NON-CAUSAL lower bound: knows the whole true alpha/kappa path, noise-free cost only.

Cost model (derived from simulator.py step math, verified below on every TWAP episode):
  IS = sum_t alpha_t v_t^2 + sum_t v_t sum_{i<t} kappa_i v_i - sum_t v_t*noise_t      (needs sum v = q0)
  'NF' = the first two terms.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from scipy.optimize import minimize
from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from heuristic_policy import heuristic_velocity, HeuristicParams

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed0", type=int, default=900_000)
ap.add_argument("--grid", default=None, help="comma-separated multipliers; default: read from EnvConfig")
a = ap.parse_args()

sp = SimParams()
_rp0 = RegimeParams.default()
rp = RegimeParams(specs=_rp0.specs, P=_rp0.P.copy(), init_probs=_rp0.stationary_distribution())
N, tau = sp.N, sp.tau


# ---------------------------------------------------------------- cost model
def build_H(alpha, kappa):
    idx = np.arange(N)
    H = 0.5 * kappa[np.minimum.outer(idx, idx)]      # off-diagonal: 0.5*kappa_{min(i,t)}
    H[idx, idx] = alpha[:N]                          # diagonal: alpha_t only
    return H


def nf_cost(v, alpha, kappa):
    return float(v @ build_H(alpha, kappa) @ v)


def identity_residual(ep):
    v = ep.v
    perm = np.concatenate([[0.0], np.cumsum(ep.kappa[:N] * v)[:-1]])
    noise = ep.S[:N] - (sp.S0 - perm)
    return ep.IS - (nf_cost(v, ep.alpha, ep.kappa) - float(np.sum(v * noise)))


def hindsight_cost(alpha, kappa, twap_cost):
    Hs = build_H(alpha, kappa) * (sp.q0 / N) ** 2    # variables x = v/(q0/N), sum x = N
    res = minimize(lambda x: x @ Hs @ x, np.ones(N), jac=lambda x: 2.0 * Hs @ x,
                   bounds=[(0.0, None)] * N,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - N, "jac": lambda x: np.ones(N)}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    x = np.clip(res.x, 0.0, None)
    x *= N / x.sum()
    c = float(x @ Hs @ x)
    return min(c, twap_cost), bool(res.success)


# ---------------------------------------------------------------- policies
K = int(rp.n_regimes)
P = np.asarray(rp.P, dtype=float)
_eff = [rp.effective_cir(sp, r) for r in range(K)]
LAM = np.array([e["lambda_alpha"] for e in _eff], dtype=float)
TH = np.array([e["theta_alpha"] for e in _eff], dtype=float)


def forecast_alpha(alpha_t, reg_t, n_ahead):
    """E[alpha_{t+k}], k = 0..n_ahead-1, given alpha_t and regime[t] (true model, no truncation/jumps)."""
    out = np.empty(n_ahead)
    pi = np.zeros(K); pi[reg_t] = 1.0
    m = pi * alpha_t                                  # m_j = E[alpha; regime = j]
    out[0] = alpha_t
    for k in range(1, n_ahead):
        pi_next = pi @ P
        m = (1.0 - LAM * tau) * (m @ P) + LAM * TH * tau * pi_next
        pi = pi_next
        out[k] = m.sum()
    return out


def v_rhce(st):
    t, q = st["t"], st["q"]
    if t >= N - 1:
        return q
    w = 1.0 / np.maximum(forecast_alpha(st["alpha"], int(st["regime"]), N - t), 1e-12)
    return q * w[0] / w.sum()


def load_grid():
    if a.grid:
        return np.sort(np.array([float(x) for x in a.grid.split(",")]))
    try:
        from belief_mdp_env import EnvConfig
        return np.sort(np.array(EnvConfig(obs_mode="oracle").action_multipliers, dtype=float))
    except Exception as e:                             # noqa
        print(f"[warn] could not read action grid ({e}); grid policies skipped. Pass --grid.")
        return None


GRID = load_grid()


def snap(v, q, t):
    if t >= N - 1:
        return q
    base = q / (N - t)
    m = max(v / base, 1e-9)
    g = GRID[np.argmin(np.abs(np.log(GRID) - np.log(m)))]
    return g * base


hp = HeuristicParams(z_unc=0.0)
policies = {
    "twap": lambda st: sp.q0 / N,
    "heur_oracle": lambda st: heuristic_velocity(st["t"], N, st["q"], st["alpha"], 0.0, sp.theta_alpha, hp),
    "rhce_oracle": v_rhce,
}
if GRID is not None:
    print("action grid:", GRID)
    policies["rhce_grid"] = lambda st: snap(v_rhce(st), st["q"], st["t"])
    policies["const_min"] = lambda st: snap(0.0 + GRID.min() * st["q"] / (N - st["t"]), st["q"], st["t"])


def run_episode(seed, chooser):
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    gen = sim.run_stepwise(seed=seed)
    state = next(gen)
    while True:
        try:
            state = gen.send(float(chooser(state)))
        except StopIteration as stop:
            return stop.value


def cvar95(x):
    k = max(1, int(np.ceil(0.05 * len(x))))
    return float(np.sort(x)[-k:].mean())


def se(x):
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


# ---------------------------------------------------------------- run
seeds = list(range(a.seed0, a.seed0 + a.n))
res = {name: {"IS": [], "NF": []} for name in policies}
hind, hind_fail, resid = [], 0, []
for s in seeds:
    for name, ch in policies.items():
        ep = run_episode(s, ch)
        assert abs(ep.q[-1]) < 1e-6 * sp.q0, f"{name}: inventory not liquidated (seed {s})"
        res[name]["IS"].append(ep.IS)
        res[name]["NF"].append(nf_cost(ep.v, ep.alpha, ep.kappa))
        if name == "twap":
            resid.append(abs(identity_residual(ep)) / abs(ep.IS))
            c, ok = hindsight_cost(ep.alpha, ep.kappa, res["twap"]["NF"][-1])
            hind.append(c); hind_fail += (not ok)
res = {k: {m: np.array(v) for m, v in d.items()} for k, d in res.items()}
hind = np.array(hind)

print(f"\ncost-model identity check on {len(resid)} TWAP episodes: max rel. residual = {max(resid):.2e} "
      f"(want < 1e-9; if not, the model above is wrong and NOTHING below is usable)")
print(f"SLSQP non-converged hindsight solves: {hind_fail}/{len(hind)}")
tw = res["twap"]
print(f"\nTWAP: mean IS {tw['IS'].mean():.1f}  CVaR95 {cvar95(tw['IS']):.1f}   "
      f"(env eval, n=200 from 900000, gave 4852.8 / 6978.2: equality means my worlds == env worlds)")
print(f"\n{'policy':13s} {'meanIS':>8s} {'dIS vs TWAP (se)':>20s} {'meanNF':>8s} {'dNF vs TWAP (se)':>20s} {'NF/TWAP':>8s} {'beats TWAP':>10s}")
for name, r in res.items():
    d, dn = r["IS"] - tw["IS"], r["NF"] - tw["NF"]
    print(f"{name:13s} {r['IS'].mean():8.1f} {d.mean():+12.1f} ({se(d):5.1f}) {r['NF'].mean():8.1f} "
          f"{dn.mean():+12.1f} ({se(dn):5.1f}) {r['NF'].mean()/tw['NF'].mean():8.3f} {np.mean(dn < 0):10.2f}")
dn = hind - tw["NF"]
print(f"{'hindsight_nf':13s} {'':>8s} {'':>20s} {hind.mean():8.1f} {dn.mean():+12.1f} ({se(dn):5.1f}) "
      f"{hind.mean()/tw['NF'].mean():8.3f} {np.mean(dn < 0):10.2f}   <- non-causal ceiling")
print("\nHow to read: hindsight_nf is the ceiling on ANY policy's noise-free gain; rhce_oracle is what a sensible "
      "causal oracle achieves. If rhce_grid ~ rhce_oracle, the grid is not the bottleneck.")