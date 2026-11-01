"""
ens_vs_ce.py -- paired comparison of an arm's ENSEMBLE against the certainty-equivalent (CE) benchmark on the SAME worlds.
NO TRAINING. UNTESTED until you run it. Run from EDI/validation (venv active), after score_ensemble.py has saved arrays:
    python ens_vs_ce.py --arm oracle
    python ens_vs_ce.py --arm filt_full_bonus

CE here = exact-forecast certainty-equivalent receding-horizon plan with TRUE alpha and regime (same as run_stage4_benchmarks /
diag_ce_ref), continuous and snapped to the 7-point default grid. For filtered arms this gives CE an information advantage
(true state), so the comparison is conservative against the agent; the fair belief-based comparator is ce_filtered_grid in
run_stage4_benchmarks (fresh worlds only).
Self-check: my TWAP noise-free cost must equal the saved env TWAP cost world by world (max abs diff < 1e-6); otherwise the
pairing or the metric is wrong and nothing below is printed.
"""
import argparse, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="oracle")
ap.add_argument("--grid", default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
a = ap.parse_args()
GRID = np.array([float(x) for x in a.grid.split(",")])
z = np.load(os.path.join(HERE, "outputs", "stage3c", f"ens_nf_{a.arm}.npz"))

sp = SimParams(); N, tau = sp.N, sp.tau
rp = RegimeParams.default(); rp.init_probs = rp.stationary_distribution()
K = int(rp.n_regimes); Pm = np.asarray(rp.P, float).reshape(K, K)
_eff = [rp.effective_cir(sp, r) for r in range(K)]
LAM = np.array([e["lambda_alpha"] for e in _eff], float); TH = np.array([e["theta_alpha"] for e in _eff], float)


def forecast(alpha_t, reg_t, n_ahead):
    out = np.empty(n_ahead); pi = np.zeros(K); pi[reg_t] = 1.0; m = pi * alpha_t; out[0] = alpha_t
    for k in range(1, n_ahead):
        pi_next = pi @ Pm
        m = (1.0 - LAM * tau) * (m @ Pm) + LAM * TH * tau * pi_next
        pi = pi_next; out[k] = m.sum()
    return out


def ce_policy(grid):
    def f(st):
        t, q = st["t"], st["q"]
        if q <= 0:
            return 0.0
        if t >= N - 1:
            return q
        w = 1.0 / np.maximum(forecast(st["alpha"], int(st["regime"]), N - t), 1e-12)
        m = float(w[0] * (N - t) / w.sum())
        if grid is not None:
            m = float(grid[np.argmin(np.abs(np.log(grid) - np.log(max(m, 1e-9))))])
        return min(m * q / (N - t), q)
    return f


def run(seed, pol):
    sim = RegimeSwitchingSimulator(sp, rp, seed=int(seed))
    gen = sim.run_stepwise(seed=int(seed))
    state = next(gen)
    while True:
        try:
            state = gen.send(float(pol(state)))
        except StopIteration as stop:
            return stop.value


def nf(ep):
    v = np.asarray(ep.v, float)
    qb = sp.q0 - np.concatenate([[0.0], np.cumsum(v)[:-1]])
    return float(np.sum(ep.alpha[:len(v)] * v ** 2 + ep.kappa[:len(v)] * v * (qb - v)))


def stat(d):
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    return f"{d.mean():+7.1f} (se {se:4.1f}, t {d.mean() / se:+5.2f})"


for blk in ("test", "fresh"):
    seeds = z[f"{blk}_seeds"]
    twap_env, ens, mem = z[f"{blk}_twap"], z[f"{blk}_ens"], z[f"{blk}_mem"]
    twap_me = np.array([nf(run(s, lambda st: 0.0 if st["q"] <= 0 else st["q"] / (N - st["t"]))) for s in seeds])
    gap = float(np.abs(twap_me - twap_env).max())
    assert gap < 1e-6, f"{blk}: TWAP cost mismatch {gap:.2e}: pairing/metric wrong; do not trust any comparison"
    ce_c = np.array([nf(run(s, ce_policy(None))) for s in seeds])
    ce_g = np.array([nf(run(s, ce_policy(GRID))) for s in seeds])
    print(f"\n===== {a.arm}, {blk} worlds ({len(seeds)}; TWAP cost check max |diff| = {gap:.1e}) =====")
    print(f"  ensemble   - TWAP        : {stat(ens - twap_env)}")
    print(f"  mean member- TWAP        : {stat(mem.mean(axis=0) - twap_env)}")
    print(f"  CE grid    - TWAP        : {stat(ce_g - twap_env)}")
    print(f"  CE contin. - TWAP        : {stat(ce_c - twap_env)}")
    print(f"  ensemble   - CE grid     : {stat(ens - ce_g)}    (negative = ensemble cheaper)")
    print(f"  ensemble   - CE contin.  : {stat(ens - ce_c)}")
    print(f"  mean member- CE grid     : {stat(mem.mean(axis=0) - ce_g)}")
print("\nCE uses TRUE state. Only world-sampling inference: one ensemble of 5 trained agents per arm.")