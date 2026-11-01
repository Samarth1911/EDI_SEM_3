"""
test_ce_misspec.py -- Stage 4A item 4 (first look, NO simulation). NOT RUN by its author.
CE plan v_s ~ 1/E[a_s] built from a MISESTIMATED alpha forecast (lam*ls, theta*ts), evaluated under the TRUE mean forecast.
Ignores kappa, regimes and the filter: it measures forecast-parameter sensitivity of the alpha plan only.
excess = cost(plan_mis)/cost(plan_true) - 1 ; kept = share of CE's gain over TWAP retained (1 = no loss, <0 = worse than TWAP).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from simulator import SimParams
from regime_simulator import RegimeParams
from barger_lorig import params_from_regimes

sp = SimParams(); N, tau = sp.N, sp.tau
P = params_from_regimes(sp, RegimeParams.default())
SETS = {"calm": P[0], "stressed": P[1]}
SCALES = (0.7, 1.0, 1.3)


def path(a, lam, th, n):
    return th + (a - th) * (1 - lam * tau) ** np.arange(n)


def plan(ea):
    w = 1.0 / ea
    return w / w.sum()


def cost(v, ea):
    return float((ea * v * v).sum())


for name, (la, ta, lk, tk) in SETS.items():
    print(f"\n=== {name}: lam_a={la:.3g} theta_a={ta:.3g}; horizon start n={N} ===")
    print(f"{'eps':>5s} {'lam x':>6s} {'theta x':>8s} {'excess':>8s} {'kept':>7s}")
    for eps in (-0.5, 0.5, 1.0):
        a = ta * (1 + eps)
        ea = path(a, la, ta, N)
        c_opt, c_tw = cost(plan(ea), ea), cost(np.full(N, 1.0 / N), ea)
        for ls in SCALES:
            for ts in SCALES:
                v = plan(path(a, la * ls, ta * ts, N))
                c = cost(v, ea)
                kept = (c_tw - c) / (c_tw - c_opt) if c_tw > c_opt else float("nan")
                print(f"{eps:5.2f} {ls:6.2f} {ts:8.2f} {c / c_opt - 1:8.4f} {kept:7.3f}")