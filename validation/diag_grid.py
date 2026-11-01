"""
diag_grid.py -- Stage 3 revisit, step 2. NO TRAINING, simulator only. UNTESTED until you run it.
Run from EDI/validation (venv active):   python diag_grid.py            (300 design worlds, seeds 1,500,000+i)

Why: diag_agent_vs_ce.py showed the agents' trade sizes track the certainty-equivalent (CE) plan but are noisier and
over-reactive, while the CE multiple varies only about +-10% around 1 and the current grid step near 1 is 0.25.
This script measures how much of the CE gain survives snapping to candidate grids, on DESIGN worlds that nothing else has
used (train 100000*(s+1), val 800000+, test 900000+, fresh 1000000..1000499). Pick the grid here, never on test/fresh worlds.

CE = exact-forecast certainty-equivalent with TRUE alpha and regime (best-case information, receding horizon);
snapping = nearest grid multiple in log distance, applied to the multiple of remaining-TWAP q/(N-t); last step sells the rest.
This is an UPPER BOUND on what a grid can deliver. It does not predict what a DDQN will learn: finer grids also shrink
the gaps between actions, which is what the DDQN struggles with. Only the training experiment can answer that.
"""
import argparse, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed0", type=int, default=1_500_000)
ap.add_argument("--grids", default=None, help='optional extra grids, e.g. "mine:0.5,0.8,1,1.2,1.6;other:0.7,1,1.4"')
a = ap.parse_args()
assert a.seed0 >= 1_100_000, "use design worlds (>= 1,100,000), not the train/val/test/fresh blocks"


def logg(r, k):
    return tuple(round(r ** j, 3) for j in range(-k, k + 1))


GRIDS = {
    "G7_current": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    "G7_log1.15": logg(1.15, 3),
    "G7_log1.2": logg(1.2, 3),
    "G9_log1.1": logg(1.1, 4),
    "G9_log1.15": logg(1.15, 4),
    "G11_log1.1": logg(1.1, 5),
}
if a.grids:
    for part in a.grids.split(";"):
        nm, vals = part.split(":")
        GRIDS[nm.strip()] = tuple(sorted(float(x) for x in vals.split(",")))
for nm, g in GRIDS.items():
    assert 1.0 in g, f"grid {nm} must contain 1.0 (TWAP)"

sp = SimParams(); N, tau = sp.N, sp.tau
rp = RegimeParams.default(); rp.init_probs = rp.stationary_distribution()     # same worlds as Stage 3 (stationary start)
K = int(rp.n_regimes); Pm = np.asarray(rp.P, float).reshape(K, K)
_eff = [rp.effective_cir(sp, r) for r in range(K)]
LAM = np.array([e["lambda_alpha"] for e in _eff], float); TH = np.array([e["theta_alpha"] for e in _eff], float)
MREC = []


def forecast(alpha_t, reg_t, n_ahead):
    out = np.empty(n_ahead); pi = np.zeros(K); pi[reg_t] = 1.0; m = pi * alpha_t; out[0] = alpha_t
    for k in range(1, n_ahead):
        pi_next = pi @ Pm
        m = (1.0 - LAM * tau) * (m @ Pm) + LAM * TH * tau * pi_next
        pi = pi_next; out[k] = m.sum()
    return out


def make_ce(grid=None, record=False):
    g = None if grid is None else np.asarray(grid, float)

    def f(st):
        t, q = st["t"], st["q"]
        if q <= 0:
            return 0.0
        if t >= N - 1:
            return q
        w = 1.0 / np.maximum(forecast(st["alpha"], int(st["regime"]), N - t), 1e-12)
        m = float(w[0] * (N - t) / w.sum())
        if record:
            MREC.append(m)
        if g is not None:
            m = float(g[np.argmin(np.abs(np.log(g) - np.log(max(m, 1e-9))))])
        return min(m * q / (N - t), q)
    return f


def run(seed, pol):
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    gen = sim.run_stepwise(seed=seed)
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


def cvar95(x):
    k = max(1, int(math.ceil(0.05 * len(x))))
    return float(np.sort(x)[-k:].mean())


def se(d):
    return float(d.std(ddof=1) / math.sqrt(len(d)))


pols = {"twap": lambda st: 0.0 if st["q"] <= 0 else st["q"] / (N - st["t"]), "CE_continuous": make_ce(record=True)}
for nm, g in GRIDS.items():
    pols["CE@" + nm] = make_ce(g)
IS = {k: [] for k in pols}; NF = {k: [] for k in pols}
for i in range(a.n):
    for k, f in pols.items():
        ep = run(a.seed0 + i, f)
        assert abs(ep.q[-1]) < 1e-6 * sp.q0, f"{k}: not liquidated"
        IS[k].append(ep.IS); NF[k].append(nf(ep))
IS = {k: np.array(v) for k, v in IS.items()}; NF = {k: np.array(v) for k, v in NF.items()}
m = np.array(MREC)
print(f"{a.n} design worlds (seeds {a.seed0}..{a.seed0 + a.n - 1}); TWAP mean IS {IS['twap'].mean():.1f}")
print("CE multiple percentiles [1,5,25,50,75,95,99]:", np.round(np.percentile(m, [1, 5, 25, 50, 75, 95, 99]), 3))
cont = (NF["CE_continuous"] - NF["twap"]).mean()
print(f"\n{'policy':18s} {'k':>3s} {'d_noisefree (se)':>18s} {'share of continuous gain':>25s} {'CVaR95-TWAP (full IS)':>22s}")
for k in pols:
    if k == "twap":
        continue
    dn = NF[k] - NF["twap"]
    kk = len(GRIDS[k[3:]]) if k.startswith("CE@") else 0
    print(f"{k:18s} {kk:3d} {dn.mean():+10.1f} ({se(dn):5.1f}) {dn.mean() / cont:25.2f} {cvar95(IS[k]) - cvar95(IS['twap']):+22.1f}")
print("\nGrids:")
for nm, g in GRIDS.items():
    print(f"  {nm}: {g}")
print("\nNote: 'share' near 1.0 means the grid loses little; grids with more actions can only recover more of the CE gain,\n"
      "but each extra action shrinks the value gap the DDQN must resolve. Choose a grid, then run the training experiment.")