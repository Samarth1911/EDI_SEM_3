"""
diag_bl.py -- Stage 4A diagnostic for the Barger-Lorig (BL) benchmark. No training, no filter. UNTESTED until you run it.
Run from EDI/validation (venv active):   python diag_bl.py            (n=200 worlds, seeds 1,000,000+i)

Why: every bl1_* variant lost to TWAP (+101 .. +441). This script separates the candidate causes.
  World 'single' : BL's own assumptions (one CIR parameter set) hold. If bl_raw still loses to TWAP here, the
                   first-order expansion itself is out of its validity range at these parameters (or there is a bug).
  World 'regime' : the Stage 3/4 world (misspecified for BL).
Policies (all use TRUE alpha, kappa and the TRUE regime's parameters, i.e. best case for the formula):
  twap        : q/(N-t)
  bl_raw      : first-order BL multiplier, truncated at 0 only (== bl1_oracle_regime in run_stage4_benchmarks)
  bl_clip     : bl_raw multiplier clipped (not snapped) to [0.25, 2.0]  -> tests 'overshoot' vs 'grid quantisation'
  bl_damp0.5 / bl_damp0.25 : 1 + eps*(m-1)  -> tests 'right sign, too big'
  ce_exact    : certainty-equivalent plan v_s ~ 1/E[alpha_s] using the EXACT mean forecast (CIR decay + Markov regime),
                receding horizon. The BL alpha-term is the first-order Taylor expansion of this plan
                (1/T + mu/(2a); hand-derived by me; I did NOT re-derive the kappa-term and did not re-read the paper here).
                ce_exact ignores permanent impact and Jensen effects: a reference, not an optimum.
Metrics: full IS and noise-free cost sum(alpha v^2 + kappa v (q_before - v)), paired vs TWAP, negative = cheaper.
"""
import argparse, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from barger_lorig import params_from_regimes, drifts, bl_multiplier

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed0", type=int, default=1_000_000)
a = ap.parse_args()

sp = SimParams()
N, tau = sp.N, sp.tau


def make_rp(kind):
    if kind == "regime":
        rp = RegimeParams.default()
        rp.init_probs = rp.stationary_distribution()          # same as run_stage4_benchmarks / Stage 3
        return rp
    return RegimeParams.single()


def build(rp):
    K = int(rp.n_regimes)
    Pm = np.asarray(rp.P, dtype=float).reshape(K, K)
    Pp = params_from_regimes(sp, rp)
    eff = [rp.effective_cir(sp, r) for r in range(K)]
    LAM = np.array([e["lambda_alpha"] for e in eff], dtype=float)
    TH = np.array([e["theta_alpha"] for e in eff], dtype=float)
    mlog = []

    def vol(m, st):
        t, q = st["t"], st["q"]
        if q <= 0:
            return 0.0
        if t >= N - 1:
            return q
        return min(max(m, 0.0) * q / (N - t), q)

    def raw_mult(t, st):
        probs = np.eye(K)[int(st["regime"])]
        da, dk = drifts(Pp, probs, st["alpha"], st["kappa"])
        return bl_multiplier(t, N, tau, st["alpha"], st["kappa"], da, dk)

    def mk(tf, log=False):
        def f(st):
            t = st["t"]
            if t >= N - 1 or st["q"] <= 0:
                return vol(1.0, st)
            m = raw_mult(t, st)
            if log:
                mlog.append(m)
            return vol(tf(m), st)
        return f

    def forecast(alpha_t, reg_t, n_ahead):
        out = np.empty(n_ahead)
        pi = np.zeros(K); pi[reg_t] = 1.0
        m = pi * alpha_t
        out[0] = alpha_t
        for k in range(1, n_ahead):
            pi_next = pi @ Pm
            m = (1.0 - LAM * tau) * (m @ Pm) + LAM * TH * tau * pi_next
            pi = pi_next
            out[k] = m.sum()
        return out

    def ce(st):
        t, q = st["t"], st["q"]
        if t >= N - 1 or q <= 0:
            return vol(1.0, st)
        w = 1.0 / np.maximum(forecast(st["alpha"], int(st["regime"]), N - t), 1e-12)
        return q * w[0] / w.sum()

    pol = {
        "twap": lambda st: 0.0 if st["q"] <= 0 else st["q"] / (N - st["t"]),
        "bl_raw": mk(lambda m: m, log=True),
        "bl_clip": mk(lambda m: min(max(m, 0.25), 2.0)),
        "bl_damp0.5": mk(lambda m: 1.0 + 0.5 * (m - 1.0)),
        "bl_damp0.25": mk(lambda m: 1.0 + 0.25 * (m - 1.0)),
        "ce_exact": ce,
    }
    return pol, mlog


def run(seed, rp, pol):
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    gen = sim.run_stepwise(seed=seed)
    state = next(gen)
    while True:
        try:
            state = gen.send(float(pol(state)))
        except StopIteration as stop:
            return stop.value


def nf(ep):
    v = np.asarray(ep.v, dtype=float)
    qb = sp.q0 - np.concatenate([[0.0], np.cumsum(v)[:-1]])
    return float(np.sum(ep.alpha[:len(v)] * v ** 2 + ep.kappa[:len(v)] * v * (qb - v)))


def cvar95(x):
    k = max(1, int(math.ceil(0.05 * len(x))))
    return float(np.sort(x)[-k:].mean())


def se(d):
    return float(d.std(ddof=1) / math.sqrt(len(d)))


for kind in ("single", "regime"):
    rp = make_rp(kind)
    pol, mlog = build(rp)
    IS = {k: [] for k in pol}; NF = {k: [] for k in pol}
    for i in range(a.n):
        s = a.seed0 + i
        for k, f in pol.items():
            ep = run(s, rp, f)
            assert abs(ep.q[-1]) < 1e-6 * sp.q0, f"{k}: not liquidated (seed {s})"
            IS[k].append(ep.IS); NF[k].append(nf(ep))
    IS = {k: np.array(v) for k, v in IS.items()}; NF = {k: np.array(v) for k, v in NF.items()}
    print(f"\n===== world: {kind}  ({a.n} worlds, seeds {a.seed0}..{a.seed0 + a.n - 1}) =====")
    if kind == "regime":
        print(f"TWAP mean IS {IS['twap'].mean():.1f}  (run_stage4_benchmarks gave 4993.1 for the same 200 worlds)")
    print(f"{'policy':13s} {'meanIS':>8s} {'d_full(se)':>16s} {'d_noisefree(se)':>18s} {'CVaR95-TWAP':>11s}")
    for k in pol:
        df, dn = IS[k] - IS["twap"], NF[k] - NF["twap"]
        print(f"{k:13s} {IS[k].mean():8.1f} {df.mean():+8.1f} ({se(df) if k != 'twap' else 0:5.1f}) "
              f"{dn.mean():+9.1f} ({se(dn) if k != 'twap' else 0:5.1f}) {cvar95(IS[k]) - cvar95(IS['twap']):+11.1f}")
    m = np.array(mlog)
    pc = np.percentile(m, [1, 5, 25, 50, 75, 95, 99])
    print("bl_raw multiplier percentiles [1,5,25,50,75,95,99]:", np.round(pc, 2))
    print(f"bl_raw multiplier shares: >2.0: {np.mean(m > 2.0):.3f}   <0.25: {np.mean(m < 0.25):.3f}   <0 (truncated): {np.mean(m < 0):.3f}")

print("\nHow to read: (1) bl_raw loses in 'single' too -> expansion invalid at these parameters (or a formula/bug issue);"
      " wins in 'single' but loses in 'regime' -> misspecification.  (2) bl_clip/bl_damp better than bl_raw -> overshoot."
      "  (3) ce_exact beats TWAP while bl_raw does not -> the Taylor truncation, not the model-based idea, is the problem.")