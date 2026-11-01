"""
stage1_extras.py  (STAGE 1 -- Person 1, follow-up to validate_stage1.py)
-----------------------------------------------------------------------
Standalone: does NOT modify simulator.py / regime_simulator.py / hawkes_observation.py.
Place in  EDI/validation/  and run from there:   python stage1_extras.py

Three things:
  1. INIT COVERAGE  -- what fraction of default-length (N=50) episodes never visit
     the stressed regime, for a calm start vs a stationary start. Analytic value is
     init[0] * P[0,0]**N; the script compares it to a Monte-Carlo estimate.
  2. BURSTY HAWKES PRESET -- a demonstration preset with longer memory and higher
     branching ratios, keeping the calm/stressed MEAN event rate identical to the
     default preset (mu/(1-n) unchanged) so the proxies remain moment-matched.
     Stability (branching < 1) is checked with the same tests as validate_stage1:
     moment estimate of the branching ratio, count z-score, time-rescaling KS.
  3. COMPARISON FIGURE -- Gaussian vs default Hawkes vs bursty Hawkes on the same
     calm-only episode -> outputs/07_bursty_hawkes_demo.png, plus a JSON summary.

STATUS: demonstration preset only. Parameters are illustrative, NOT calibrated.
The bursty preset makes the PF's Gaussian-likelihood misspecification much worse
by design; it is meant for visualisation and stress-testing, not as the new default.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from observation_model import ObsParams
from hawkes_observation import (HawkesParams, HawkesProcess, HawkesObservationModel,
                                GaussianObservation, hawkes_compensator_increments)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TAU = 0.02
N_PROXY = 6000

# Same mean intensity as the default preset:
#   calm:     15   / (1-0.85) = 100   (default 50 /(1-0.50)  = 100)
#   stressed: 22.5 / (1-0.92) = 281   (default 75 /(1-0.75)  = 300)
BURSTY = dict(mu0=15.0, mu_mult=(1.0, 1.5), branching=(0.85, 0.92), beta=10.0)


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = float(np.dot(x, x))
    return np.array([np.dot(x[:-k], x[k:]) / d for k in range(1, nlags + 1)])


# ---------------------------------------------------------------------------
# 1. initial-regime coverage
# ---------------------------------------------------------------------------
def check_init_coverage(n_draws=10000, seed=0):
    base = SimParams()                       # default N = 50
    N = base.N
    out = {"N": N, "n_draws": n_draws}
    for label in ("calm_start", "stationary_start"):
        rp = RegimeParams.default()
        if label == "stationary_start":
            rp.init_probs = rp.stationary_distribution()
        sim = RegimeSwitchingSimulator(base, rp, seed=seed)
        rng = np.random.default_rng([seed, 1])
        never = 0
        for _ in range(n_draws):
            path = sim.sample_regime_path(N, rng)
            never += int(path.max() == 0)
        theory = float(rp.init_probs[0] * rp.P[0, 0] ** N)
        out[label] = {"init_probs": [float(x) for x in rp.init_probs],
                      "frac_never_stressed_empirical": never / n_draws,
                      "frac_never_stressed_theory": theory}
    return out


# ---------------------------------------------------------------------------
# 2. Hawkes stability checks (same tests as validate_stage1.check_hawkes_process)
# ---------------------------------------------------------------------------
def check_process(hp, T=300.0, seed=3):
    res = {}
    for r, name in enumerate(("calm", "stressed")):
        mu, a = hp.baseline(r), hp.branching[r]
        proc = HawkesProcess(hp.beta, np.random.default_rng([seed, r]))
        times = proc.advance(mu, a, T)
        n = len(times)
        inc = hawkes_compensator_increments(times, mu, a, hp.beta)
        ks = stats.kstest(inc, "expon")
        lam_theory = hp.mean_intensity(r)
        z = (n - lam_theory * T) / np.sqrt(lam_theory * T / (1.0 - a) ** 2)  # asymptotic-variance z
        res[name] = {"branching_theory": float(a),
                     "branching_hat_moment": float(1.0 - mu * T / n),
                     "n_events": int(n),
                     "mean_intensity_hat": float(n / T),
                     "mean_intensity_theory": float(lam_theory),
                     "count_z_asymptotic_var": float(z),
                     "timerescale_ks_p": float(ks.pvalue)}
    return res


# ---------------------------------------------------------------------------
# 3. proxy comparison
# ---------------------------------------------------------------------------
def make_calm_episode(N, seed):
    sp = SimParams(T=N * TAU, N=N)
    sim = RegimeSwitchingSimulator(sp, RegimeParams.default(), seed=seed)
    ep = sim.simulate_episode(velocities=np.zeros(N), seed=seed,
                              forced_regime_path=np.zeros(N + 1, dtype=int))
    return ep, sp


def resid(df, op):
    ar = df.true_alpha.values / op.alpha_ref
    return df.spread.values - op.spread_base * ar


def summarise(rs, op, counts=None):
    a = acf(rs, 3)
    out = {"resid_std": float(rs.std()),
           "resid_std_over_assumed_noise": float(rs.std() / op.spread_noise_std),
           "skew": float(stats.skew(rs)),
           "excess_kurtosis": float(stats.kurtosis(rs)),
           "acf1": float(a[0]), "acf2": float(a[1]), "acf3": float(a[2]),
           "frac_gt_3_noise_std": float(np.mean(rs > 3 * op.spread_noise_std))}
    if counts is not None:
        out["count_mean_per_step"] = float(counts.mean())
        out["count_fano"] = float(counts.var() / counts.mean())
        out["count_acf1"] = float(acf(counts, 1)[0])
    return out


def fig_demo(rs_g, rs_d, rs_b, cnt_d, cnt_b, m_ref, op, path, seed=0):
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    n = 600
    pois = np.random.default_rng(seed).poisson(m_ref, len(cnt_d))

    def roll(c, w=10):
        return np.convolve(c, np.ones(w) / w, mode="valid")

    x = np.arange(n)
    ax[0, 0].plot(x, roll(pois)[:n], color="0.55", lw=1.0, label="Poisson (same mean)")
    ax[0, 0].plot(x, roll(cnt_d)[:n], color="tab:orange", lw=1.0, label="Hawkes default (n=0.5, beta=25)")
    ax[0, 0].plot(x, roll(cnt_b)[:n], color="tab:red", lw=1.2, label="Hawkes bursty (n=0.85, beta=10)")
    ax[0, 0].axhline(m_ref, color="k", ls=":", lw=0.8)
    ax[0, 0].set_title("Rolling 10-step mean event count (calm regime)")
    ax[0, 0].set_xlabel("step"); ax[0, 0].legend(fontsize=8)

    m = 300; xs = np.arange(m)
    ax[0, 1].plot(xs, rs_g[:m], color="0.55", lw=0.9, label="Gaussian")
    ax[0, 1].plot(xs, rs_b[:m], color="tab:red", lw=0.9, label="Hawkes bursty")
    ax[0, 1].set_title("Spread residual (proxy minus noise-free mean)")
    ax[0, 1].set_xlabel("step"); ax[0, 1].legend()

    L = 10; w = 0.27; lags = np.arange(1, L + 1)
    ax[1, 0].bar(lags - w, acf(rs_g, L), w, color="0.55", label="Gaussian")
    ax[1, 0].bar(lags, acf(rs_d, L), w, color="tab:orange", label="Hawkes default")
    ax[1, 0].bar(lags + w, acf(rs_b, L), w, color="tab:red", label="Hawkes bursty")
    ax[1, 0].set_title("Autocorrelation of spread residual")
    ax[1, 0].set_xlabel("lag (steps)"); ax[1, 0].legend()

    cmax = int(max(cnt_d.max(), cnt_b.max())) + 1
    bins = np.arange(0, cmax + 2) - 0.5
    ax[1, 1].hist(cnt_d, bins=bins, density=True, histtype="step", color="tab:orange", lw=1.5, label="Hawkes default")
    ax[1, 1].hist(cnt_b, bins=bins, density=True, histtype="step", color="tab:red", lw=1.5, label="Hawkes bursty")
    ks_ = np.arange(0, cmax + 1)
    ax[1, 1].plot(ks_, stats.poisson.pmf(ks_, m_ref), "k--", lw=1, label="Poisson pmf")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_ylim(1e-5, 1)
    ax[1, 1].set_title("Event-count distribution per step (log scale)")
    ax[1, 1].set_xlabel("events in step"); ax[1, 1].legend()

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    R = {}

    # ---- 1. init coverage ------------------------------------------------
    R["init_coverage"] = ic = check_init_coverage()
    print(f"[INIT] N={ic['N']} steps, {ic['n_draws']} draws. Fraction of episodes NEVER stressed:")
    for k in ("calm_start", "stationary_start"):
        v = ic[k]
        print(f"   {k:17s} empirical={v['frac_never_stressed_empirical']:.4f}  "
              f"theory={v['frac_never_stressed_theory']:.4f}  init={np.round(v['init_probs'], 3)}")

    # ---- 2. bursty preset stability --------------------------------------
    hp_def, hp_bur = HawkesParams(), HawkesParams(**BURSTY)
    R["stability_default"] = check_process(hp_def)
    R["stability_bursty"] = check_process(hp_bur)
    for label, d in (("default", R["stability_default"]), ("bursty ", R["stability_bursty"])):
        for name, v in d.items():
            print(f"[STAB {label}] {name:8s} n_theory={v['branching_theory']:.2f} "
                  f"n_hat={v['branching_hat_moment']:.3f} lam_hat={v['mean_intensity_hat']:.1f} "
                  f"lam_theory={v['mean_intensity_theory']:.1f} z={v['count_z_asymptotic_var']:+.2f} "
                  f"KS_p={v['timerescale_ks_p']:.3f}")

    # ---- 3. proxy comparison ---------------------------------------------
    seed = 7
    ep, sp = make_calm_episode(N_PROXY, seed)
    op = ObsParams()
    g = GaussianObservation(op, seed=seed).observe_episode(ep, seed=seed)
    hd = HawkesObservationModel(op, hp_def, tau=sp.tau, seed=seed).observe_episode(ep, seed=seed)
    hb = HawkesObservationModel(op, hp_bur, tau=sp.tau, seed=seed).observe_episode(ep, seed=seed)
    rg, rd, rb = resid(g, op), resid(hd, op), resid(hb, op)

    R["proxies_calm"] = {
        "gaussian": summarise(rg, op),
        "hawkes_default": summarise(rd, op, hd.n_events.values),
        "hawkes_bursty": summarise(rb, op, hb.n_events.values),
        "mean_spread_diff_default_minus_gaussian": float(hd.spread.mean() - g.spread.mean()),
        "mean_spread_diff_bursty_minus_gaussian": float(hb.spread.mean() - g.spread.mean()),
    }
    print("[PROXY calm-only, 6000 steps]")
    for k in ("gaussian", "hawkes_default", "hawkes_bursty"):
        s = R["proxies_calm"][k]
        extra = (f" count_fano={s['count_fano']:.2f} count_acf1={s['count_acf1']:+.3f}"
                 if "count_fano" in s else "")
        print(f"   {k:15s} resid_std/noise={s['resid_std_over_assumed_noise']:.2f} "
              f"acf1={s['acf1']:+.3f} skew={s['skew']:+.2f} exkurt={s['excess_kurtosis']:+.2f} "
              f"P(>3sd)={s['frac_gt_3_noise_std']:.4f}{extra}")
    print(f"   mean spread diff vs Gaussian: default={R['proxies_calm']['mean_spread_diff_default_minus_gaussian']:+.6f} "
          f"bursty={R['proxies_calm']['mean_spread_diff_bursty_minus_gaussian']:+.6f}")

    fig_demo(rg, rd, rb, hd.n_events.values, hb.n_events.values, hp_def.expected_count(0, sp.tau), op,
             os.path.join(OUT_DIR, "07_bursty_hawkes_demo.png"), seed=seed)

    with open(os.path.join(OUT_DIR, "stage1_extras_summary.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(f"Wrote 07_bursty_hawkes_demo.png and stage1_extras_summary.json to {OUT_DIR}")


if __name__ == "__main__":
    main()