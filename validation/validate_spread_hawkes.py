"""
validate_spread_hawkes.py  (STAGE 1 -- Person 1)
------------------------------------------------
Place in EDI/validation/ and run from there:   python validate_spread_hawkes.py
Requires EDI/src/spread_hawkes.py (plus the existing Stage 1 modules).

HARD checks (must pass)
  H1  construction: branching >= 1 is rejected; indicator variant without s_ref is rejected.
  H2  stationary mean spread (ticks) per regime vs the EXACT value 1 + (mu/(1-a))/lam_close,
      with a batch-means standard error (|z| < 3.5).
  H3  opening-event rate per step vs mean_intensity*tau (rel err < 5%), and
      closing rate == opening rate in stationarity (rel diff < 5%).
  H4  time-rescaling goodness-of-fit (KS vs Exp(1), p > 0.005) for the OPENING process
      (Hawkes compensator) and the CLOSING process (compensator = integral of
      lam_close*(S-1)) in each regime -> confirms the thinning sampler.
INFO (reported, not pass/fail)
  - indicator-rule phases (why the stabilised rule is the default)
  - calm-only proxy comparison: Gaussian vs v1 Hawkes vs spread Hawkes
  - switching episode: mean spread ticks by TRUE regime vs stationary theory
  - figure outputs/08_spread_hawkes.png, summary outputs/spread_hawkes_validation_summary.json
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
from hawkes_observation import (GaussianObservation, HawkesObservationModel, HawkesParams,
                                hawkes_compensator_increments)
from spread_hawkes import SpreadHawkesParams, SpreadHawkesProcess, SpreadHawkesObservationModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TAU = 0.02
REG_NAMES = ("calm", "stressed")


# ---------------------------------------------------------------------------
def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = float(np.dot(x, x))
    return np.array([np.dot(x[:-k], x[k:]) / d for k in range(1, nlags + 1)])


def batch_mean_se(x, nb=40):
    x = np.asarray(x, float)
    L = len(x) // nb
    m = x[:L * nb].reshape(nb, L).mean(axis=1)
    return float(m.mean()), float(m.std(ddof=1) / np.sqrt(nb))


def long_params(n_steps):
    return SimParams(T=n_steps * TAU, N=n_steps)


def shade(ax, reg):
    change = np.flatnonzero(np.diff(reg) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(reg)]])
    for s, e in zip(starts, ends):
        if reg[s] == 1:
            ax.axvspan(s, e, color="tab:red", alpha=0.12, lw=0)


# ---------------------------------------------------------------------------
# H1
def check_construction():
    out = {}
    try:
        SpreadHawkesParams(branching=(0.4, 1.0)); out["unstable_rejected"] = False
    except ValueError:
        out["unstable_rejected"] = True
    try:
        SpreadHawkesParams(stabilised=False); out["indicator_without_sref_rejected"] = False
    except ValueError:
        out["indicator_without_sref_rejected"] = True
    h = SpreadHawkesParams()
    out["indicator_rule_phases"] = {REG_NAMES[r]: h.phase(r) for r in range(h.n_regimes)}
    out["indicator_alpha_c"] = {REG_NAMES[r]: h.alpha_c(r) for r in range(h.n_regimes)}
    out["hard_pass"] = out["unstable_rejected"] and out["indicator_without_sref_rejected"]
    return out


# H2 + H3
def check_stationary(hp, n_steps=30000, burn=500, seed=1):
    res = {}
    for r in range(hp.n_regimes):
        mu, a = hp.baseline(r), hp.branching[r]
        proc = SpreadHawkesProcess(hp.beta, hp.lam_close, hp.stabilised, np.random.default_rng([seed, r]))
        S_end = np.empty(n_steps); n_o = np.empty(n_steps); n_c = np.empty(n_steps)
        for i in range(burn + n_steps):
            no, nc, _ = proc.advance(mu, a, TAU)
            if i >= burn:
                S_end[i - burn] = proc.S; n_o[i - burn] = no; n_c[i - burn] = nc
        th_S = hp.mean_spread_ticks(r)
        m_S, se_S = batch_mean_se(S_end)
        z = (m_S - th_S) / se_S
        th_rate = hp.mean_open_intensity(r) * TAU
        rate_open = float(n_o.mean()); rate_close = float(n_c.mean())
        rel_open = abs(rate_open - th_rate) / th_rate
        rel_bal = abs(rate_open - rate_close) / th_rate
        res[REG_NAMES[r]] = {
            "mean_ticks_theory": th_S, "mean_ticks_hat": m_S, "mean_ticks_se": se_S, "z": float(z),
            "open_per_step_theory": th_rate, "open_per_step_hat": rate_open,
            "close_per_step_hat": rate_close,
            "open_rel_err": float(rel_open), "open_close_rel_diff": float(rel_bal),
            "std_ticks": float(S_end.std()), "count_fano_open": float(n_o.var() / n_o.mean()),
            "hard_pass": bool(abs(z) < 3.5 and rel_open < 0.05 and rel_bal < 0.05)}
    res["hard_pass"] = all(v["hard_pass"] for k, v in res.items() if k in REG_NAMES)
    return res


# H4
def closing_compensator_increments(log, lam_minus_fn, S0=1):
    S, t_prev, acc, inc = S0, 0.0, 0.0, []
    for t, ty in log:
        acc += lam_minus_fn(S) * (t - t_prev)
        t_prev = t
        if ty == 1:
            S += 1
        else:
            inc.append(acc); acc = 0.0; S -= 1
    return np.asarray(inc, float)


def check_time_rescaling(hp, T=200.0, seed=3):
    res = {}
    for r in range(hp.n_regimes):
        mu, a = hp.baseline(r), hp.branching[r]
        proc = SpreadHawkesProcess(hp.beta, hp.lam_close, hp.stabilised, np.random.default_rng([seed, r]))
        n_o, n_c, log = proc.advance(mu, a, T, record=True)          # fresh process: S=1, E=0
        t_open = np.array([t for t, ty in log if ty == 1])
        inc_o = hawkes_compensator_increments(t_open, mu, a, hp.beta)
        inc_c = closing_compensator_increments(log, proc.lam_minus)
        ks_o = stats.kstest(inc_o, "expon"); ks_c = stats.kstest(inc_c, "expon")
        res[REG_NAMES[r]] = {"n_open": int(n_o), "n_close": int(n_c),
                             "ks_p_opening": float(ks_o.pvalue), "ks_p_closing": float(ks_c.pvalue),
                             "mean_compensator_inc_open": float(inc_o.mean()),
                             "mean_compensator_inc_close": float(inc_c.mean()),
                             "hard_pass": bool(ks_o.pvalue > 0.005 and ks_c.pvalue > 0.005)}
    res["hard_pass"] = all(v["hard_pass"] for k, v in res.items() if k in REG_NAMES)
    return res


# ---------------------------------------------------------------------------
# INFO: proxies
def resid(df, op):
    ar = df.true_alpha.values / op.alpha_ref
    return df.spread.values - op.spread_base * ar


def resid_summary(rs, op):
    a = acf(rs, 3)
    return {"std_over_assumed_noise": float(rs.std() / op.spread_noise_std),
            "skew": float(stats.skew(rs)), "excess_kurtosis": float(stats.kurtosis(rs)),
            "acf1": float(a[0]), "acf2": float(a[1]), "acf3": float(a[2]),
            "frac_gt_3_noise_std": float(np.mean(rs > 3 * op.spread_noise_std))}


def calm_comparison(hp_s, seed=7, n=6000):
    sp = long_params(n); op = ObsParams()
    sim = RegimeSwitchingSimulator(sp, RegimeParams.default(), seed=seed)
    ep = sim.simulate_episode(velocities=np.zeros(n), seed=seed,
                              forced_regime_path=np.zeros(n + 1, dtype=int))
    g = GaussianObservation(op, seed=seed).observe_episode(ep, seed=seed)
    h1 = HawkesObservationModel(op, HawkesParams(), tau=sp.tau, seed=seed).observe_episode(ep, seed=seed)
    h2 = SpreadHawkesObservationModel(op, hp_s, tau=sp.tau, seed=seed).observe_episode(ep, seed=seed)
    rg, r1, r2 = resid(g, op), resid(h1, op), resid(h2, op)
    out = {"gaussian": resid_summary(rg, op), "hawkes_v1": resid_summary(r1, op),
           "spread_hawkes": resid_summary(r2, op),
           "mean_spread_diff_v1_minus_gaussian": float(h1.spread.mean() - g.spread.mean()),
           "mean_spread_diff_spread_hawkes_minus_gaussian": float(h2.spread.mean() - g.spread.mean()),
           "spread_ticks_mean": float(h2.spread_ticks.mean()),
           "spread_ticks_mean_theory": hp_s.mean_spread_ticks(0),
           "n_open_fano": float(h2.n_open.var() / h2.n_open.mean())}
    return out, (rg, r1, r2), h2, op


def switching_episode(hp_s, seed=9, n=3000):
    sp = long_params(n); op = ObsParams()
    sim = RegimeSwitchingSimulator(sp, RegimeParams.default(), seed=seed)
    ep = sim.simulate_episode(velocities=np.zeros(n), seed=seed)
    df = SpreadHawkesObservationModel(op, hp_s, tau=sp.tau, seed=seed).observe_episode(ep, seed=seed)
    by = {}
    for r in range(hp_s.n_regimes):
        m = df.regime.values == r
        by[REG_NAMES[r]] = {"mean_ticks_realised": float(df.spread_ticks.values[m].mean()),
                            "mean_ticks_stationary_theory": hp_s.mean_spread_ticks(r),
                            "frac_time": float(m.mean())}
    return by, df


def fig_all(rs3, df_calm, df_sw, path):
    rg, r1, r2 = rs3
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))

    n = min(400, len(df_sw)); x = np.arange(n)
    ax[0, 0].step(x, df_sw.spread_ticks.values[:n], where="post", color="tab:blue", lw=0.9)
    shade(ax[0, 0], df_sw.regime.values[:n])
    ax[0, 0].set_title("Spread state S_t (ticks); red = stressed regime")
    ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("ticks")

    mx = int(df_sw.spread_ticks.max()) + 1
    bins = np.arange(1, mx + 2) - 0.5
    for r, c in ((0, "tab:blue"), (1, "tab:red")):
        v = df_sw.spread_ticks.values[df_sw.regime.values == r]
        ax[0, 1].hist(v, bins=bins, density=True, alpha=0.55, color=c, label=REG_NAMES[r])
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("Distribution of spread ticks by TRUE regime (log)")
    ax[0, 1].set_xlabel("ticks"); ax[0, 1].legend()

    L = 10; w = 0.27; lags = np.arange(1, L + 1)
    ax[1, 0].bar(lags - w, acf(rg, L), w, color="0.55", label="Gaussian")
    ax[1, 0].bar(lags, acf(r1, L), w, color="tab:orange", label="Hawkes v1 (count noise)")
    ax[1, 0].bar(lags + w, acf(r2, L), w, color="tab:red", label="Spread Hawkes (state)")
    ax[1, 0].set_title("Autocorrelation of spread residual (calm only)")
    ax[1, 0].set_xlabel("lag (steps)"); ax[1, 0].legend(fontsize=8)

    z0 = ObsParams().spread_noise_std
    b = np.linspace(-6, 12, 80)
    ax[1, 1].hist(rg / z0, bins=b, density=True, alpha=0.5, color="0.55", label="Gaussian")
    ax[1, 1].hist(r2 / z0, bins=b, density=True, alpha=0.5, color="tab:red", label="Spread Hawkes")
    xs = np.linspace(-6, 12, 300)
    ax[1, 1].plot(xs, stats.norm.pdf(xs), "k--", lw=1, label="N(0,1)")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_ylim(1e-4, 1)
    ax[1, 1].set_title("Residual / assumed noise std (log): tails & skew")
    ax[1, 1].legend()

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    R = {}
    hp = SpreadHawkesParams()

    R["H1_construction"] = c = check_construction()
    print(f"[HARD] construction: unstable rejected={c['unstable_rejected']} "
          f"indicator-without-s_ref rejected={c['indicator_without_sref_rejected']}")
    print(f"[INFO] indicator-rule phases (why stabilised is default): {c['indicator_rule_phases']} "
          f"alpha_c={ {k: round(v, 3) for k, v in c['indicator_alpha_c'].items()} }")

    R["H2H3_stationary"] = s = check_stationary(hp)
    for name in REG_NAMES:
        v = s[name]
        print(f"[HARD] stationary {name:8s} ticks theory={v['mean_ticks_theory']:.3f} hat={v['mean_ticks_hat']:.3f} "
              f"(se {v['mean_ticks_se']:.3f}, z={v['z']:+.2f}) | open/step theory={v['open_per_step_theory']:.3f} "
              f"hat={v['open_per_step_hat']:.3f} close/step={v['close_per_step_hat']:.3f} | "
              f"std ticks={v['std_ticks']:.2f} Fano(open)={v['count_fano_open']:.2f} pass={v['hard_pass']}")

    R["H4_time_rescaling"] = t = check_time_rescaling(hp)
    for name in REG_NAMES:
        v = t[name]
        print(f"[HARD] time-rescaling {name:8s} n_open={v['n_open']} n_close={v['n_close']} "
              f"KS_p(open)={v['ks_p_opening']:.3f} KS_p(close)={v['ks_p_closing']:.3f} pass={v['hard_pass']}")

    R["I_calm_proxies"], rs3, df_calm, op = calm_comparison(hp)
    cc = R["I_calm_proxies"]
    print("[INFO] calm-only proxies (6000 steps):")
    for k in ("gaussian", "hawkes_v1", "spread_hawkes"):
        v = cc[k]
        print(f"   {k:14s} std/noise={v['std_over_assumed_noise']:.2f} acf1={v['acf1']:+.3f} acf2={v['acf2']:+.3f} "
              f"skew={v['skew']:+.2f} exkurt={v['excess_kurtosis']:+.2f} P(>3sd)={v['frac_gt_3_noise_std']:.4f}")
    print(f"   mean spread diff vs Gaussian: v1={cc['mean_spread_diff_v1_minus_gaussian']:+.6f} "
          f"spread_hawkes={cc['mean_spread_diff_spread_hawkes_minus_gaussian']:+.6f}; "
          f"mean ticks={cc['spread_ticks_mean']:.3f} (theory {cc['spread_ticks_mean_theory']:.3f}); "
          f"opening-count Fano={cc['n_open_fano']:.2f}")

    R["I_switching_by_regime"], df_sw = switching_episode(hp)
    print(f"[INFO] switching episode, mean spread ticks by TRUE regime: {R['I_switching_by_regime']}")

    fig_all(rs3, df_calm, df_sw, os.path.join(OUT_DIR, "08_spread_hawkes.png"))

    hard = [c["hard_pass"], s["hard_pass"], t["hard_pass"]]
    R["ALL_HARD_PASS"] = bool(all(hard))
    with open(os.path.join(OUT_DIR, "spread_hawkes_validation_summary.json"), "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nALL HARD CHECKS PASS: {R['ALL_HARD_PASS']}  ({hard})")
    print(f"Wrote 08_spread_hawkes.png and spread_hawkes_validation_summary.json to {OUT_DIR}")


if __name__ == "__main__":
    main()