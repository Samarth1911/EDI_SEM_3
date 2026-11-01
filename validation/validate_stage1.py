"""
validate_stage1.py  (STAGE 1 -- Person 1)
-----------------------------------------
Run from project root:   python validation/validate_stage1.py
Quick smoke test:        python validation/validate_stage1.py --quick

Two kinds of checks, kept separate on purpose:
  [HARD]  mathematical-correctness checks with an explicit pass/fail tolerance
          (regression vs original simulator, Markov chain, CIR mean-reversion,
          Hawkes thinning correctness, stability).
  [INFO]  observational results with NO pass/fail (how separated the regimes
          are, how bursty Hawkes proxies are) -- reported as numbers only.

Writes to validation/outputs/:
  04_regime_paths.png, 05_hawkes_vs_gaussian.png, 06_hawkes_by_regime.png,
  stage1_validation_summary.json, stage1_handoff.md  (numbers auto-filled).
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from simulator import MarketSimulator, SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from observation_model import ObsParams
from hawkes_observation import (HawkesParams, HawkesProcess, HawkesObservationModel,
                                GaussianObservation, hawkes_compensator_increments)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
TAU = 0.02
QUICK = "--quick" in sys.argv
N_LONG = 4000 if QUICK else 20000
N_CHAIN = 100_000 if QUICK else 400_000
T_HAWKES = 60.0 if QUICK else 300.0
N_PROXY = 1500 if QUICK else 6000


# ------------------------------------------------------------------ helpers
def long_params(n_steps: int) -> SimParams:
    """Same tau=0.02 as the default (T=1, N=50) but a longer horizon."""
    return SimParams(T=n_steps * TAU, N=n_steps)


def runs(path):
    change = np.flatnonzero(np.diff(path) != 0) + 1
    starts = np.concatenate([[0], change]); ends = np.concatenate([change, [len(path)]])
    return starts, ends, path[starts]


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x); d = float(np.dot(x, x))
    return np.array([np.dot(x[:-k], x[k:]) / d for k in range(1, nlags + 1)])


def auc(x1, x0):
    U = stats.mannwhitneyu(x1, x0, alternative="two-sided").statistic
    return float(U / (len(x1) * len(x0)))


def shade(ax, reg):
    s, e, r = runs(reg)
    for a, b, k in zip(s, e, r):
        if k == 1:
            ax.axvspan(a - 0.5, b - 0.5, color="tab:red", alpha=0.13, lw=0)


# ------------------------------------------------------------------ A. regression
def check_regression():
    worst = 0.0
    for s, shock in ((3, None), (17, None), (101, 25)):
        base = MarketSimulator(seed=s); tw = base.twap_schedule()
        kw = dict(velocities=tw, seed=s)
        if shock is not None:
            kw.update(shock_step=shock, shock_jump_alpha_mult=3.0)
        e0 = base.simulate_episode(**kw)
        e1 = RegimeSwitchingSimulator(regime_params=RegimeParams.single(), seed=s).simulate_episode(**kw)
        for f in ("S", "S_exec", "kappa", "alpha", "v", "q"):
            worst = max(worst, float(np.max(np.abs(getattr(e0, f) - getattr(e1, f)))))
        worst = max(worst, abs(e0.IS - e1.IS))
    return {"max_abs_diff": worst, "bit_for_bit": worst == 0.0, "n_cases": 3}


# ------------------------------------------------------------------ B. Markov chain
def check_markov_chain(rp, seed=0):
    sim = RegimeSwitchingSimulator(regime_params=rp, seed=seed)
    path = sim.sample_regime_path(N_CHAIN, np.random.default_rng(seed))
    K = rp.n_regimes
    counts = np.zeros((K, K)); np.add.at(counts, (path[:-1], path[1:]), 1)
    P_hat = counts / counts.sum(1, keepdims=True)
    pi_hat = np.bincount(path, minlength=K) / len(path)
    s, e, r = runs(path); lens = (e - s)[1:-1]; rr = r[1:-1]     # drop censored first/last run
    soj_hat = [float(lens[rr == k].mean()) for k in range(K)]
    soj_true = [float(x) for x in rp.expected_sojourn_steps()]
    err = float(np.max(np.abs(P_hat - rp.P)))
    soj_rel = float(max(abs(a - b) / b for a, b in zip(soj_hat, soj_true)))
    return {"n": N_CHAIN, "P_true": rp.P.tolist(), "P_hat": P_hat.tolist(), "max_abs_P_err": err,
            "pi_true": rp.stationary_distribution().tolist(), "pi_hat": pi_hat.tolist(),
            "sojourn_true": soj_true, "sojourn_hat": soj_hat, "max_rel_sojourn_err": soj_rel,
            "hard_pass": bool(err < 0.005 and soj_rel < 0.05)}


# ------------------------------------------------------------------ C. distinctness
def check_distinctness(rp, seed=5):
    sp = long_params(N_LONG)
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    ep = sim.simulate_episode(velocities=np.zeros(N_LONG), seed=seed)   # v=0: pure world statistics
    reg = ep.regime; K = rp.n_regimes
    dS = np.diff(ep.S); reg_gen = reg[1:]                                # regime that generated step t->t+1
    out = {"n_steps": N_LONG, "regimes": {}}
    for r in range(K):
        m = reg == r; e = sim._eff[r]
        d = {"frac_time": float(m.mean()),
             "alpha_mean": float(ep.alpha[m].mean()), "alpha_longrun_theory": float(e["theta_alpha"]),
             "kappa_mean": float(ep.kappa[m].mean()), "kappa_longrun_theory": float(e["theta_kappa"]),
             "dS_std": float(dS[reg_gen == r].std()), "dS_std_theory": float(e["sigma_price"] * math.sqrt(sp.tau))}
        for nm, arr in (("alpha", ep.alpha), ("kappa", ep.kappa)):
            idx = np.flatnonzero(reg_gen == r)
            x = e["theta_" + nm] - arr[idx]; y = arr[idx + 1] - arr[idx]
            d[nm + "_revert_hat"] = float(np.dot(x, y) / np.dot(x, x))       # OLS through origin
            d[nm + "_revert_theory"] = float(e["lambda_" + nm] * sp.tau)
        out["regimes"][rp.names[r]] = d
    revert_err = max(abs(v["alpha_revert_hat"] / v["alpha_revert_theory"] - 1) for v in out["regimes"].values())
    revert_err = max(revert_err, max(abs(v["kappa_revert_hat"] / v["kappa_revert_theory"] - 1) for v in out["regimes"].values()))
    out["revert_max_rel_err"] = float(revert_err)
    out["hard_pass_revert"] = bool(revert_err < 0.15)
    if K >= 2:
        a0, a1 = ep.alpha[reg == 0], ep.alpha[reg == 1]
        k0, k1 = ep.kappa[reg == 0], ep.kappa[reg == 1]
        df = GaussianObservation(ObsParams(), seed=seed).observe_episode(ep, seed=seed)
        sv = df.spread.values; rg = df.regime.values
        c, s_ = out["regimes"][rp.names[0]], out["regimes"][rp.names[1]]
        out["separation"] = {
            "alpha_auc": auc(a1, a0), "kappa_auc": auc(k1, k0),
            "alpha_ks_stat": float(stats.ks_2samp(a1, a0).statistic),
            "alpha_cohens_d": float((a1.mean() - a0.mean()) / math.sqrt(0.5 * (a1.var() + a0.var()))),
            "alpha_ratio_realised": float(a1.mean() / a0.mean()),
            "alpha_ratio_theory": float(s_["alpha_longrun_theory"] / c["alpha_longrun_theory"]),
            "dS_std_ratio_realised": float(s_["dS_std"] / c["dS_std"]),
            "dS_std_ratio_theory": float(s_["dS_std_theory"] / c["dS_std_theory"]),
            "spread_auc_one_step_gaussian": auc(sv[rg == 1], sv[rg == 0])}
        e0, e1 = sim._eff[0], sim._eff[1]
        mid = 0.5 * (e0["theta_alpha"] + e1["theta_alpha"])
        s, e, r = runs(reg); first = []; lens = []
        for a, b, k in zip(s[1:-1], e[1:-1], r[1:-1]):
            if k != 1:
                continue
            lens.append(b - a)
            hit = np.flatnonzero(ep.alpha[a:b] >= mid)
            first.append(int(hit[0]) if hit.size else -1)
        first = np.array(first); ok = first >= 0
        out["stress_run_reachability"] = {
            "n_runs": int(len(first)), "mean_run_len_steps": float(np.mean(lens)),
            "frac_runs_where_alpha_reaches_midpoint": float(ok.mean()),
            "median_steps_to_midpoint": float(np.median(first[ok])) if ok.any() else None}
    return out, ep, sp


def fig_regimes(ep, sp, path):
    n = min(400, len(ep.regime) - 1); x = np.arange(n); reg = ep.regime[:n]
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    ax[0, 0].plot(x, ep.alpha[:n] / sp.alpha0, lw=1.1, color="tab:blue"); shade(ax[0, 0], reg)
    ax[0, 0].set_title("Temporary impact alpha_t / alpha0  (red = stressed regime)")
    ax[0, 1].plot(x, ep.kappa[:n] / sp.kappa0, lw=1.1, color="tab:green"); shade(ax[0, 1], reg)
    ax[0, 1].set_title("Permanent impact kappa_t / kappa0")
    ax[1, 0].plot(x, ep.S[:n], lw=1.1, color="k"); shade(ax[1, 0], reg)
    ax[1, 0].set_title("Mid-price S_t (no trading) -- wider moves in red")
    for k, (lab, col) in enumerate((("calm", "tab:blue"), ("stressed", "tab:red"))):
        v = ep.alpha[ep.regime == k] / sp.alpha0
        if v.size:
            ax[1, 1].hist(v, bins=70, density=True, alpha=0.55, color=col, label=lab)
    ax[1, 1].set_title("Distribution of alpha_t / alpha0 by TRUE regime"); ax[1, 1].legend()
    for a in ax[:, 0]: a.set_xlabel("step")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ D. Hawkes process
def check_hawkes_process(hp):
    res = {"params": {"mu0": hp.mu0, "mu_mult": list(hp.mu_mult), "branching": list(hp.branching),
                      "beta": hp.beta, "tau": TAU, "T_sim": T_HAWKES}, "regimes": {}}
    all_hard = True
    for r, name in enumerate(("calm", "stressed")[:hp.n_regimes]):
        mu, a = hp.baseline(r), hp.branching[r]
        proc = HawkesProcess(hp.beta, np.random.default_rng([2024, r]))
        ev = proc.advance(mu, a, T_HAWKES)
        n = len(ev); Lam = hp.mean_intensity(r)
        z = (n - Lam * T_HAWKES) / math.sqrt(Lam * T_HAWKES / (1 - a) ** 2)
        inc = hawkes_compensator_increments(ev, mu, a, hp.beta)
        ks = stats.kstest(inc, "expon")
        hard = bool(abs(z) < 4 and ks.pvalue > 0.01)
        all_hard &= hard
        res["regimes"][name] = {
            "mu": mu, "branching_theory": a, "branching_hat_moment": float(1 - mu * T_HAWKES / n),
            "mean_intensity_theory": Lam, "mean_intensity_hat": float(n / T_HAWKES), "n_events": int(n),
            "count_z": float(z), "timerescale_ks_stat": float(ks.statistic), "timerescale_ks_p": float(ks.pvalue),
            "compensator_acf1": float(acf(inc, 1)[0]),
            "expected_count_per_step": hp.expected_count(r, TAU), "fano_asymptotic": hp.fano_asymptotic(r),
            "hard_pass": hard}
    try:
        HawkesParams(branching=(0.5, 1.0)); rejected = False
    except ValueError:
        rejected = True
    res["unstable_branching_rejected"] = rejected
    res["hard_pass"] = bool(all_hard and rejected and max(hp.branching) < 1)
    return res


# ------------------------------------------------------------------ E. proxy comparison
def resid_stats(df, op):
    ar = df.true_alpha.values / op.alpha_ref
    rs = df.spread.values - op.spread_base * ar
    rv = df.volume.values - op.volume_base / np.maximum(ar, 1e-3)
    a = acf(rs, 3)
    return {"mean": float(rs.mean()), "std": float(rs.std()), "skew": float(stats.skew(rs)),
            "excess_kurtosis": float(stats.kurtosis(rs)), "acf1": float(a[0]), "acf2": float(a[1]),
            "acf3": float(a[2]), "frac_gt_3_noise_std": float(np.mean(rs > 3 * op.spread_noise_std)),
            "corr_spread_volume": float(np.corrcoef(rs, rv)[0, 1])}, rs


def proxy_comparison(rp, hp, regime_const, seed=7):
    sp = long_params(N_PROXY); op = ObsParams()
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    ep = sim.simulate_episode(velocities=np.zeros(N_PROXY), seed=seed,
                              forced_regime_path=np.full(N_PROXY + 1, regime_const, dtype=int))
    g = GaussianObservation(op, seed=seed).observe_episode(ep, seed=seed)
    h = HawkesObservationModel(op, hp, tau=sp.tau, seed=seed).observe_episode(ep, seed=seed)
    sg, rg = resid_stats(g, op); sh, rh = resid_stats(h, op)
    cnt = h.n_events.values
    out = {"regime": regime_const, "gaussian": sg, "hawkes": sh,
           "hawkes_count_mean_per_step": float(cnt.mean()), "hawkes_count_fano": float(cnt.var() / cnt.mean()),
           "hawkes_count_acf1": float(acf(cnt, 1)[0]),
           "resid_std_ratio_hawkes_over_gaussian": float(sh["std"] / sg["std"]),
           "mean_spread_diff_hawkes_minus_gaussian": float(h.spread.mean() - g.spread.mean())}
    return out, g, h, rg, rh, op


def fig_hawkes(g, h, rg, rh, op, cmean, path):
    n = min(300, len(g)); x = np.arange(n)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    ax[0, 0].plot(x, rg[:n], color="0.55", lw=0.9, label="Gaussian")
    ax[0, 0].plot(x, rh[:n], color="tab:red", lw=0.9, label="Hawkes")
    ax[0, 0].set_title("Spread residual (proxy minus noise-free mean), calm regime"); ax[0, 0].legend()
    pois = np.random.default_rng(0).poisson(cmean, n)
    ax[0, 1].bar(x, h.n_events.values[:n], color="tab:red", alpha=0.7, label="Hawkes events / step")
    ax[0, 1].step(x, pois, color="0.4", lw=0.9, label=f"Poisson, same mean ({cmean:.1f})")
    ax[0, 1].set_title("Event counts per step: clustered vs Poisson"); ax[0, 1].legend()
    L = 10; w = 0.38; lags = np.arange(1, L + 1)
    ax[1, 0].bar(lags - w / 2, acf(rg, L), w, color="0.55", label="Gaussian")
    ax[1, 0].bar(lags + w / 2, acf(rh, L), w, color="tab:red", label="Hawkes")
    ax[1, 0].set_title("Autocorrelation of spread residual"); ax[1, 0].set_xlabel("lag (steps)"); ax[1, 0].legend()
    z0 = op.spread_noise_std; bins = np.linspace(-6, 12, 80)
    ax[1, 1].hist(rg / z0, bins=bins, density=True, alpha=0.55, color="0.55", label="Gaussian")
    ax[1, 1].hist(rh / z0, bins=bins, density=True, alpha=0.55, color="tab:red", label="Hawkes")
    xs = np.linspace(-6, 12, 300); ax[1, 1].plot(xs, stats.norm.pdf(xs), "k--", lw=1, label="N(0,1)")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_ylim(1e-4, 1)
    ax[1, 1].set_title("Residual / assumed noise std (log scale): tails & skew"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ F. Hawkes by regime
def hawkes_by_regime(ep, sp, hp, path, seed=9):
    h = HawkesObservationModel(ObsParams(), hp, tau=sp.tau, seed=seed).observe_episode(ep, seed=seed)
    out = {}
    for r, name in enumerate(("calm", "stressed")[:hp.n_regimes]):
        m = h.regime.values == r
        out[name] = {"mean_events_per_step_realised": float(h.n_events.values[m].mean()),
                     "mean_events_per_step_stationary_theory": hp.expected_count(r, sp.tau)}
    n = 400; x = np.arange(n); reg = h.regime.values[:n]
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(x, h.true_alpha.values[:n] / sp.alpha0, color="tab:blue", lw=1.1); shade(ax[0], reg)
    ax[0].set_title("True alpha_t / alpha0 (red = stressed regime)")
    ax[1].bar(x, h.n_events.values[:n], color="tab:red", alpha=0.7); shade(ax[1], reg)
    ax[1].set_title("Hawkes stress-event count per step -- activity clusters and rises in stress")
    ax[1].set_xlabel("step"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return out


# ------------------------------------------------------------------ handoff writer
def write_handoff(R, path):
    A, B, Cd, Cs = R["A_regression"], R["B_chain_default"], R["C_default"], R["C_subtle"]
    D, Ec, Es, F = R["D_hawkes_process"], R["E_proxy_calm"], R["E_proxy_stressed"], R["F_hawkes_by_regime"]
    sep, ssep = Cd["separation"], Cs["separation"]; reach = Cd["stress_run_reachability"]
    pf = lambda b: "PASS" if b else "FAIL"
    Dc, Ds = D["regimes"]["calm"], D["regimes"]["stressed"]
    md = f"""# STAGE 1 -- HANDOFF SYNOPSIS (Person 1: World Generation)
_Auto-generated by validation/validate_stage1.py{' [--quick run: reduced sample sizes]' if QUICK else ''}. All numbers below are from that run._

## (a) What was built
1. `src/regime_simulator.py` -- `RegimeSwitchingSimulator(MarketSimulator)`: Hamilton-style 2-state Markov chain (0=calm, 1=stressed), per-STEP transition matrix P={np.round(np.array(B['P_true']),3).tolist()}. CIR kappa/alpha run conditional on the active regime via multipliers on the base SimParams (long-run mean, mean-reversion speed, vol-of-vol, price sigma). `simulator.py` is untouched; the class subclasses it.
2. `src/hawkes_observation.py` -- `HawkesObservationModel` (univariate exponential-kernel Hawkes, Ogata thinning; regime-dependent baseline and branching ratio) and `GaussianObservation` (the existing validated model behind the same interface). Selector: `make_observation_model("gaussian" | "hawkes", ...)`.
3. `validation/validate_stage1.py` (this run).

## (b) What the validation shows
**[HARD] correctness checks**
- Single-regime config vs original `MarketSimulator`: max |diff| over S, S_exec, kappa, alpha, v, q, IS across {A['n_cases']} cases (incl. one with a shock) = {A['max_abs_diff']:.3e} -> bit-for-bit {pf(A['bit_for_bit'])}.
- Markov chain ({B['n']:,} steps): max |P_hat - P| = {B['max_abs_P_err']:.4f}; stationary pi_hat = {np.round(B['pi_hat'],3).tolist()} (theory {np.round(B['pi_true'],3).tolist()}); mean sojourn hat = {np.round(B['sojourn_hat'],1).tolist()} steps (theory {np.round(B['sojourn_true'],1).tolist()}) -> {pf(B['hard_pass'])}.
- CIR mean-reversion per regime (OLS of d(alpha,kappa) on long-run-mean gap): worst relative error vs lambda*tau = {Cd['revert_max_rel_err']*100:.1f}% (tolerance 15%) -> {pf(Cd['hard_pass_revert'])}.
- Hawkes thinning (constant baseline, T={D['params']['T_sim']:.0f}): calm n_events={Dc['n_events']:,}, count z-score {Dc['count_z']:+.2f}, time-rescaling KS p={Dc['timerescale_ks_p']:.3f}; stressed n_events={Ds['n_events']:,}, z {Ds['count_z']:+.2f}, KS p={Ds['timerescale_ks_p']:.3f} -> {pf(D['hard_pass'])}.
- **Branching ratio (stability)**: theory calm = {Dc['branching_theory']:.2f}, stressed = {Ds['branching_theory']:.2f} (both < 1; branching >= 1 is rejected at construction: {pf(D['unstable_branching_rejected'])}). Moment estimate 1 - mu*T/N: calm {Dc['branching_hat_moment']:.3f}, stressed {Ds['branching_hat_moment']:.3f}. Mean intensity hat vs theory: calm {Dc['mean_intensity_hat']:.1f} vs {Dc['mean_intensity_theory']:.1f}; stressed {Ds['mean_intensity_hat']:.1f} vs {Ds['mean_intensity_theory']:.1f} events/unit time.

**[INFO] regime distinctness (default preset, v=0, {Cd['n_steps']:,} steps)** -- see `04_regime_paths.png`
- Time fraction stressed = {Cd['regimes']['stressed']['frac_time']:.3f}. Realised mean alpha ratio stressed/calm = {sep['alpha_ratio_realised']:.2f} vs long-run-mean ratio {sep['alpha_ratio_theory']:.2f} (they differ because stressed spells are short relative to mean-reversion time, so alpha often does not reach its stressed long-run mean).
- Price-diffusion std ratio stressed/calm = {sep['dS_std_ratio_realised']:.2f} (theory {sep['dS_std_ratio_theory']:.2f}).
- Separability of TRUE alpha by regime: AUC = {sep['alpha_auc']:.3f}, KS = {sep['alpha_ks_stat']:.3f}, Cohen's d = {sep['alpha_cohens_d']:.2f}. One-step Gaussian spread proxy AUC = {sep['spread_auc_one_step_gaussian']:.3f}.
- Stressed spells: mean length {reach['mean_run_len_steps']:.1f} steps; alpha reaches the calm/stressed midpoint in {reach['frac_runs_where_alpha_reaches_midpoint']*100:.0f}% of spells (median {reach['median_steps_to_midpoint']} steps after onset among those that do).
- `subtle` preset (for stress-testing detectors): alpha AUC = {ssep['alpha_auc']:.3f}, Cohen's d = {ssep['alpha_cohens_d']:.2f}.

**[INFO] Hawkes vs Gaussian proxies** (calm-only forced path, {N_PROXY:,} steps) -- see `05_hawkes_vs_gaussian.png`
- Mean spread difference Hawkes - Gaussian = {Ec['mean_spread_diff_hawkes_minus_gaussian']:+.5f} $ (moment-matching check; ~0 expected).
- Spread-residual lag-1 autocorrelation: Gaussian {Ec['gaussian']['acf1']:+.3f} vs Hawkes {Ec['hawkes']['acf1']:+.3f}. Skew: {Ec['gaussian']['skew']:+.2f} vs {Ec['hawkes']['skew']:+.2f}. Excess kurtosis: {Ec['gaussian']['excess_kurtosis']:+.2f} vs {Ec['hawkes']['excess_kurtosis']:+.2f}. P(resid > 3 assumed-noise-std): {Ec['gaussian']['frac_gt_3_noise_std']:.4f} vs {Ec['hawkes']['frac_gt_3_noise_std']:.4f}.
- Residual std ratio Hawkes/Gaussian = {Ec['resid_std_ratio_hawkes_over_gaussian']:.2f} (the filter assumes 1.00 -> expect over-confidence). Event-count Fano factor = {Ec['hawkes_count_fano']:.2f} (Poisson = 1), count ACF1 = {Ec['hawkes_count_acf1']:+.3f}. Corr(spread resid, volume resid): Gaussian {Ec['gaussian']['corr_spread_volume']:+.2f} vs Hawkes {Ec['hawkes']['corr_spread_volume']:+.2f}.
- Stressed-only forced path: mean spread shift Hawkes - Gaussian = {Es['mean_spread_diff_hawkes_minus_gaussian']:+.4f} $, residual std ratio = {Es['resid_std_ratio_hawkes_over_gaussian']:.2f}, spread-residual ACF1 = {Es['hawkes']['acf1']:+.3f}.
- Activity by TRUE regime on the full switching episode (`06_hawkes_by_regime.png`): mean events/step calm {F['calm']['mean_events_per_step_realised']:.2f} (stationary theory {F['calm']['mean_events_per_step_stationary_theory']:.2f}), stressed {F['stressed']['mean_events_per_step_realised']:.2f} (theory {F['stressed']['mean_events_per_step_stationary_theory']:.2f}).

## (c) Data interface produced (DRAFT -- Stage 0 has not yet ratified this)
**Simulator -> Observation Model / Filter / Decision**
- `RegimeSwitchingSimulator(params: SimParams|None, regime_params: RegimeParams|None, seed)`; `.run_stepwise(shock_step, shock_jump_kappa_mult, shock_jump_alpha_mult, seed, forced_regime_path)` is the same pull-based generator protocol as `MarketSimulator` (`state = next(gen); state = gen.send(v_t); StopIteration.value = episode`).
- Per-step `state` dict: `{{t, S, kappa, alpha, q, N, shock_step, regime}}`. `kappa`, `alpha`, `regime` are TRUE/hidden values -- belief-based policies must not read them (oracle only).
- `RegimeEpisode` = existing `Episode` fields + `regime: np.ndarray (N+1,)` int, aligned with `kappa`, `alpha`, `S`. Timing: `regime[t]` is the regime whose parameters generated `x_t` from `x_{{t-1}}` (Blom & Bar-Shalom jump-system form `x_{{t+1}} = f(x_t, s_{{t+1}})`); `regime[0]` only fixes the initial state.
- `force_regime_window(path, start, duration, regime=1)` supports controlled tests and the dashboard shock-injection control.

**Observation Model -> Particle Filter**
- Both models: `observe_step(kappa_t, alpha_t, regime=0, rng=None) -> dict` with keys `spread, volume, realvol` (the only keys the current filter reads, by name); Hawkes additionally returns `n_events, hawkes_excitation`. `observe_episode(episode, seed) -> pandas.DataFrame` with columns `t, spread, volume, realvol, [n_events, hawkes_excitation,] regime, true_kappa, true_alpha`. Hawkes is STATEFUL: call `observe_step` in time order after `reset(seed)`.

**Presets**: `RegimeParams.default()` (stressed: alpha long-run x2.5, kappa x1.6, lambda x2, price sigma x1.6), `.subtle()` (alpha x1.15, kappa x1.08), `.single()` (== original simulator). Hawkes defaults: mu0={D['params']['mu0']}, mu_mult={D['params']['mu_mult']}, branching={D['params']['branching']}, beta={D['params']['beta']}.

## (d) Assumptions, simplifications, open questions
1. **Stage 0 not yet done.** The interface above is the as-built Stage 1 output, offered as input to Stage 0, not agreed. Flag any mismatch rather than silently adapting.
2. **Uncalibrated parameters.** Regime multipliers, transition matrix, Hawkes mu0/branching/beta and proxy loadings are illustrative, not fitted to data.
3. **Short episodes vs regime persistence.** Default horizon is N=50 steps with P per step; starting calm, P(no stressed step in an episode) is roughly 0.96^50 = 0.13, so ~13% of default-length training episodes never see stress unless `init_probs` is set to the stationary distribution (option exists) or the horizon is lengthened.
4. **Regimes only partly express within a spell.** Stressed spells are short relative to CIR mean-reversion time (see reachability numbers above); the detector will face gradual transitions, not step changes.
5. **Hawkes look-ahead convention.** Row t counts events in [t*tau, (t+1)*tau) conditioned on x_t (same-index convention as Gaussian). Open question for Stage 0: keep, or shift to a strictly causal (t-1, t] window.
6. **Filter misspecification (intended).** The PF's likelihood is Gaussian with known means; Hawkes proxies are non-Gaussian, autocorrelated, and mean-shifted in stress. Stage 2 should expect worse calibration on Hawkes data and decide whether to add a count-based likelihood.
7. **Model scope.** Univariate Hawkes (Bacry et al. 2013 is mutually exciting), and the Fosset-Bouchaud-Benzaquen feedback mechanism is not reproduced; no dependence of events on our own trading. Verify both papers before citing this as their implementation.
8. **Existing jump shock retained.** `shock_step` (within-regime jump) is distinct from a regime switch; the old filter validation used the former.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


# ------------------------------------------------------------------ main
def _jsonable(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(type(o))


def main():
    print(f"validate_stage1 ({'QUICK' if QUICK else 'FULL'})")
    R = {}
    R["A_regression"] = check_regression()
    print(f"[HARD] regression vs original simulator: max|diff|={R['A_regression']['max_abs_diff']:.2e} "
          f"bit-for-bit={R['A_regression']['bit_for_bit']}")

    rp, rp_sub = RegimeParams.default(), RegimeParams.subtle()
    R["B_chain_default"] = check_markov_chain(rp)
    b = R["B_chain_default"]
    print(f"[HARD] Markov chain: max|P err|={b['max_abs_P_err']:.4f} sojourn hat={np.round(b['sojourn_hat'],1)} "
          f"theory={np.round(b['sojourn_true'],1)} pass={b['hard_pass']}")

    R["C_default"], ep_long, sp_long = check_distinctness(rp)
    R["C_subtle"], _, _ = check_distinctness(rp_sub, seed=6)
    fig_regimes(ep_long, sp_long, os.path.join(OUT_DIR, "04_regime_paths.png"))
    c = R["C_default"]
    print(f"[HARD] CIR mean-reversion worst rel err={c['revert_max_rel_err']*100:.1f}% pass={c['hard_pass_revert']}")
    print(f"[INFO] regime separation (default): {c['separation']}")
    print(f"[INFO] stressed-spell reachability: {c['stress_run_reachability']}")
    print(f"[INFO] regime separation (subtle):  {R['C_subtle']['separation']}")

    hp = HawkesParams()
    R["D_hawkes_process"] = check_hawkes_process(hp)
    for name, v in R["D_hawkes_process"]["regimes"].items():
        print(f"[HARD] Hawkes {name}: n_theory={v['branching_theory']:.2f} n_hat={v['branching_hat_moment']:.3f} "
              f"count_z={v['count_z']:+.2f} KS_p={v['timerescale_ks_p']:.3f} pass={v['hard_pass']}")
    print(f"[HARD] unstable branching rejected: {R['D_hawkes_process']['unstable_branching_rejected']}")

    R["E_proxy_calm"], g, h, rg, rh, op = proxy_comparison(rp, hp, 0)
    R["E_proxy_stressed"] = proxy_comparison(rp, hp, 1)[0]
    fig_hawkes(g, h, rg, rh, op, R["E_proxy_calm"]["hawkes_count_mean_per_step"],
               os.path.join(OUT_DIR, "05_hawkes_vs_gaussian.png"))
    e = R["E_proxy_calm"]
    print(f"[INFO] calm proxies: ACF1 gaussian={e['gaussian']['acf1']:+.3f} hawkes={e['hawkes']['acf1']:+.3f}; "
          f"resid std ratio={e['resid_std_ratio_hawkes_over_gaussian']:.2f}; count Fano={e['hawkes_count_fano']:.2f}")

    R["F_hawkes_by_regime"] = hawkes_by_regime(ep_long, sp_long, hp, os.path.join(OUT_DIR, "06_hawkes_by_regime.png"))
    print(f"[INFO] Hawkes activity by regime: {R['F_hawkes_by_regime']}")

    hard = [R["A_regression"]["bit_for_bit"], b["hard_pass"], c["hard_pass_revert"], R["D_hawkes_process"]["hard_pass"]]
    R["ALL_HARD_PASS"] = bool(all(hard))
    with open(os.path.join(OUT_DIR, "stage1_validation_summary.json"), "w") as f:
        json.dump(R, f, indent=2, default=_jsonable)
    write_handoff(R, os.path.join(OUT_DIR, "stage1_handoff.md"))
    print(f"\nALL HARD CHECKS PASS: {R['ALL_HARD_PASS']}  ({hard})")
    print(f"Wrote figures, stage1_validation_summary.json and stage1_handoff.md to {OUT_DIR}")


if __name__ == "__main__":
    main()