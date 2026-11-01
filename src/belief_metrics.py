"""
belief_metrics.py  (STAGE 2 -- Person 2)
----------------------------------------
Filter-agnostic scoring of BELIEFS against truth, so that every estimator (bootstrap PF,
IMM-PF, anything later) and every observation arm is scored by the same code.

Stage 0 (STAGE0_INTERFACES.md, Contract C4) asks for belief-quality metrics -- alpha RMSE,
coverage, regime-detection accuracy / Brier -- to live next to `interfaces.paired_metrics`
so all policies and filters are scored identically. This module is Person 2's proposed
version of that; Person 4 can import it or fold it into the shared metrics module.

Pure numpy/scipy. It knows nothing about particle filters: it takes arrays.

Conventions
    * All arrays are aligned in time: index t <-> decision step t <-> truth x_t, regime[t].
    * regime ids: 0 = calm, 1 = stressed (K = 2 worlds). p1 = filtered P(regime = 1).
    * "Onset" = a maximal run of stressed steps that STARTS at t >= 1 (a run that already
      began at t = 0 is left-censored and excluded). "Recovery" is the same for calm runs.
    * Detection delay at threshold th = first step inside the run with P(target regime) >= th,
      minus the run's first step. It is only defined for DETECTED runs, so it is optimistic on
      its own -- always read it together with `detect_rate`.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

LEVELS = (0.5, 0.8, 0.9, 0.95)


# ---------------------------------------------------------------------------
def _rmse(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def coverage(err, std, level: float = 0.9) -> float:
    """Fraction of steps where |err| < z_level * std (Gaussian central interval)."""
    err, std = np.asarray(err, dtype=float), np.asarray(std, dtype=float)
    if err.size == 0:
        return float("nan")
    zc = float(stats.norm.ppf(0.5 + level / 2.0))
    return float(np.mean(np.abs(err) / np.maximum(std, 1e-30) < zc))


def paired_diff(a, b) -> dict:
    """Paired difference a - b over episodes (same episodes, same order)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return dict(mean=float(d.mean()), se=se, n=int(len(d)), frac_a_lower=float(np.mean(d < 0)))


def beliefs_to_arrays(beliefs) -> dict:
    """list[interfaces.BeliefState] -> dict of arrays (duck-typed on the attribute names)."""
    out = dict(
        alpha_mean=np.array([b.alpha_mean for b in beliefs], dtype=float),
        alpha_std=np.array([b.alpha_std for b in beliefs], dtype=float),
        kappa_mean=np.array([b.kappa_mean for b in beliefs], dtype=float),
        kappa_std=np.array([b.kappa_std for b in beliefs], dtype=float),
        ess=np.array([b.ess for b in beliefs], dtype=float),
        p=np.array([b.regime_probs for b in beliefs], dtype=float),                  # (T, K)
        loglik=np.array([np.nan if b.loglik_increment is None else b.loglik_increment for b in beliefs], dtype=float),
    )
    if beliefs and beliefs[0].per_regime:
        out["ess_regime"] = np.array([[np.nan if rb.ess is None else rb.ess for rb in b.per_regime]
                                      for b in beliefs], dtype=float)                # (T, K)
    return out


# ---------------------------------------------------------------------------
class TrackingPool:
    """Accumulates per-episode tracking errors, then summarises them pooled over steps."""

    def __init__(self):
        self._a_err, self._a_std, self._k_err, self._k_std = [], [], [], []
        self._a_true, self._k_true, self._reg, self._win, self._ess = [], [], [], [], []
        self.ep_alpha_rmse, self.ep_kappa_rmse = [], []

    def add(self, a_mean, a_std, k_mean, k_std, a_true, k_true, regime, ess=None, window=None):
        a_mean, a_std = np.asarray(a_mean, dtype=float), np.asarray(a_std, dtype=float)
        k_mean, k_std = np.asarray(k_mean, dtype=float), np.asarray(k_std, dtype=float)
        n = a_mean.size
        a_true = np.asarray(a_true, dtype=float)[:n]
        k_true = np.asarray(k_true, dtype=float)[:n]
        ae, ke = a_mean - a_true, k_mean - k_true
        self._a_err.append(ae); self._a_std.append(a_std)
        self._k_err.append(ke); self._k_std.append(k_std)
        self._a_true.append(a_true); self._k_true.append(k_true)
        self._reg.append(np.asarray(regime, dtype=int)[:n])
        self._win.append(np.zeros(n, dtype=bool) if window is None else np.asarray(window, dtype=bool)[:n])
        self._ess.append(np.full(n, np.nan) if ess is None else np.asarray(ess, dtype=float)[:n])
        self.ep_alpha_rmse.append(_rmse(ae))
        self.ep_kappa_rmse.append(_rmse(ke))

    def summary(self, theta_alpha=None, theta_kappa=None) -> dict:
        if not self._a_err:
            return {}
        c = np.concatenate
        ae, asd, ke, ksd = c(self._a_err), c(self._a_std), c(self._k_err), c(self._k_std)
        at, kt, reg, win, ess = c(self._a_true), c(self._k_true), c(self._reg), c(self._win), c(self._ess)
        ep_a = np.asarray(self.ep_alpha_rmse)
        out = dict(
            n_episodes=int(len(ep_a)), n_steps=int(ae.size),
            alpha_rmse=_rmse(ae), kappa_rmse=_rmse(ke),
            alpha_cov90=coverage(ae, asd, 0.9), kappa_cov90=coverage(ke, ksd, 0.9),
            alpha_cov_curve={str(l): coverage(ae, asd, l) for l in LEVELS},
            ep_alpha_rmse_mean=float(ep_a.mean()),
            ep_alpha_rmse_sd=float(ep_a.std(ddof=1)) if len(ep_a) > 1 else float("nan"),
        )
        if theta_alpha is not None:
            out["alpha_rmse_naive_theta"] = _rmse(theta_alpha - at)     # "always guess the calm long-run mean"
        if theta_kappa is not None:
            out["kappa_rmse_naive_theta"] = _rmse(theta_kappa - kt)
        for r in (0, 1):
            m = reg == r
            if m.any():
                out[f"n_steps_regime{r}"] = int(m.sum())
                out[f"alpha_rmse_regime{r}"] = _rmse(ae[m])
                out[f"alpha_bias_regime{r}"] = float(ae[m].mean())      # signed: estimate - truth
                out[f"alpha_cov90_regime{r}"] = coverage(ae[m], asd[m], 0.9)
                out[f"kappa_rmse_regime{r}"] = _rmse(ke[m])
        if win.any():
            out["alpha_rmse_window"] = _rmse(ae[win])
            out["alpha_cov90_window"] = coverage(ae[win], asd[win], 0.9)
        fin = np.isfinite(ess)
        if fin.any():
            out["min_ess"] = float(ess[fin].min())
            out["median_ess"] = float(np.median(ess[fin]))
        return out


# ---------------------------------------------------------------------------
def _runs_starting(reg: np.ndarray, target: int):
    """Maximal runs of `target` that start at t >= 1. Returns [(start, end_exclusive, right_censored)]."""
    N, out, t = len(reg), [], 1
    while t < N:
        if reg[t] == target and reg[t - 1] != target:
            e = t
            while e < N and reg[e] == target:
                e += 1
            out.append((t, e, e == N))
            t = e
        else:
            t += 1
    return out


def spell_detection_stats(p1_list, reg_list, target: int, theta: float) -> dict:
    """How often and how fast a switch INTO `target` is declared (P(target) >= theta)."""
    delays, lengths, detected, pre = [], [], [], 0
    for p1, reg in zip(p1_list, reg_list):
        p = np.asarray(p1, dtype=float)
        p = p if target == 1 else 1.0 - p
        reg = np.asarray(reg, dtype=int)
        for s, e, _cens in _runs_starting(reg, target):
            hit = np.flatnonzero(p[s:e] >= theta)
            lengths.append(e - s)
            detected.append(bool(hit.size))
            pre += int(p[s - 1] >= theta)                    # already declared BEFORE the switch (early / false alarm)
            if hit.size:
                delays.append(int(hit[0]))
    n = len(lengths)
    det, lens = np.asarray(detected, dtype=bool), np.asarray(lengths)
    long = lens >= 5
    return dict(
        n_spells=int(n),
        detect_rate=float(det.mean()) if n else float("nan"),
        detect_rate_len_ge5=float(det[long].mean()) if long.any() else float("nan"),
        n_len_ge5=int(long.sum()),
        median_delay=float(np.median(delays)) if delays else float("nan"),
        mean_delay=float(np.mean(delays)) if delays else float("nan"),
        p90_delay=float(np.percentile(delays, 90)) if delays else float("nan"),
        pre_declared_frac=float(pre / n) if n else float("nan"),
        delays=[int(d) for d in delays],
    )


def steady_state_rates(p1_list, reg_list, theta: float, grace: int = 3) -> dict:
    """False-alarm rate in STEADY calm (calm now and for `grace` steps before) and recall in
    STEADY stress. Steady-state selection removes the unavoidable lag right after a switch."""
    fa_n = fa_d = rc_n = rc_d = 0
    for p1, reg in zip(p1_list, reg_list):
        p1, reg = np.asarray(p1, dtype=float), np.asarray(reg, dtype=int)
        for t in range(grace, len(reg)):
            win = reg[t - grace:t + 1]
            if np.all(win == 0):
                fa_d += 1; fa_n += int(p1[t] >= theta)
            elif np.all(win == 1):
                rc_d += 1; rc_n += int(p1[t] >= theta)
    return dict(false_alarm_rate=fa_n / fa_d if fa_d else float("nan"),
                stressed_recall=rc_n / rc_d if rc_d else float("nan"),
                n_steady_calm=int(fa_d), n_steady_stressed=int(rc_d))


def regime_detection_report(p1_list, reg_list, thresholds=(0.5, 0.9), grace: int = 3, n_bins: int = 10) -> dict:
    """Regime-detection accuracy report (Stage 2 deliverable).

    p1_list / reg_list: one array per episode, filtered P(stressed) and true regime, aligned.
    Reports (pooled over steps): MAP accuracy and its naive baseline, per-class recall,
    balanced accuracy, Brier score and its skill vs the best constant predictor, log-loss vs
    the best constant predictor, AUC, and a reliability table with ECE. Then, per threshold:
    onset/recovery detection rate and delay, and steady-state false-alarm rate / recall.
    """
    p1 = np.concatenate([np.asarray(x, dtype=float) for x in p1_list])
    y = np.concatenate([np.asarray(x, dtype=int) for x in reg_list])
    n = int(y.size)
    pi1 = float(np.mean(y == 1))
    pred = (p1 >= 0.5).astype(int)
    rec0 = float(np.mean(pred[y == 0] == 0)) if np.any(y == 0) else float("nan")
    rec1 = float(np.mean(pred[y == 1] == 1)) if np.any(y == 1) else float("nan")

    brier = float(np.mean((p1 - y) ** 2))
    brier_c = pi1 * (1.0 - pi1)
    eps = 1e-12
    pc = np.clip(p1, eps, 1.0 - eps)
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1.0 - pc)))
    logloss_c = float(-(pi1 * math.log(pi1) + (1 - pi1) * math.log(1 - pi1))) if 0.0 < pi1 < 1.0 else float("nan")

    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    auc = float("nan")
    if n1 and n0:
        auc = float(stats.mannwhitneyu(p1[y == 1], p1[y == 0], alternative="two-sided").statistic / (n1 * n0))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p1, edges[1:-1]), 0, n_bins - 1)
    cnt, pm, em = [], [], []
    for b in range(n_bins):
        m = which == b
        c = int(m.sum())
        cnt.append(c)
        pm.append(float(p1[m].mean()) if c else float("nan"))
        em.append(float(y[m].mean()) if c else float("nan"))
    ece = float(sum(c / n * abs(e - p) for c, p, e in zip(cnt, pm, em) if c > 0))

    per = {}
    for th in thresholds:
        per[str(th)] = dict(onset=spell_detection_stats(p1_list, reg_list, 1, th),
                            recovery=spell_detection_stats(p1_list, reg_list, 0, th),
                            **steady_state_rates(p1_list, reg_list, th, grace))
    return dict(
        n_steps=n, frac_stressed=pi1,
        acc=float(np.mean(pred == y)), acc_always_majority=float(max(pi1, 1.0 - pi1)),
        recall_calm=rec0, recall_stressed=rec1, bal_acc=0.5 * (rec0 + rec1),
        brier=brier, brier_const=brier_c, brier_skill=(1.0 - brier / brier_c) if brier_c > 0 else float("nan"),
        logloss=logloss, logloss_const=logloss_c, auc=auc, ece=ece,
        reliability=dict(edges=[float(e) for e in edges], count=cnt, pred=pm, emp=em),
        by_threshold=per,
    )
