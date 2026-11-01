"""
barger_lorig.py -- Stage 4 (Person 4): Barger & Lorig (2019) closed-form approximate benchmark.
NOT RUN by its author. Self-test:  python barger_lorig.py

SOURCE (read from arXiv:1804.04170, Int. J. Theor. Appl. Finance 22(2), 2019; eqs. numbers are the arXiv v1 ones)
  Model: dS = -g(b) nu dt + sigma dW (permanent impact), execution price S - f(a) nu (temporary impact), with
  da = mu(a)dt + omega(a)dB2, db = eta(b)dt + psi(b)dB1 (correlated scalar diffusions), f(a)=a, g(b)=b, CIR dynamics (4.2).
  Objective: expected terminal cash, liquidate by T (kappa->inf), no inventory penalty (phi->0).
  Zeroth-order strategy (3.29):  nu0 = q/(T-t)                               == time-weighted average (TWAP)
  First-order strategy  (3.30):  nu1 = q * [ 1/(T-t) + mu(a) f'(a) / (2 f(a)) + (T-t) eta(b) g'(b) / (6 f(a)) ]
                          (4.3, CIR): nu1 = q * [ 1/(T-t) + lam_a (theta_a - a)/(2a) + (T-t) lam_b (theta_b - b)/(6a) ]
  The paper notes nu1 can turn negative (buy) when impact is high and suggests truncating to max(0, nu).

MAPPING TO THE SIMULATOR (regime_simulator.py / simulator.py)
  b = kappa (permanent), g(b)=b:      S[t+1] = S[t] - kappa*v + ...   with v = shares per step = nu*tau  -> same kappa.
  Temporary impact: S_exec = S - alpha*v. Continuous version: f*nu = f*v/tau  =>  f(a) = tau*alpha_sim.
  So with a = tau*alpha the CIR parameters are lam, tau*theta, sigma*sqrt(tau) and every ratio in nu1 is unchanged:
    v_t = nu1*tau = q/(N-t) * [ 1 + T_rem*drift_alpha/(2*alpha) + T_rem^2*drift_kappa/(6*tau*alpha) ],
    T_rem = (N-t)*tau,  drift_alpha = lam_alpha*(theta_alpha - alpha),  drift_kappa = lam_kappa*(theta_kappa - kappa).
  test_unit_mapping() checks this algebra against the paper's formula evaluated with a = tau*alpha.

WHAT THIS BENCHMARK IS NOT (read before quoting it)
  * Their model has ONE set of CIR parameters. Our world switches regimes; so
      - "oracle_regime": true alpha, kappa AND the true regime's CIR parameters (best case for the formula),
      - "oracle_calm"  : true alpha, kappa but calm-regime parameters (what a single-regime deployment would use),
      - "filtered_mix" : IMM-PF posterior means with probability-weighted drifts (certainty-equivalent plug-in).
    Both are misspecified approximations of the true regime-switching optimum.
  * First-order truncation of an asymptotic expansion; objective is expected cost only (no risk term).
  * Parameters come from the simulator (uncalibrated). Last step sells the remainder (forced liquidation).
"""
from __future__ import annotations

import numpy as np


def params_from_regimes(sp, rp):
    """[(lam_alpha, theta_alpha, lam_kappa, theta_kappa)] per regime, from RegimeParams.effective_cir."""
    out = []
    for r in range(rp.n_regimes):
        e = rp.effective_cir(sp, r)
        out.append((e["lambda_alpha"], e["theta_alpha"], e["lambda_kappa"], e["theta_kappa"]))
    return out


def drifts(P, probs, alpha, kappa):
    """Probability-weighted CIR drifts (mean-reversion pull) of alpha and kappa at the given levels."""
    da = sum(float(p) * la * (ta - alpha) for p, (la, ta, lk, tk) in zip(probs, P))
    dk = sum(float(p) * lk * (tk - kappa) for p, (la, ta, lk, tk) in zip(probs, P))
    return da, dk


def bl_multiplier(t, N, tau, alpha, kappa, drift_a, drift_k):
    """First-order Barger-Lorig volume as a MULTIPLE of remaining-TWAP q/(N-t). 1.0 == TWAP == zeroth order."""
    T_rem = (N - t) * tau
    a = max(float(alpha), 1e-12)
    return 1.0 + T_rem * drift_a / (2.0 * a) + T_rem ** 2 * drift_k / (6.0 * tau * a)


def bl_volume(t, N, tau, q, alpha, kappa, P, probs, grid=None):
    """Shares to sell at step t. Returns (v, clipped_at_zero). Last step sells everything (forced liquidation).
    grid: optional tuple of multipliers to snap to (to compare on the Stage 3 action grid)."""
    if q <= 0:
        return 0.0, False
    if t >= N - 1:
        return float(q), False
    da, dk = drifts(P, probs, alpha, kappa)
    m = bl_multiplier(t, N, tau, alpha, kappa, da, dk)
    clipped = m < 0.0
    m = max(m, 0.0)                      # the paper's max(0, nu) truncation (no buying while liquidating)
    if grid is not None:
        g = np.asarray(grid, dtype=float)
        m = float(g[np.argmin(np.abs(g - m))])
    return float(min(m * q / (N - t), q)), bool(clipped)


# ---------------------------------------------------------------------------
def test_unit_mapping():
    """Compare bl_volume with the paper's eq. (4.3) evaluated in continuous units with a = tau*alpha."""
    errs = []
    N, tau, q = 50, 0.02, 12000.0
    lam_a, th_a, lam_k, th_k = 3.0, 1.5e-4, 2.0, 1.25e-5
    P = [(lam_a, th_a, lam_k, th_k)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = int(rng.integers(0, N - 1))
        alpha = th_a * float(rng.uniform(0.5, 3.0)); kappa = th_k * float(rng.uniform(0.5, 3.0))
        v, clipped = bl_volume(t, N, tau, q, alpha, kappa, P, [1.0])
        T_rem, a, b = (N - t) * tau, tau * alpha, kappa
        nu = q * (1.0 / T_rem + lam_a * (tau * th_a - a) / (2.0 * a) + T_rem * lam_k * (th_k - b) / (6.0 * a))
        v_paper = max(nu, 0.0) * tau                         # paper's truncation; per-step shares = nu * tau
        v_paper = min(v_paper, q)
        if abs(v - v_paper) > 1e-9 * max(1.0, abs(v_paper)):
            errs.append(f"mapping mismatch t={t}: {v} vs {v_paper}")
    # at the long-run means the correction vanishes -> exactly TWAP
    v, _ = bl_volume(10, N, tau, q, th_a, th_k, P, [1.0])
    if abs(v - q / (N - 10)) > 1e-9:
        errs.append("not TWAP at the long-run mean")
    # high temporary impact -> slow down; low -> speed up
    hi, _ = bl_volume(10, N, tau, q, 2.0 * th_a, th_k, P, [1.0])
    lo, _ = bl_volume(10, N, tau, q, 0.6 * th_a, th_k, P, [1.0])
    if not (hi < q / (N - 10) < lo):
        errs.append("direction of the temporary-impact correction is wrong")
    v, _ = bl_volume(N - 1, N, tau, q, th_a, th_k, P, [1.0])
    if v != q:
        errs.append("last step must liquidate the remainder")
    # truncation at zero for extreme alpha
    v, c = bl_volume(0, N, tau, q, 50.0 * th_a, th_k, P, [1.0])
    if not (c and v == 0.0):
        errs.append("negative first-order volume must be truncated to 0")
    return errs


if __name__ == "__main__":
    e = test_unit_mapping()
    print("barger_lorig self-test:", "PASS" if not e else f"FAIL {e[:3]}")