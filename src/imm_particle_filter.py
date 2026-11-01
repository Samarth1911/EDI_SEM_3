"""
imm_particle_filter.py  (STAGE 2 -- Person 2)
---------------------------------------------
Interacting-Multiple-Model Particle Filter (IMM-PF) for the regime-switching
latent-liquidity world built in Stage 1 (regime_simulator.py).

The validated bootstrap filter in particle_filter.py is NOT modified. This module
re-uses its public helper `predicted_proxy_means` (so the likelihood is the same
function of (kappa, alpha)) and copies its transition/roughening/jump design, but
runs one particle bank per regime hypothesis and lets the banks interact.

WHAT IT DOES (one call to `update(obs)` = one filtering step)
    State per bank j (regime hypothesis s_t = j):  particles x_t = (kappa_t, alpha_t)
    with within-bank weights w[j, :], plus the regime probability mu[j].

    1. UPDATE. Every particle gets the Gaussian proxy log-likelihood (spread, volume,
       realvol -- the three CORE channels only, identical maths to the bootstrap PF).
         L_j          = sum_i w[j,i] * p(y_t | x_{j,i})              (bank marginal likelihood)
         mu_j(t)      ~ mu_prior_j * L_j                              (regime posterior)
         w[j,i](t)    ~ w[j,i] * p(y_t | x_{j,i})                     (within-bank posterior)
       The belief for step t is recorded HERE: reweighted, pre-resample (Stage 0 convention).
    2. INTERACTION (Blom & Bar-Shalom 1988 mixing, realised with particles).
         mu_pred_j    = sum_i P[i,j] mu_i(t)                          (prior for s_{t+1})
         M[i,j]       = P[i,j] mu_i(t) / mu_pred_j                    (mixing weights)
       The particle set that starts regime j's next prediction is drawn from the UNION
       of all banks' posterior particles with weights M[i,j] * w[i,:]  (systematic
       resampling). This is what lets bank j inherit particles from bank i when the
       regime is likely to have switched i -> j.
    3. ROUGHENING (Gordon-Salmond-Smith 1993) after every resample, jitter =
       roughening_frac x the regime's OWN CIR stationary std (known-model value, not
       the particles' empirical spread -- same reasoning as particle_filter.py).
    4. PREDICTION. Bank j's particles are propagated one Euler-Maruyama CIR step with
       regime j's effective parameters (RegimeParams.effective_cir), then the alpha
       jump component of the bootstrap PF is applied (see IMMParams.jump_prob).

TIMING / SEMANTICS (matches STAGE0_INTERFACES.md and regime_simulator.py)
    regime[t] is the regime that generated x_t from x_{t-1}; x_{t+1} = f(x_t, s_{t+1});
    y_t depends on x_t. So the filtered `regime_probs` in the BeliefState for step t are
    p(s_t = j | y_0..y_t)  -- FILTERED, not one-step-ahead. The one-step-ahead
    prediction p(s_{t+1} | y_0..y_t) is available afterwards as `next_regime_probs`.
    Before the first update the regime prior is RegimeParams.init_probs.

BELIEF CONTRACT (interfaces.BeliefState, K = number of regimes)
    * regime_probs: filtered posterior, sums to 1.
    * per_regime[j]: mean/std of (kappa, alpha) CONDITIONAL on regime j, its prob, and
      its within-bank ESS.
    * kappa_mean, kappa_std, alpha_mean, alpha_std: moments of the MIXTURE over regimes
      (between-regime spread is included in the std).
    * ess: DECISION 3 as proposed in Stage 0 -- ESS of the COMBINED weighted particle set,
      i.e. 1 / sum_j sum_i (mu_j * w[j,i])^2, in [1, K*n_particles].

RESAMPLING RULE (a choice, flagged)
    K > 1: the interaction step resamples every bank at every step (it has to, to mix
           particles across banks). Adaptive ESS-triggered resampling is NOT used.
    K = 1: mixing is the identity, so the filter falls back to the bootstrap rule
           (ESS-triggered systematic resampling + roughening). With RegimeParams.single(),
           the same PFParams-equivalent settings and jump_prob = 0.08 it is the same
           algorithm as particle_filter.ParticleFilter, and it draws random numbers in the
           same order, so the two are expected to agree very closely (the quick run gave
           RMSE ratios of 1.000); validate_imm_pf.py checks this with a loose tolerance.

DEFAULTS ARE NOT TUNED. n_particles, jump_prob, roughening_frac and obs_noise_inflation
are chosen by reasoning (see IMMParams comments), not fitted. validate_imm_pf.py sweeps
jump_prob and obs_noise_inflation so the choice can be made from data.

WHAT IS NOT IMPLEMENTED (deliberately, so nobody assumes it is):
    * The likelihood reads only spread/volume/realvol. The extras of the Hawkes models
      (spread_ticks, n_open, n_close, n_events, hawkes_excitation) are ignored, so on the
      Hawkes arms the filter's Gaussian likelihood is misspecified (Stage 1 measured
      residual std 1.36x-1.41x the assumed noise) and it will tend to be overconfident.
    * The observation likelihood is the same function of x in every bank. Regime evidence
      therefore comes from the regime-specific DYNAMICS only (different theta/lambda/sigma),
      not from regime-specific observation models.
    * The mid-price return is not part of the C2 contract, so the regime information in
      price volatility (Stage 1: about 1.58x in stress) is unreachable for this filter.
    * P and the per-regime CIR parameters are taken as KNOWN (Stage 0 DECISION 2). They are
      not learned. Misspecified-model sensitivity runs live in validate_imm_pf.py.
    * Particle counts per bank are fixed and equal.

REFERENCES (I have not re-verified either paper line by line; treat "follows" as "in the
spirit of"): Blom & Bar-Shalom (1988), IMM estimator; Boers & Driessen (2003), particle-filter
IMM. This code is the per-mode-PF-with-mixing variant asked for in the project brief, not a
transcription of Boers & Driessen's algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from simulator import SimParams
from observation_model import ObsParams
from regime_simulator import RegimeParams
from particle_filter import predicted_proxy_means
from interfaces import Observation, BeliefState, RegimeBelief


# ---------------------------------------------------------------------------
@dataclass
class IMMParams:
    n_particles: int = 2000               # PER REGIME BANK (total = K * n_particles)
    ess_resample_threshold: float = 0.5   # used only when K == 1 (bootstrap fallback)
    init_kappa_std: float = 4.6875e-7     # same as PFParams; scaled per regime (see __init__)
    init_alpha_std: float = 3.75e-6
    roughening_frac: float = 0.15         # jitter std / regime CIR stationary std (same as PFParams)
    process_noise_inflation: float = 1.5  # mild CIR-sigma mismatch, same as PFParams

    # alpha jump component (same mechanism as PFParams). It keeps injected shocks
    # (dashboard shock control, simulator shock_step) inside the filter's model class.
    # TRADE-OFF: a jump-capable calm bank can partly "explain" a stressed level, which
    # weakens regime discrimination. 0.03 is a reasoned middle between 0 and the
    # bootstrap PF's validated 0.08 -- NOT tuned. validate_imm_pf.py sweeps {0, 0.03, 0.08}.
    jump_prob: float = 0.03
    jump_mult_low: float = 0.5
    jump_mult_high: float = 6.0

    # >1 widens the assumed proxy noise. The default 1.0 is the validated Gaussian model.
    # On Hawkes data the true residual std is about 1.4x the assumed value (Stage 1), so
    # 1.4 is the natural value to try; it is swept in validation, not applied by default.
    obs_noise_inflation: float = 1.0

    regime_prob_floor: float = 1e-12      # numerical floor so no regime is exactly 0 forever


# ---------------------------------------------------------------------------
def imm_mixing(mu: np.ndarray, P: np.ndarray):
    """IMM interaction algebra.

    mu[i]  = p(s_t = i | y_0..y_t)                  (filtered)
    P[i,j] = Pr(s_{t+1} = j | s_t = i)
    Returns (mu_pred, M):
      mu_pred[j] = sum_i P[i,j] mu[i]               = p(s_{t+1} = j | y_0..y_t)
      M[i,j]     = P[i,j] mu[i] / mu_pred[j]        = p(s_t = i | s_{t+1} = j, y_0..y_t)
    Columns of M sum to 1. If mu_pred[j] == 0 (regime j unreachable) the column falls back
    to mu so that bank j still receives valid particles.
    """
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)
    joint = mu[:, None] * P
    mu_pred = joint.sum(axis=0)
    M = joint / np.maximum(mu_pred, 1e-300)[None, :]
    dead = mu_pred <= 1e-300
    if np.any(dead):
        M[:, dead] = mu[:, None]
    return mu_pred, M


def _systematic_indices(weights: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: m indices from a weight vector of ANY length."""
    w = np.asarray(weights, dtype=float)
    c = np.cumsum(w)
    c = c / c[-1]
    c[-1] = 1.0
    pos = (rng.random() + np.arange(m)) / m
    return np.minimum(np.searchsorted(c, pos), len(w) - 1)


def _log_gauss(x, mean, std: float):
    return -0.5 * math.log(2.0 * math.pi * std * std) - 0.5 * ((x - mean) / std) ** 2


def _cir_stationary_std(theta, lam, sigma):
    return np.sqrt(sigma ** 2 * theta / (2.0 * lam))


# ---------------------------------------------------------------------------
class IMMParticleFilter:
    """Implements interfaces.Estimator:  reset(seed), update(obs) -> BeliefState.

    sim_params    : simulator.SimParams (base CIR parameters, tau, kappa0/alpha0)
    obs_params    : observation_model.ObsParams (proxy maps and noise scales; must match
                    the observation model that generated the data)
    regime_params : regime_simulator.RegimeParams -- the filter's regime model (P, per-regime
                    CIR multipliers, init_probs). Passing the TRUE one is Stage 0 DECISION 2's
                    default; pass a different one to run a misspecification test.
    """

    def __init__(self, sim_params: SimParams, obs_params: ObsParams, regime_params: RegimeParams,
                 imm_params: IMMParams | None = None, seed: int | None = None):
        self.sp = sim_params
        self.op = obs_params
        self.rp = regime_params
        self.pp = imm_params or IMMParams()
        self.K = int(self.rp.n_regimes)
        self.n = int(self.pp.n_particles)
        self.P = np.asarray(self.rp.P, dtype=float)
        self.seed = seed

        eff = [self.rp.effective_cir(self.sp, r) for r in range(self.K)]

        def col(key, scale=1.0):
            return (np.array([e[key] for e in eff], dtype=float) * scale)[:, None]     # (K, 1)

        infl = self.pp.process_noise_inflation
        self._theta_k, self._lam_k = col("theta_kappa"), col("lambda_kappa")
        self._theta_a, self._lam_a = col("theta_alpha"), col("lambda_alpha")
        self._sig_k_f = col("sigma_kappa", infl)       # filter's (inflated) process noise
        self._sig_a_f = col("sigma_alpha", infl)

        # regime-specific stationary std -> roughening scale and initial spread
        stat_k = _cir_stationary_std(self._theta_k, self._lam_k, col("sigma_kappa"))    # (K,1)
        stat_a = _cir_stationary_std(self._theta_a, self._lam_a, col("sigma_alpha"))
        self._jit_k = self.pp.roughening_frac * stat_k
        self._jit_a = self.pp.roughening_frac * stat_a
        base_k = float(_cir_stationary_std(self.sp.theta_kappa, self.sp.lambda_kappa, self.sp.sigma_kappa))
        base_a = float(_cir_stationary_std(self.sp.theta_alpha, self.sp.lambda_alpha, self.sp.sigma_alpha))
        self._init_k_std = self.pp.init_kappa_std * stat_k / base_k                      # (K,1)
        self._init_a_std = self.pp.init_alpha_std * stat_a / base_a

        cov = np.array([[1.0, self.sp.rho], [self.sp.rho, 1.0]])
        self._L = np.linalg.cholesky(cov)

        self.reset(seed)

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        K, n, sp = self.K, self.n, self.sp
        # each bank starts at ITS regime's own level (the simulator starts there too)
        k0 = np.array([sp.kappa0 * s.theta_kappa_mult for s in self.rp.specs], dtype=float)[:, None]
        a0 = np.array([sp.alpha0 * s.theta_alpha_mult for s in self.rp.specs], dtype=float)[:, None]
        self.kappa = np.clip(self.rng.normal(k0, self._init_k_std, size=(K, n)), 1e-6, None)
        self.alpha = np.clip(self.rng.normal(a0, self._init_a_std, size=(K, n)), 1e-6, None)
        self.w = np.full((K, n), 1.0 / n)
        self.mu_prior = np.asarray(self.rp.init_probs, dtype=float).copy()   # p(s_t | y_0..t-1)
        self.mu = self.mu_prior.copy()                                       # posterior after last update
        self.beliefs: list = []
        self._t = 0

    @property
    def next_regime_probs(self) -> np.ndarray:
        """p(s_{t+1} | y_0..y_t) after the latest update (before any update: init_probs)."""
        return self.mu_prior.copy()

    # ------------------------------------------------------------------
    def update(self, obs) -> BeliefState:
        """Consume the proxies for step t and return the filtering belief for step t.
        `obs` is an interfaces.Observation (only its core channels are read) or a dict
        with spread/volume/realvol."""
        if hasattr(obs, "core_dict"):
            y, t = obs.core_dict(), int(obs.t)
        else:
            y, t = obs, self._t
        op, infl = self.op, self.pp.obs_noise_inflation

        ps, pv, pr = predicted_proxy_means(self.kappa, self.alpha, op)               # each (K, n)
        ll = (_log_gauss(y["spread"], ps, op.spread_noise_std * infl)
              + _log_gauss(y["volume"], pv, op.volume_noise_std * infl)
              + _log_gauss(y["realvol"], pr, op.vol_noise_std * infl))

        with np.errstate(divide="ignore"):
            a = np.log(self.w) + ll                                                  # (K, n)
            logL = logsumexp(a, axis=1)                                              # (K,) bank marginal log-lik
            log_joint = np.log(self.mu_prior) + logL
        loglik_inc = float(logsumexp(log_joint))

        if np.isfinite(loglik_inc):
            mu = np.exp(log_joint - loglik_inc)
            w_post = np.exp(a - np.where(np.isfinite(logL), logL, 0.0)[:, None])
            bad = ~np.isfinite(logL)                                                 # bank with no usable particle
            if np.any(bad):
                w_post[bad] = self.w[bad]
        else:                                                                        # degenerate: keep the prior
            mu, w_post, loglik_inc = self.mu_prior.copy(), self.w.copy(), None   # None, not NaN: json.dumps would emit bare NaN

        mu = np.maximum(mu, self.pp.regime_prob_floor)
        self.mu = mu / mu.sum()
        self.w = w_post

        belief = self._make_belief(t, loglik_inc)      # reweighted, PRE-resample posterior
        self.beliefs.append(belief)
        self._predict()                                # interaction -> resample -> propagate to t+1
        self._t += 1
        return belief

    def run(self, obs_df) -> list:
        """Convenience: filter a whole observation DataFrame (rows = steps 0..N-1)."""
        return [self.update(Observation.from_dict(i, r)) for i, r in enumerate(obs_df.to_dict("records"))]

    # ------------------------------------------------------------------
    def _make_belief(self, t: int, loglik_inc: float) -> BeliefState:
        w, kap, alp, mu = self.w, self.kappa, self.alpha, self.mu
        km, am = (w * kap).sum(axis=1), (w * alp).sum(axis=1)                        # (K,)
        kv = np.maximum((w * (kap - km[:, None]) ** 2).sum(axis=1), 0.0)
        av = np.maximum((w * (alp - am[:, None]) ** 2).sum(axis=1), 0.0)
        ess_j = 1.0 / (w ** 2).sum(axis=1)
        ess_tot = 1.0 / float(((mu[:, None] * w) ** 2).sum())                        # combined weighted set

        k_mean, a_mean = float((mu * km).sum()), float((mu * am).sum())
        # mixture variance = within-regime + between-regime (no cancellation)
        k_var = float((mu * kv).sum() + (mu * (km - k_mean) ** 2).sum())
        a_var = float((mu * av).sum() + (mu * (am - a_mean) ** 2).sum())

        per = [RegimeBelief(regime=j, prob=float(mu[j]), kappa_mean=float(km[j]), kappa_std=float(math.sqrt(kv[j])),
                            alpha_mean=float(am[j]), alpha_std=float(math.sqrt(av[j])), ess=float(ess_j[j]))
               for j in range(self.K)]
        return BeliefState(t=t, regime_probs=[float(x) for x in mu], kappa_mean=k_mean,
                           kappa_std=math.sqrt(max(k_var, 0.0)), alpha_mean=a_mean,
                           alpha_std=math.sqrt(max(a_var, 0.0)), ess=float(ess_tot), per_regime=per,
                           n_particles=self.K * self.n, loglik_increment=loglik_inc)

    def _roughen(self) -> None:
        K, n = self.K, self.n
        self.kappa = np.maximum(self.kappa + self.rng.normal(0.0, 1.0, (K, n)) * self._jit_k, 1e-8)
        self.alpha = np.maximum(self.alpha + self.rng.normal(0.0, 1.0, (K, n)) * self._jit_a, 1e-8)

    def _predict(self) -> None:
        K, n, rng = self.K, self.n, self.rng
        mu_pred, M = imm_mixing(self.mu, self.P)

        if K == 1:                                   # bootstrap fallback: adaptive resampling
            if 1.0 / float(np.sum(self.w[0] ** 2)) < self.pp.ess_resample_threshold * n:
                idx = _systematic_indices(self.w[0], n, rng)
                self.kappa, self.alpha = self.kappa[:, idx], self.alpha[:, idx]
                self.w = np.full((1, n), 1.0 / n)
                self._roughen()
        else:                                        # interaction: mix banks by resampling from the union
            kf, af = self.kappa.reshape(-1), self.alpha.reshape(-1)
            new_k, new_a = np.empty_like(self.kappa), np.empty_like(self.alpha)
            for j in range(K):
                cand = (M[:, j][:, None] * self.w).reshape(-1)                       # weights over K*n candidates
                idx = _systematic_indices(cand, n, rng)
                new_k[j], new_a[j] = kf[idx], af[idx]
            self.kappa, self.alpha = new_k, new_a
            self.w = np.full((K, n), 1.0 / n)
            self._roughen()

        self.mu_prior = mu_pred
        self._propagate()

    def _propagate(self) -> None:
        """One Euler-Maruyama CIR step per bank with that regime's parameters, then jumps."""
        K, n, rng, tau, pp = self.K, self.n, self.rng, self.sp.tau, self.pp
        z = rng.normal(size=(K, n, 2)) * math.sqrt(tau)
        dB1 = self._L[0, 0] * z[..., 0] + self._L[0, 1] * z[..., 1]                  # == (z @ L.T)[..., 0]
        dB2 = self._L[1, 0] * z[..., 0] + self._L[1, 1] * z[..., 1]

        k_pos, a_pos = np.maximum(self.kappa, 0.0), np.maximum(self.alpha, 0.0)
        self.kappa = np.maximum(self.kappa + self._lam_k * (self._theta_k - k_pos) * tau
                                + self._sig_k_f * np.sqrt(k_pos) * dB1, 1e-8)
        self.alpha = np.maximum(self.alpha + self._lam_a * (self._theta_a - a_pos) * tau
                                + self._sig_a_f * np.sqrt(a_pos) * dB2, 1e-8)

        if pp.jump_prob > 0.0:                                                        # alpha-only jumps
            is_jump = rng.random((K, n)) < pp.jump_prob
            log_m = rng.uniform(math.log(pp.jump_mult_low), math.log(pp.jump_mult_high), (K, n))
            self.alpha = self.alpha * np.where(is_jump, np.exp(log_m), 1.0)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # quick look, run from src/:  python imm_particle_filter.py
    from regime_simulator import RegimeSwitchingSimulator
    from spread_hawkes import make_observation_model_v2

    sp = SimParams()
    rp = RegimeParams.default()
    sim = RegimeSwitchingSimulator(sp, rp, seed=11)
    ep = sim.simulate_episode(velocities=sim.twap_schedule(), seed=11)
    om = make_observation_model_v2("gaussian", seed=10_011, tau=sp.tau)
    df = om.observe_episode(ep, seed=10_011)
    f = IMMParticleFilter(sp, ObsParams(), rp, seed=20_011)
    bel = f.run(df)
    print("true regime :", "".join(map(str, ep.regime[:len(bel)])))
    print("MAP regime  :", "".join(str(int(np.argmax(b.regime_probs))) for b in bel))
    print("p(stressed) :", " ".join(f"{b.regime_probs[1]:.2f}" for b in bel[:12]), "...")
    a_err = np.array([b.alpha_mean for b in bel]) - ep.alpha[:len(bel)]
    print(f"alpha RMSE: {np.sqrt(np.mean(a_err ** 2)):.3e}   ESS range [{min(b.ess for b in bel):.0f}, {max(b.ess for b in bel):.0f}]")
