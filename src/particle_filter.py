"""
particle_filter.py
-------------------
Bootstrap Sequential Monte Carlo (SMC) particle filter estimating the
latent liquidity state (kappa_t, alpha_t) using only the noisy proxies
from observation_model.py.

Four things worth flagging explicitly -- the first two were bugs, the
third is a standard technique, the fourth is a model-class fix found by
z-score calibration checks (see validation/ for the trail):

  1. Observation noise must be calibrated relative to the actual scale of
     particle-to-particle differences in kappa_t/alpha_t (~1e-5), not
     picked independently.
  2. The reported posterior for time t is computed from the REWEIGHTED,
     PRE-RESAMPLE particle set -- resampling is a variance-reduction step
     for the next propagation, not part of the Bayesian update itself.
  3. ROUGHENING (Gordon, Salmond & Smith 1993): even with (1) fixed,
     three observation channels compounding in the likelihood can still
     occasionally collapse weight onto one particle (ESS -> 1);
     resampling then duplicates that single particle across the whole
     cloud with no way to recover diversity. After each resample, a small
     Gaussian jitter (15% of each state's true CIR stationary std,
     computed from the known model parameters -- NOT from the particles'
     own empirical spread, since that's exactly what's collapsed) is
     added to every particle. This is standard practice for bootstrap
     filters with informative multi-channel likelihoods; the jitter scale
     is small enough not to bias the filter's real CIR process-noise
     dynamics.

  4. EXPLICIT JUMP COMPONENT (v2). v1 handled shocks with a "wide kernel"
     hack; z-score calibration showed it was confidently wrong for ~5 steps
     after a shock (ESS=1, std ~1e-14, kappa spuriously dragged). It is
     replaced by a proper jump-diffusion transition model: each step each
     particle may have alpha multiplied by LogUniform(0.5, 6), independent
     of kappa. Result: alpha RMSE 3.8e-5 -> 2.2e-6, shock-window 90%
     coverage 0.38 -> 0.91. LIMIT: shocks outside 0.5x-6x are outside the
     filter's prior and still collapse it (tested, not fixed).

Design:
  - State: x_t = (kappa_t, alpha_t), each particle is one hypothesis.
  - Transition (predict): same CIR SDE class as the true simulator (known
    theta/lambda/sigma/rho, sigma inflated x1.5), via Euler-Maruyama, plus
    the alpha jump component above. The filter does NOT know if/when a
    shock occurs.
  - Likelihood (update): Gaussian, same deterministic proxy-mean functions
    as observation_model.py, same known noise scales.
  - Resampling: systematic resampling, triggered when ESS < n/2, applied
    AFTER recording the current step's posterior, followed by roughening.
  - Ordering per interval t: reweight using obs at time t -> RECORD
    posterior for time t -> resample + roughen if needed -> propagate to t+1.

STATUS: milestone-1 build. Bootstrap SMC only (no Hawkes observation
model, no Interacting-Multiple-Model regime switching) -- both next-phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from observation_model import ObsParams


def predicted_proxy_means(kappa: np.ndarray, alpha: np.ndarray, op: ObsParams):
    """Vectorised, noise-free proxy means for an array of particles.
    Must mirror ObservationModel.observe_step's deterministic core
    exactly (relative-scale design -- see observation_model.py)."""
    alpha_rel = alpha / op.alpha_ref
    kappa_rel = kappa / op.kappa_ref

    spread = op.spread_base * alpha_rel
    volume = op.volume_base / np.maximum(alpha_rel, 1e-3)
    volume = np.maximum(volume, op.volume_floor)
    realvol = op.vol_base * kappa_rel
    return spread, volume, realvol


def _log_gaussian(x, mean, std):
    return -0.5 * np.log(2 * np.pi * std ** 2) - 0.5 * ((x - mean) / std) ** 2


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def _cir_stationary_std(theta, lam, sigma_x):
    """Stationary standard deviation of a CIR process: Var = sigma^2*theta/(2*lambda)."""
    return np.sqrt(sigma_x ** 2 * theta / (2 * lam))


@dataclass
class PFParams:
    n_particles: int = 5000
    ess_resample_threshold: float = 0.5
    init_kappa_std: float = 4.6875e-7
    init_alpha_std: float = 3.75e-6
    roughening_frac: float = 0.15   # jitter std, as a fraction of each state's CIR stationary std
    process_noise_inflation: float = 1.5   # mild, documented model mismatch on CIR sigma (normal-time robustness)

    # --- explicit JUMP component in the filter's transition model ---
    # Each step, each particle independently has probability `jump_prob` of
    # its alpha being multiplied by m ~ LogUniform(jump_mult_low, jump_mult_high),
    # applied AFTER the CIR diffusion (mirrors the simulator, where a shock
    # multiplies the state at time t). The filter is never told WHEN a shock
    # happens; this is simply part of its model class ("liquidity can jump").
    # Jumps hit alpha only and use independent draws, so they cannot drag
    # kappa around through the rho=0.9 diffusion correlation.
    #
    # Replaces an earlier `wide_frac/wide_inflation` hack (10% of particles
    # given a 40x wider correlated kernel). Diagnostics showed that hack
    # failed exactly when it mattered: on a 3x alpha jump the filter's
    # estimate lagged truth by ~10 steps, ESS collapsed to 1 for ~5 steps,
    # reported std fell to ~1e-14 while the error was ~1e-4 (z-scores up to
    # 1e9), and kappa was spuriously pushed up ~2x although the true kappa
    # never moved. That is confident-and-wrong, not "uncertainty widening".
    jump_prob: float = 0.08
    jump_mult_low: float = 0.5
    jump_mult_high: float = 6.0


class ParticleFilter:
    def __init__(
        self,
        sim_params,           # simulator.SimParams: known model class + known
                               # long-run params (theta, lambda, sigma, rho) --
                               # NOT the realised path, NOT knowledge of a shock.
        obs_params: ObsParams,
        pf_params: PFParams | None = None,
        seed: int | None = None,
    ):
        self.sp = sim_params
        self.op = obs_params
        self.pp = pf_params or PFParams()
        self.rng = np.random.default_rng(seed)

        n = self.pp.n_particles
        self.kappa = np.clip(self.rng.normal(self.sp.kappa0, self.pp.init_kappa_std, n), 1e-6, None)
        self.alpha = np.clip(self.rng.normal(self.sp.alpha0, self.pp.init_alpha_std, n), 1e-6, None)
        self.weights = np.full(n, 1.0 / n)

        cov = np.array([[1.0, self.sp.rho], [self.sp.rho, 1.0]])
        self._L = np.linalg.cholesky(cov)

        # filter's own (deliberately inflated) process noise -- see
        # PFParams.process_noise_inflation
        self._sigma_kappa_f = self.sp.sigma_kappa * self.pp.process_noise_inflation
        self._sigma_alpha_f = self.sp.sigma_alpha * self.pp.process_noise_inflation

        # fixed jitter scale for roughening -- based on the KNOWN model's
        # stationary spread, not the (potentially collapsed) particle spread
        self._jitter_kappa = self.pp.roughening_frac * _cir_stationary_std(
            self.sp.theta_kappa, self.sp.lambda_kappa, self.sp.sigma_kappa
        )
        self._jitter_alpha = self.pp.roughening_frac * _cir_stationary_std(
            self.sp.theta_alpha, self.sp.lambda_alpha, self.sp.sigma_alpha
        )

        self.history = {
            "kappa_mean": [], "kappa_std": [],
            "alpha_mean": [], "alpha_std": [],
            "ess": [],
        }
        self._record()  # index 0: prior before any observation

    def _record(self):
        w = self.weights
        k_mean = float(np.sum(w * self.kappa))
        a_mean = float(np.sum(w * self.alpha))
        k_std = float(np.sqrt(max(np.sum(w * (self.kappa - k_mean) ** 2), 0.0)))
        a_std = float(np.sqrt(max(np.sum(w * (self.alpha - a_mean) ** 2), 0.0)))
        ess = float(1.0 / np.sum(w ** 2))
        self.history["kappa_mean"].append(k_mean)
        self.history["kappa_std"].append(k_std)
        self.history["alpha_mean"].append(a_mean)
        self.history["alpha_std"].append(a_std)
        self.history["ess"].append(ess)

    def step(self, obs: dict):
        n = self.pp.n_particles
        tau = self.sp.tau

        # --- update (reweight) using the observation for the CURRENT particle time ---
        pred_spread, pred_volume, pred_realvol = predicted_proxy_means(self.kappa, self.alpha, self.op)
        log_lik = (
            _log_gaussian(obs["spread"], pred_spread, self.op.spread_noise_std)
            + _log_gaussian(obs["volume"], pred_volume, self.op.volume_noise_std)
            + _log_gaussian(obs["realvol"], pred_realvol, self.op.vol_noise_std)
        )
        log_lik -= log_lik.max()
        lik = np.exp(log_lik)

        w = self.weights * lik
        w_sum = w.sum()
        self.weights = (w / w_sum) if w_sum > 0 else np.full(n, 1.0 / n)

        # record the REWEIGHTED, pre-resample posterior -- the actual
        # filtering distribution for time t
        self._record()

        # resample + roughen AFTER recording
        ess = self.history["ess"][-1]
        if ess < self.pp.ess_resample_threshold * n:
            idx = systematic_resample(self.weights, self.rng)
            self.kappa = self.kappa[idx]
            self.alpha = self.alpha[idx]
            self.weights = np.full(n, 1.0 / n)

            # roughening: break exact duplication so the cloud can't collapse
            # to a literal point mass
            self.kappa = np.maximum(self.kappa + self.rng.normal(0, self._jitter_kappa, n), 1e-8)
            self.alpha = np.maximum(self.alpha + self.rng.normal(0, self._jitter_alpha, n), 1e-8)

        # --- propagate (predict) to prepare particles for the NEXT time step ---
        z = self.rng.normal(size=(n, 2)) * np.sqrt(tau)
        dB = z @ self._L.T
        dB1, dB2 = dB[:, 0], dB[:, 1]

        k_pos = np.maximum(self.kappa, 0.0)
        a_pos = np.maximum(self.alpha, 0.0)
        self.kappa = np.maximum(
            self.kappa + self.sp.lambda_kappa * (self.sp.theta_kappa - k_pos) * tau
            + self._sigma_kappa_f * np.sqrt(k_pos) * dB1,
            1e-8,
        )
        self.alpha = np.maximum(
            self.alpha + self.sp.lambda_alpha * (self.sp.theta_alpha - a_pos) * tau
            + self._sigma_alpha_f * np.sqrt(a_pos) * dB2,
            1e-8,
        )

        # jump component (alpha only, independent of the diffusion noise)
        is_jump = self.rng.random(n) < self.pp.jump_prob
        log_m = self.rng.uniform(np.log(self.pp.jump_mult_low), np.log(self.pp.jump_mult_high), n)
        self.alpha = self.alpha * np.where(is_jump, np.exp(log_m), 1.0)

    def run(self, obs_df) -> dict:
        for _, row in obs_df.iterrows():
            self.step({"spread": row["spread"], "volume": row["volume"], "realvol": row["realvol"]})
        return self.history


if __name__ == "__main__":
    # sanity check: python src/particle_filter.py
    from simulator import MarketSimulator
    from observation_model import ObservationModel

    sim = MarketSimulator(seed=3)
    ep = sim.simulate_episode(velocities=sim.twap_schedule(), shock_step=25, shock_jump_alpha_mult=3.0)
    obs_model = ObservationModel(seed=3)
    obs_df = obs_model.observe_episode(ep, seed=3)

    pf = ParticleFilter(sim.p, obs_model.p, seed=3)
    hist = pf.run(obs_df)

    k_mean = np.array(hist["kappa_mean"][1:])
    a_mean = np.array(hist["alpha_mean"][1:])
    k_rmse = np.sqrt(np.mean((k_mean - ep.kappa[: len(k_mean)]) ** 2))
    a_rmse = np.sqrt(np.mean((a_mean - ep.alpha[: len(a_mean)]) ** 2))
    print(f"kappa RMSE: {k_rmse:.6f}   alpha RMSE: {a_rmse:.6f}")
    print(f"ESS range: [{min(hist['ess']):.0f}, {max(hist['ess']):.0f}]  (n_particles={pf.pp.n_particles})")
    a_std = hist["alpha_std"][1:]
    print(f"alpha_std just before shock (step 24): {a_std[24]:.6f}")
    print(f"alpha_std peak, steps 25-34:            {max(a_std[25:35]):.6f}")
    print(f"alpha_std at step 45:                    {a_std[45]:.6f}")