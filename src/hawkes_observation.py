"""
hawkes_observation.py  (STAGE 1 -- Person 1)
--------------------------------------------
Hawkes-process-driven observation model: a SELECTABLE ALTERNATIVE to the
validated Gaussian ObservationModel in observation_model.py (which is NOT
modified). Same output columns (spread, volume, realvol) so the particle
filter can consume either; extra diagnostic columns are added.

WHAT IS IMPLEMENTED
    A univariate, linear Hawkes process of "liquidity-stress events"
    (spread-widening events) with exponential kernel
        phi(u) = a * beta * exp(-beta * u),      integral(phi) = a = branching ratio n
        lambda(s) = mu_r + sum_{t_i < s} phi(s - t_i)
    simulated by Ogata's modified thinning algorithm. Stability <=> n < 1
    (enforced at construction). Mean intensity = mu / (1 - n).
    Per simulator step, the per-step event count n_t drives the proxies:
        spread_t  = spread_base*alpha_rel  + spread_jump*(n_t - m_ref) + noise
        volume_t  = volume_base/alpha_rel  - volume_jump*(n_t - m_ref) + noise
        realvol_t = vol_base*kappa_rel     + vol_jump  *(n_t - m_ref) + noise
    m_ref = expected count per step in the CALM regime, so in calm/stationary
    conditions the Hawkes and Gaussian proxies have the SAME MEAN (moment-
    matched); the Hawkes version adds clustered, autocorrelated, right-skewed
    residuals. Baseline mu and branching ratio depend on the ACTIVE REGIME
    (stressed = higher baseline and higher branching, i.e. closer to
    criticality -- the "endogenous liquidity crisis" flavour).

TIMING CONVENTION (matches regime_simulator.py)
    Row t of the observation table uses the events in the window
    [t*tau, (t+1)*tau), with baseline set by regime[t] and proxies centred on
    x_t = (kappa[t], alpha[t]). This is the same "proxy[t] is a function of
    x_t" convention as the Gaussian model, BUT the count is accumulated over
    the step following t, so it carries a within-step look-ahead. It does not
    leak trade outcomes (events are independent of v_t). FLAGGED for Stage 0.

HONEST STATUS / WHAT IS NOT CLAIMED
    - Parameters (mu0, branching, beta, jumps) are ILLUSTRATIVE, NOT calibrated
      to data. Loadings were chosen by back-of-envelope so the Hawkes-driven
      variance is comparable to the Gaussian noise variance in calm; the
      validation script reports the realised numbers.
    - Bacry-Delattre-Hoffmann-Muzy (2013) use MUTUALLY exciting processes for
      up/down price ticks; this is a simpler univariate version for spread
      events. The specific feedback mechanism in Fosset-Bouchaud-Benzaquen
      (endogenous liquidity crises) is NOT reproduced -- only the generic
      "spread/liquidity events as a self-exciting point process" framing.
      Check both papers before citing this as an implementation of them.
    - Events do not respond to our own trading (no impact feedback loop).
    - The particle filter's Gaussian likelihood is MISSPECIFIED for these
      proxies (non-Gaussian, autocorrelated residuals, and in the stressed
      regime a mean shift beyond what alpha explains). That is the intended
      ablation, but Stage 2 should expect a degraded / overconfident filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator import Episode
from observation_model import ObservationModel, ObsParams

HAWKES_RNG_TAG = 0x4A57


# ---------------------------------------------------------------------------
@dataclass
class HawkesParams:
    mu0: float = 50.0                 # baseline event rate (events per unit time), calm
    mu_mult: tuple = (1.0, 1.5)       # per-regime multiplier on mu0
    branching: tuple = (0.5, 0.75)    # per-regime branching ratio n_r = a_r (must be < 1)
    beta: float = 25.0                # kernel decay rate; memory 1/beta = 0.04 = 2 steps at tau=0.02
    spread_jump: float = 0.001        # $ added to spread per excess event
    volume_jump: float = 60.0         # shares of volume removed per excess event
    vol_jump: float = 0.0004          # realised-vol proxy added per excess event
    warmup_steps: int = 50            # burn-in (calm baseline) so t=0 is not a cold start
    allow_unstable: bool = False      # only for demonstrating instability

    def __post_init__(self):
        self.mu_mult = tuple(float(x) for x in self.mu_mult)
        self.branching = tuple(float(x) for x in self.branching)
        if len(self.mu_mult) != len(self.branching):
            raise ValueError("mu_mult and branching must have one entry per regime")
        if self.mu0 <= 0 or self.beta <= 0 or any(m <= 0 for m in self.mu_mult):
            raise ValueError("mu0, beta, mu_mult must be > 0")
        if any(a < 0 for a in self.branching):
            raise ValueError("branching ratios must be >= 0")
        if not self.allow_unstable and any(a >= 1.0 for a in self.branching):
            raise ValueError(f"unstable Hawkes: branching ratio must be < 1, got {self.branching}")

    @property
    def n_regimes(self) -> int:
        return len(self.branching)

    def baseline(self, r: int) -> float:
        return self.mu0 * self.mu_mult[r]

    def mean_intensity(self, r: int) -> float:
        return self.baseline(r) / (1.0 - self.branching[r])

    def expected_count(self, r: int, tau: float) -> float:
        return self.mean_intensity(r) * tau

    def fano_asymptotic(self, r: int) -> float:
        """Long-window Fano factor 1/(1-n)^2 (Poisson = 1). Over a SHORT window
        such as one step the realised Fano factor is smaller."""
        return 1.0 / (1.0 - self.branching[r]) ** 2


# ---------------------------------------------------------------------------
class HawkesProcess:
    """Exponential-kernel Hawkes process with a piecewise-constant baseline,
    simulated by Ogata thinning. State = excitation E(s) = sum_i a*beta*exp(-beta*(s-t_i)),
    so lambda(s) = mu + E(s). Between events lambda is non-increasing, so the
    intensity just after the last candidate time is a valid upper bound."""

    def __init__(self, beta: float, rng: np.random.Generator):
        self.beta = float(beta)
        self.rng = rng
        self.E = 0.0

    def advance(self, mu: float, a: float, dt: float) -> np.ndarray:
        """Simulate the window (0, dt] with baseline mu and branching a.
        Returns event times RELATIVE to the window start; updates self.E to the
        end of the window (so consecutive calls form one continuous process)."""
        beta, rng = self.beta, self.rng
        E, s = self.E, 0.0
        events = []
        while True:
            M = mu + E                              # upper bound until next event
            w = rng.exponential(1.0 / M)            # candidate waiting time
            if s + w >= dt:
                E *= math.exp(-beta * (dt - s))
                break
            s += w
            E *= math.exp(-beta * w)
            if rng.random() * M <= mu + E:          # accept w.p. lambda(s)/M
                events.append(s)
                E += a * beta
        self.E = E
        return np.asarray(events, dtype=float)


def hawkes_compensator_increments(times, mu, a, beta, E0: float = 0.0) -> np.ndarray:
    """Integrated intensity between consecutive events (from t=0), constant mu.
    Under a correctly simulated Hawkes process these are iid Exp(1)
    (random time-change theorem) -- the standard goodness-of-fit test."""
    t = np.asarray(times, dtype=float)
    out = np.empty(len(t))
    E, prev = E0, 0.0
    for i, ti in enumerate(t):
        d = ti - prev
        out[i] = mu * d + E * (1.0 - math.exp(-beta * d)) / beta
        E = E * math.exp(-beta * d) + a * beta
        prev = ti
    return out


# ---------------------------------------------------------------------------
class GaussianObservation(ObservationModel):
    """The existing validated Gaussian model with the unified interface
    (regime kwarg accepted and ignored; regime column added to episode tables)."""
    kind = "gaussian"

    def observe_step(self, kappa_t, alpha_t, regime: int = 0, rng=None) -> dict:
        return super().observe_step(kappa_t, alpha_t, rng=rng)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def observe_episode(self, episode: Episode, seed=None) -> pd.DataFrame:
        df = super().observe_episode(episode, seed=seed)
        reg = getattr(episode, "regime", None)
        df["regime"] = np.zeros(len(df), dtype=int) if reg is None else np.asarray(reg)[:len(df)]
        return df


class HawkesObservationModel:
    """Stateful: the excitation carries across steps, so call observe_step in
    time order t=0,1,2,... after reset(). observe_episode() resets for you."""
    kind = "hawkes"

    def __init__(self, obs_params: ObsParams | None = None,
                 hawkes_params: HawkesParams | None = None,
                 tau: float = 0.02, seed: int | None = None):
        self.p = obs_params or ObsParams()
        self.h = hawkes_params or HawkesParams()
        self.tau = float(tau)
        self.seed = seed
        self.m_ref = self.h.expected_count(0, self.tau)
        self.reset(seed)

    def reset(self, seed=None):
        s = self.seed if seed is None else seed
        self.rng = np.random.default_rng(None if s is None else [int(s), HAWKES_RNG_TAG])
        self.proc = HawkesProcess(self.h.beta, self.rng)
        for _ in range(self.h.warmup_steps):      # burn-in at calm baseline
            self.proc.advance(self.h.baseline(0), self.h.branching[0], self.tau)

    def observe_step(self, kappa_t: float, alpha_t: float, regime: int = 0, rng=None) -> dict:
        p, h = self.p, self.h
        r = int(regime)
        if r >= h.n_regimes:
            raise IndexError(f"regime {r} but HawkesParams defines {h.n_regimes} regimes")
        noise = rng or self.rng
        ev = self.proc.advance(h.baseline(r), h.branching[r], self.tau)
        n = len(ev)
        excess = n - self.m_ref

        alpha_rel = alpha_t / p.alpha_ref
        kappa_rel = kappa_t / p.kappa_ref

        spread = p.spread_base * alpha_rel + h.spread_jump * excess + noise.normal(0, p.spread_noise_std)
        volume = p.volume_base / max(alpha_rel, 1e-3) - h.volume_jump * excess + noise.normal(0, p.volume_noise_std)
        realvol = p.vol_base * kappa_rel + h.vol_jump * excess + noise.normal(0, p.vol_noise_std)
        return {"spread": max(spread, 1e-5), "volume": max(volume, p.volume_floor),
                "realvol": max(realvol, 1e-5), "n_events": n, "hawkes_excitation": self.proc.E}

    def observe_episode(self, episode: Episode, seed=None) -> pd.DataFrame:
        self.reset(seed)
        N = len(episode.v)
        reg = getattr(episode, "regime", None)
        reg = np.zeros(N + 1, dtype=int) if reg is None else np.asarray(reg)
        rows = [self.observe_step(episode.kappa[t], episode.alpha[t], regime=int(reg[t])) for t in range(N)]
        df = pd.DataFrame(rows)
        df.insert(0, "t", episode.t[:N])
        df["regime"] = reg[:N]
        df["true_kappa"] = episode.kappa[:N]
        df["true_alpha"] = episode.alpha[:N]
        return df


def make_observation_model(kind: str = "gaussian", *, seed: int | None = None, tau: float = 0.02,
                           obs_params: ObsParams | None = None,
                           hawkes_params: HawkesParams | None = None):
    """Selector for the realistic-vs-simplified ablation: kind in {'gaussian','hawkes'}."""
    if kind == "gaussian":
        return GaussianObservation(obs_params, seed=seed)
    if kind == "hawkes":
        return HawkesObservationModel(obs_params, hawkes_params, tau=tau, seed=seed)
    raise ValueError(f"unknown observation model kind: {kind!r}")


if __name__ == "__main__":
    # quick look: run from src/:  python hawkes_observation.py
    from regime_simulator import RegimeSwitchingSimulator
    sim = RegimeSwitchingSimulator(seed=11)
    ep = sim.simulate_episode(velocities=sim.twap_schedule(), seed=11)
    for kind in ("gaussian", "hawkes"):
        m = make_observation_model(kind, seed=11, tau=sim.p.tau)
        df = m.observe_episode(ep, seed=11)
        print(f"--- {kind} ---")
        print(df.head(6).to_string(index=False))