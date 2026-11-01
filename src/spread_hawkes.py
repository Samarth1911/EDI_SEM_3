"""
spread_hawkes.py  (STAGE 1 -- Person 1, "v2" Hawkes observation model)
----------------------------------------------------------------------
Place in EDI/src/.  Does NOT modify hawkes_observation.py or observation_model.py.

WHAT THIS IMPLEMENTS
    The spread-as-point-process model of Fosset, Bouchaud & Benzaquen,
    "Endogenous Liquidity Crises" (arXiv:1912.00359), Section 4 (and the
    stabilising mechanism of Sec. 4.5). The spread is an INTEGER STATE
    S_t >= 1 (in ticks), driven by two event types:

        opening events  (S -> S+1), intensity
            lambda+(s) = mu_r + sum_{opening events t_i < s} a_r * beta * exp(-beta (s - t_i))
            i.e. a linear exponential-kernel Hawkes process: spread-opening
            events are likely to trigger more spread-opening events (Eq. 8a).
        closing events  (S -> S-1), intensity
            stabilised (DEFAULT, paper Eq. 13):  lambda-(s) = lam_close * (S - 1)
            indicator  (paper Eq. 8b):           lambda-(s) = lam_close * 1{S >= 2}

    a_r is the branching ratio (paper's alpha). Stability of the Hawkes part
    needs a_r < 1 (enforced). With the STABILISED closing rate the spread is
    stationary for every a_r < 1. With the indicator closing rate the paper shows
    three phases (stationary / linear growth / explosive) split at
    alpha_c = 1 - mu/lam_close; `SpreadHawkesParams.phase(r)` reports which one a
    regime would be in -- with the defaults below BOTH regimes would be in the
    linear-growth phase under the indicator rule, which is why stabilised is the
    default.

    EXACT MEAN (stabilised, constant regime): the opening intensity does not depend
    on S, so E[dS/dt] = E[lambda+] - lam_close*(E[S]-1) gives, in stationarity,
        E[S] = 1 + (mu/(1-a)) / lam_close.
    validate_spread_hawkes.py tests this by simulation.

    Simulation: Ogata thinning on the superposed process. Between events the
    excitation decays and S is constant, so the total intensity mu + E + lambda-(S)
    is non-increasing and its value at the last candidate time is a valid bound.

PROXIES (same three keys the particle filter reads: spread, volume, realvol)
    dS = S_t - S_ref,  S_ref = calm stationary mean ticks (moment matching)
    spread_t  = spread_base*alpha_rel + tick_value*dS       + noise
    volume_t  = volume_base/alpha_rel - volume_per_tick*dS  + noise
    realvol_t = vol_base*kappa_rel    + vol_per_tick*dS     + noise
    In calm/stationary conditions the mean equals the Gaussian model's; the
    difference is clustered, autocorrelated, right-skewed spread excursions that
    are genuine STATE (they persist and decay), unlike per-step count noise.
    Optional alpha_coupling g: opening baseline is scaled by alpha_rel**g so the
    LATENT liquidity state also drives event intensity (g=0 -> off, default).
    With g>0 the calm mean is no longer exactly moment-matched (alpha_rel varies).

TIMING (differs from hawkes_observation.py -- CAUSAL labelling)
    Row t = the process over ((t-1)*tau, t*tau], baseline set by regime[t] (the
    regime that generated x_t, matching regime_simulator.py) and observed at the
    END of the window, so nothing after time t is used. (The v1 Hawkes model
    labels its window [t, t+1) and so has a within-step look-ahead.) Row 0 uses
    regime[0] after a calm warm-up. This resolves handoff open question 5 for
    this model only -- Stage 0 should still ratify one convention for all models.

HONEST STATUS
    - Parameters (mu0, branching, beta, lam_close, tick_value, per-tick loadings)
      are ILLUSTRATIVE, NOT calibrated to data.
    - Still univariate in the opening process. Bacry-Delattre-Hoffmann-Muzy (2013)
      use mutually exciting up/down tick processes -- not reproduced here (and I
      have not re-verified that paper this session).
    - The paper's price-feedback (quadratic / Q-Hawkes, Sec. 2, 5, App. D) is NOT
      implemented; opening intensity does not depend on the price path or on our
      own trading. Regime enters only through mu and the branching ratio.
    - The paper is a stylised order-book/spread model; mapping S onto our
      dollar-spread proxy via tick_value is a modelling choice made here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator import Episode
from observation_model import ObsParams

SPREAD_HAWKES_RNG_TAG = 0x5348


# ---------------------------------------------------------------------------
@dataclass
class SpreadHawkesParams:
    mu0: float = 50.0                 # opening baseline (events / unit time), calm
    mu_mult: tuple = (1.0, 1.5)       # per-regime multiplier on mu0
    branching: tuple = (0.4, 0.8)     # per-regime branching ratio a_r (must be < 1)
    beta: float = 10.0                # kernel decay rate; memory 1/beta = 0.1 = 5 steps at tau=0.02
    lam_close: float = 60.0           # closing rate per unit time (per (S-1) if stabilised)
    stabilised: bool = True           # paper Eq. 13 (True) vs Eq. 8b indicator (False)
    s_ref: float | None = None        # reference ticks; None => analytic (stabilised only)
    tick_value: float = 0.0015        # $ of spread per tick of excess spread
    volume_per_tick: float = 100.0    # shares of volume removed per excess tick
    vol_per_tick: float = 0.0004      # realised-vol proxy added per excess tick
    alpha_coupling: float = 0.0       # opening baseline *= alpha_rel**alpha_coupling
    warmup_steps: int = 100           # calm burn-in so row 0 is not a cold start
    allow_unstable: bool = False      # only for demonstrating instability

    def __post_init__(self):
        self.mu_mult = tuple(float(x) for x in self.mu_mult)
        self.branching = tuple(float(x) for x in self.branching)
        if len(self.mu_mult) != len(self.branching):
            raise ValueError("mu_mult and branching must have one entry per regime")
        if self.mu0 <= 0 or self.beta <= 0 or self.lam_close <= 0 or any(m <= 0 for m in self.mu_mult):
            raise ValueError("mu0, beta, lam_close, mu_mult must be > 0")
        if self.tick_value <= 0:
            raise ValueError("tick_value must be > 0")
        if any(a < 0 for a in self.branching):
            raise ValueError("branching ratios must be >= 0")
        if not self.allow_unstable and any(a >= 1.0 for a in self.branching):
            raise ValueError(f"unstable Hawkes: branching ratio must be < 1, got {self.branching}")
        if (not self.stabilised) and self.s_ref is None:
            raise ValueError("indicator (non-stabilised) variant has no closed-form mean: set s_ref explicitly")

    @property
    def n_regimes(self) -> int:
        return len(self.branching)

    def baseline(self, r: int, alpha_rel: float = 1.0) -> float:
        return self.mu0 * self.mu_mult[r] * float(alpha_rel) ** self.alpha_coupling

    def mean_open_intensity(self, r: int, alpha_rel: float = 1.0) -> float:
        return self.baseline(r, alpha_rel) / (1.0 - self.branching[r])

    def mean_spread_ticks(self, r: int, alpha_rel: float = 1.0) -> float:
        """Exact stationary mean of S under a CONSTANT regime (stabilised rule only)."""
        if not self.stabilised:
            return float("nan")
        return 1.0 + self.mean_open_intensity(r, alpha_rel) / self.lam_close

    def reference_ticks(self) -> float:
        return float(self.s_ref) if self.s_ref is not None else self.mean_spread_ticks(0, 1.0)

    def alpha_c(self, r: int) -> float:
        """Paper's critical feedback for the INDICATOR closing rule: 1 - mu/lam_close."""
        return 1.0 - self.baseline(r) / self.lam_close

    def phase(self, r: int) -> str:
        """Phase of regime r under the INDICATOR closing rule (paper Sec. 4.1)."""
        a = self.branching[r]
        if a >= 1.0:
            return "explosive"
        return "stationary" if a < self.alpha_c(r) else "linear_growth"


# ---------------------------------------------------------------------------
class SpreadHawkesProcess:
    """State (S, E): integer spread and Hawkes excitation E(s)=sum a*beta*exp(-beta(s-t_i))
    over opening events. One continuous process across consecutive advance() calls."""

    def __init__(self, beta: float, lam_close: float, stabilised: bool, rng: np.random.Generator):
        self.beta = float(beta)
        self.lam_close = float(lam_close)
        self.stabilised = bool(stabilised)
        self.rng = rng
        self.E = 0.0
        self.S = 1

    def lam_minus(self, S: int) -> float:
        if self.stabilised:
            return self.lam_close * (S - 1)
        return self.lam_close if S >= 2 else 0.0

    def advance(self, mu: float, a: float, dt: float, record: bool = False):
        """Simulate the window (0, dt] with opening baseline mu and branching a.
        Returns (n_open, n_close, log). log is None unless record=True, in which
        case it is a time-ordered list of (time_in_window, +1 open | -1 close)."""
        beta, rng = self.beta, self.rng
        E, S, s = self.E, self.S, 0.0
        n_open = n_close = 0
        log = [] if record else None
        while True:
            lc = self.lam_minus(S)
            M = mu + E + lc                       # bound: nothing increases until the next event
            w = rng.exponential(1.0 / M)
            if s + w >= dt:
                E *= math.exp(-beta * (dt - s))
                break
            s += w
            E *= math.exp(-beta * w)
            u = rng.random() * M
            lam_open = mu + E
            if u < lam_open:                      # accepted opening event
                S += 1
                E += a * beta
                n_open += 1
                if record:
                    log.append((s, 1))
            elif u < lam_open + lc:               # accepted closing event
                S -= 1
                n_close += 1
                if record:
                    log.append((s, -1))
            # else: rejected candidate (thinning)
        self.E, self.S = E, S
        return n_open, n_close, log


# ---------------------------------------------------------------------------
class SpreadHawkesObservationModel:
    """Stateful: call observe_step in time order t=0,1,2,... after reset().
    observe_episode() resets for you. Same observe_step / observe_episode API as
    the other observation models."""
    kind = "spread_hawkes"

    def __init__(self, obs_params: ObsParams | None = None,
                 hawkes_params: SpreadHawkesParams | None = None,
                 tau: float = 0.02, seed: int | None = None):
        self.p = obs_params or ObsParams()
        self.h = hawkes_params or SpreadHawkesParams()
        self.tau = float(tau)
        self.seed = seed
        self.s_ref = self.h.reference_ticks()
        self.reset(seed)

    def reset(self, seed=None):
        s = self.seed if seed is None else seed
        self.rng = np.random.default_rng(None if s is None else [int(s), SPREAD_HAWKES_RNG_TAG])
        self.proc = SpreadHawkesProcess(self.h.beta, self.h.lam_close, self.h.stabilised, self.rng)
        for _ in range(self.h.warmup_steps):      # calm burn-in
            self.proc.advance(self.h.baseline(0), self.h.branching[0], self.tau)

    def observe_step(self, kappa_t: float, alpha_t: float, regime: int = 0, rng=None) -> dict:
        p, h = self.p, self.h
        r = int(regime)
        if r >= h.n_regimes:
            raise IndexError(f"regime {r} but SpreadHawkesParams defines {h.n_regimes} regimes")
        noise = rng or self.rng

        alpha_rel = alpha_t / p.alpha_ref
        kappa_rel = kappa_t / p.kappa_ref
        mu = h.baseline(r, max(alpha_rel, 1e-3))
        n_open, n_close, _ = self.proc.advance(mu, h.branching[r], self.tau)
        S = self.proc.S
        dS = S - self.s_ref

        spread = p.spread_base * alpha_rel + h.tick_value * dS + noise.normal(0, p.spread_noise_std)
        volume = p.volume_base / max(alpha_rel, 1e-3) - h.volume_per_tick * dS + noise.normal(0, p.volume_noise_std)
        realvol = p.vol_base * kappa_rel + h.vol_per_tick * dS + noise.normal(0, p.vol_noise_std)
        return {"spread": max(spread, 1e-5), "volume": max(volume, p.volume_floor),
                "realvol": max(realvol, 1e-5),
                "spread_ticks": S, "n_open": n_open, "n_close": n_close,
                "n_events": n_open,                       # = self-exciting (opening) events, for v1 column compatibility
                "hawkes_excitation": self.proc.E}

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


def make_observation_model_v2(kind: str = "gaussian", *, seed: int | None = None, tau: float = 0.02,
                              obs_params: ObsParams | None = None, hawkes_params=None):
    """Selector for the ablation: kind in {'gaussian', 'hawkes', 'spread_hawkes'}.
    'gaussian' and 'hawkes' delegate to hawkes_observation.make_observation_model.
    hawkes_params must be HawkesParams for 'hawkes' and SpreadHawkesParams for 'spread_hawkes'."""
    if kind == "spread_hawkes":
        return SpreadHawkesObservationModel(obs_params, hawkes_params, tau=tau, seed=seed)
    from hawkes_observation import make_observation_model
    return make_observation_model(kind, seed=seed, tau=tau, obs_params=obs_params, hawkes_params=hawkes_params)


if __name__ == "__main__":
    # quick look, run from src/:  python spread_hawkes.py
    from regime_simulator import RegimeSwitchingSimulator
    sim = RegimeSwitchingSimulator(seed=11)
    ep = sim.simulate_episode(velocities=sim.twap_schedule(), seed=11)
    h = SpreadHawkesParams()
    print("theory mean ticks: calm %.3f  stressed %.3f" % (h.mean_spread_ticks(0), h.mean_spread_ticks(1)))
    print("indicator-rule phases:", [h.phase(r) for r in range(h.n_regimes)])
    df = SpreadHawkesObservationModel(hawkes_params=h, tau=sim.p.tau, seed=11).observe_episode(ep, seed=11)
    print(df.head(8).to_string(index=False))