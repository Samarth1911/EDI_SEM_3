"""
regime_simulator.py  (STAGE 1 -- Person 1)
------------------------------------------
Discrete Markov regime-switching layer on top of the validated CIR simulator
in simulator.py (which is NOT modified -- this module subclasses it).

Framework: Hamilton (1989), a latent first-order Markov chain s_t in {0..K-1}
(here K=2: 0=calm, 1=stressed) with row-stochastic transition matrix
    P[i, j] = Pr(s_{t+1} = j | s_t = i)          (per simulator STEP, not per unit time)
The continuous CIR impact dynamics run CONDITIONAL on the active regime.

TIMING CONVENTION (important for the IMM filter, Stage 2)
    regime[t], t = 0..N, is aligned with kappa[t], alpha[t], S[t] (all N+1 long).
    regime[t] is the regime IN FORCE AT TIME t, and it is the regime whose
    parameters generated (kappa[t], alpha[t], S[t]) from time t-1:
        s_{t+1}                  ~ P[s_t, :]
        (kappa,alpha)[t+1]       = CIR_step( (kappa,alpha)[t] ; params(s_{t+1}) )
        S[t+1]                   = S[t] - kappa[t]*v_t + sigma_price(s_{t+1})*dW
    i.e. the standard Markov-jump-system form x_{t+1} = f(x_t, s_{t+1}) used by
    Blom & Bar-Shalom (1988). regime[0] only fixes the initial state.
    The proxies observed at time t depend on x_t (generated under regime[t]).

PER-REGIME PARAMETERS are expressed as MULTIPLIERS on the base SimParams, so a
regime with all multipliers = 1 IS the original simulator.
    theta_*_mult   : long-run mean multiplier
    lambda_mult    : mean-reversion speed multiplier
    volvol_mult    : vol-of-vol multiplier RELATIVE to CIR self-similarity.
                     sigma_eff = sigma * sqrt(theta_mult) * volvol_mult, so 1.0
                     preserves the process's relative dynamics exactly (same
                     law used in simulator.py's docstring: x->cx needs
                     theta->c*theta, sigma->sqrt(c)*sigma).
    price_sigma_mult: multiplier on the mid-price diffusion sigma.

RNG DISCIPLINE: the regime chain is drawn from its OWN stream, seeded by
[seed, REGIME_RNG_TAG]. The main stream is consumed in exactly the same order
as MarketSimulator (dW, then a 2-vector per step), so with a single regime of
unit multipliers this class reproduces MarketSimulator bit-for-bit (tested in
validation/validate_stage1.py).

STATUS: Stage-1 build. Provisional choices are flagged in HANDOFF (stage1_handoff.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from simulator import MarketSimulator, SimParams, Episode, _cir_step

REGIME_RNG_TAG = 0xA11CE


@dataclass
class RegimeSpec:
    name: str = "calm"
    theta_kappa_mult: float = 1.0
    theta_alpha_mult: float = 1.0
    lambda_mult: float = 1.0
    volvol_mult: float = 1.0
    price_sigma_mult: float = 1.0


def _default_specs():
    return (
        RegimeSpec("calm"),
        RegimeSpec("stressed", theta_kappa_mult=1.6, theta_alpha_mult=2.5,
                   lambda_mult=2.0, volvol_mult=1.25, price_sigma_mult=1.6),
    )


@dataclass
class RegimeParams:
    specs: tuple = field(default_factory=_default_specs)
    P: np.ndarray = field(default_factory=lambda: np.array([[0.96, 0.04],
                                                             [0.08, 0.92]]))
    init_probs: np.ndarray | None = None   # None => start in regime 0 (calm) w.p. 1

    def __post_init__(self):
        self.P = np.asarray(self.P, dtype=float)
        K = len(self.specs)
        if self.P.shape != (K, K):
            raise ValueError(f"P must be {K}x{K}")
        if np.any(self.P < 0) or not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("P must be row-stochastic")
        if self.init_probs is None:
            self.init_probs = np.eye(K)[0]
        self.init_probs = np.asarray(self.init_probs, dtype=float)
        if not np.isclose(self.init_probs.sum(), 1.0):
            raise ValueError("init_probs must sum to 1")

    # ---- presets -------------------------------------------------------
    @classmethod
    def default(cls) -> "RegimeParams":
        """Clearly separated calm/stressed (stressed alpha long-run mean = 2.5x)."""
        return cls()

    @classmethod
    def subtle(cls) -> "RegimeParams":
        """Hard-to-detect regimes (alpha +15%, kappa +8%) -- for stress-testing
        the regime detector; see handoff for the separation numbers."""
        return cls(specs=(RegimeSpec("calm"),
                          RegimeSpec("stressed", theta_kappa_mult=1.08, theta_alpha_mult=1.15,
                                     lambda_mult=1.5, volvol_mult=1.1, price_sigma_mult=1.2)))

    @classmethod
    def single(cls) -> "RegimeParams":
        """Degenerate one-regime config == the original simulator (regression tests)."""
        return cls(specs=(RegimeSpec("calm"),), P=np.array([[1.0]]))

    # ---- derived quantities -------------------------------------------
    @property
    def n_regimes(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> list:
        return [s.name for s in self.specs]

    def stationary_distribution(self) -> np.ndarray:
        w, v = np.linalg.eig(self.P.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        return pi / pi.sum()

    def expected_sojourn_steps(self) -> np.ndarray:
        """Mean geometric sojourn 1/(1-P_ii) in steps."""
        d = np.diag(self.P)
        return np.where(d < 1.0, 1.0 / np.maximum(1.0 - d, 1e-300), np.inf)

    def effective_cir(self, sp: SimParams, r: int) -> dict:
        s = self.specs[r]
        sk = sp.sigma_kappa * math.sqrt(s.theta_kappa_mult) * s.volvol_mult
        sa = sp.sigma_alpha * math.sqrt(s.theta_alpha_mult) * s.volvol_mult
        return dict(
            theta_kappa=sp.theta_kappa * s.theta_kappa_mult,
            lambda_kappa=sp.lambda_kappa * s.lambda_mult,
            sigma_kappa=sk,
            theta_alpha=sp.theta_alpha * s.theta_alpha_mult,
            lambda_alpha=sp.lambda_alpha * s.lambda_mult,
            sigma_alpha=sa,
            sigma_price=sp.sigma * s.price_sigma_mult,
        )

    def feller_ratios(self, sp: SimParams) -> list:
        """2*lambda*theta / sigma^2 per regime and process (>=1 => Feller holds)."""
        out = []
        for r in range(self.n_regimes):
            e = self.effective_cir(sp, r)
            out.append(dict(
                regime=self.specs[r].name,
                kappa=2 * e["lambda_kappa"] * e["theta_kappa"] / e["sigma_kappa"] ** 2,
                alpha=2 * e["lambda_alpha"] * e["theta_alpha"] / e["sigma_alpha"] ** 2))
        return out


@dataclass
class RegimeEpisode(Episode):
    """Episode + the true regime path (N+1,), aligned with kappa/alpha/S."""
    regime: np.ndarray | None = None


def force_regime_window(path: np.ndarray, start: int, duration: int, regime: int = 1) -> np.ndarray:
    """Return a copy of a regime path with [start, start+duration) forced to `regime`
    (used for controlled validation and for the dashboard's shock-injection control)."""
    out = np.array(path, dtype=int, copy=True)
    out[start:start + duration] = regime
    return out


class RegimeSwitchingSimulator(MarketSimulator):
    """MarketSimulator + Hamilton regime layer. Same API shape as the parent."""

    def __init__(self, params: SimParams | None = None,
                 regime_params: RegimeParams | None = None, seed: int | None = None):
        super().__init__(params, seed)
        self.rp = regime_params or RegimeParams.default()
        self._eff = [self.rp.effective_cir(self.p, r) for r in range(self.rp.n_regimes)]
        self._regime_rng_default = (np.random.default_rng([seed, REGIME_RNG_TAG])
                                    if seed is not None else np.random.default_rng())

    # ------------------------------------------------------------------
    def sample_regime_path(self, N: int, rng: np.random.Generator) -> np.ndarray:
        """Draw s_0..s_N. Uniforms are drawn upfront so stream consumption
        does not depend on the visited states."""
        rp = self.rp
        u = rng.random(N + 1)
        path = np.empty(N + 1, dtype=int)
        path[0] = int(np.searchsorted(np.cumsum(rp.init_probs), u[0]))
        cP = np.cumsum(rp.P, axis=1)
        cP[:, -1] = 1.0
        for t in range(N):
            path[t + 1] = int(np.searchsorted(cP[path[t]], u[t + 1]))
        return np.minimum(path, rp.n_regimes - 1)

    # ------------------------------------------------------------------
    def run_stepwise(self, shock_step: int | None = None,
                     shock_jump_kappa_mult: float = 1.0,
                     shock_jump_alpha_mult: float = 3.0,
                     seed: int | None = None,
                     forced_regime_path: np.ndarray | None = None):
        """Pull-based generator, same protocol as MarketSimulator.run_stepwise
        (state = next(gen); state = gen.send(v_t); StopIteration.value = episode),
        with two additions: state['regime'] (TRUE regime -- belief-based policies
        must not read it, same rule as state['kappa'/'alpha']) and the optional
        `forced_regime_path` (N+1,) overriding the sampled chain.
        The single `shock_step` JUMP mechanism of the parent is retained
        (a jump within a regime is distinct from a regime switch)."""
        p, rp = self.p, self.rp
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        rrng = (np.random.default_rng([seed, REGIME_RNG_TAG]) if seed is not None
                else self._regime_rng_default)
        N, tau = p.N, p.tau

        if forced_regime_path is not None:
            regime = np.asarray(forced_regime_path, dtype=int)
            if regime.shape != (N + 1,) or regime.min() < 0 or regime.max() >= rp.n_regimes:
                raise ValueError("forced_regime_path must be int array (N+1,) with valid regime ids")
        else:
            regime = self.sample_regime_path(N, rrng)

        S = np.empty(N + 1); S_exec = np.empty(N)
        kappa = np.empty(N + 1); alpha = np.empty(N + 1)
        q = np.empty(N + 1); v = np.empty(N)

        r0 = regime[0]
        S[0] = p.S0
        kappa[0] = p.kappa0 * rp.specs[r0].theta_kappa_mult   # start at the regime's own level
        alpha[0] = p.alpha0 * rp.specs[r0].theta_alpha_mult
        q[0] = p.q0
        shock_info = None

        L = np.linalg.cholesky(np.array([[1.0, p.rho], [p.rho, 1.0]]))

        for t in range(N):
            if shock_step is not None and t == shock_step:
                pre_k, pre_a = kappa[t], alpha[t]
                kappa[t] *= shock_jump_kappa_mult
                alpha[t] *= shock_jump_alpha_mult
                shock_info = {"step": t, "pre_kappa": pre_k, "pre_alpha": pre_a,
                              "post_kappa": kappa[t], "post_alpha": alpha[t]}

            v_t = yield {"t": t, "S": S[t], "kappa": kappa[t], "alpha": alpha[t],
                         "q": q[t], "N": N, "shock_step": shock_step,
                         "regime": int(regime[t])}
            v_t = float(np.clip(0.0 if v_t is None else v_t, 0.0, q[t]))
            v[t] = v_t

            e = self._eff[regime[t + 1]]           # regime that generates x_{t+1}
            S_exec[t] = S[t] - alpha[t] * v_t
            dW = rng.normal(0.0, np.sqrt(tau))
            S[t + 1] = S[t] - kappa[t] * v_t + e["sigma_price"] * dW

            z = rng.normal(size=2) * np.sqrt(tau)
            dB1, dB2 = L @ z
            kappa[t + 1] = _cir_step(kappa[t], e["theta_kappa"], e["lambda_kappa"], e["sigma_kappa"], tau, dB1)
            alpha[t + 1] = _cir_step(alpha[t], e["theta_alpha"], e["lambda_alpha"], e["sigma_alpha"], tau, dB2)
            q[t + 1] = q[t] - v_t

        IS = p.S0 * p.q0 - float(np.sum(S_exec * v))
        return RegimeEpisode(t=np.arange(N + 1) * tau, S=S, S_exec=S_exec, kappa=kappa,
                             alpha=alpha, v=v, q=q, IS=IS, shock_step=shock_step,
                             shock_info=shock_info, regime=regime)

    # ------------------------------------------------------------------
    def simulate_episode(self, velocities=None, policy=None, shock_step=None,
                         shock_jump_kappa_mult: float = 1.0,
                         shock_jump_alpha_mult: float = 3.0,
                         seed: int | None = None,
                         forced_regime_path: np.ndarray | None = None) -> RegimeEpisode:
        """Same contract as MarketSimulator.simulate_episode (fixed schedule or
        oracle `policy(t, q, kappa, alpha, N)`), implemented by driving run_stepwise."""
        N, q0 = self.p.N, self.p.q0
        gen = self.run_stepwise(shock_step, shock_jump_kappa_mult, shock_jump_alpha_mult,
                                seed, forced_regime_path)
        state = next(gen)
        while True:
            t = state["t"]
            if policy is not None:
                v_t = policy(t, state["q"], state["kappa"], state["alpha"], N)
            elif velocities is not None:
                v_t = float(velocities[t])
            else:
                v_t = q0 / N
            try:
                state = gen.send(v_t)
            except StopIteration as stop:
                return stop.value


if __name__ == "__main__":
    sim = RegimeSwitchingSimulator(seed=11)
    ep = sim.simulate_episode(velocities=sim.twap_schedule(), seed=11)
    print("regime path:", "".join(map(str, ep.regime)))
    print(f"IS={ep.IS:.1f}  alpha range [{ep.alpha.min():.2e}, {ep.alpha.max():.2e}]")
    print("pi =", sim.rp.stationary_distribution().round(3), " sojourn(steps) =", sim.rp.expected_sojourn_steps())
    print("Feller:", sim.rp.feller_ratios(sim.p))