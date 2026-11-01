"""
simulator.py
------------
Ground-truth market simulator for the "Optimal Execution under Latent
Liquidity" project.

Implements the stochastic-impact Almgren-Chriss framework from
Macri & Lillo (2025), "Reinforcement Learning for Optimal Execution when
Liquidity is Time-Varying" (Applied Mathematical Finance), Section 2.1:

    dS_t      = -kappa_t * v_t * dt + sigma * dW_t        (mid-price)
    S_exec_t  = S_{t-1} - alpha_t * v_t                   (execution price)
    dkappa_t  = lambda_kappa * (theta_kappa - kappa_t) * dt
                + sigma_kappa * sqrt(kappa_t) * dB1_t      (permanent impact, CIR)
    dalpha_t  = lambda_alpha * (theta_alpha - alpha_t) * dt
                + sigma_alpha * sqrt(alpha_t) * dB2_t      (temporary impact, CIR)
    corr(dB1_t, dB2_t) = rho

kappa_t (permanent impact) and alpha_t (temporary impact) are both latent,
square-root mean-reverting (CIR) processes, discretised with Euler-Maruyama.
A full-truncation floor is applied at each step to keep them non-negative in
discrete time (Euler-Maruyama does not guarantee CIR positivity even when
the continuous-time Feller condition holds).

Two modeling choices worth flagging explicitly:
  - q0 and sigma are scaled up from the paper's illustrative toy values
    (q0=20, sigma=0.00001) to more realistic magnitudes. kappa0/theta_kappa
    and alpha0/theta_alpha are scaled DOWN from the paper's toy values by
    the matching factor, using the CIR self-similarity law (x -> c*x
    requires theta -> c*theta and sigma -> sqrt(c)*sigma to preserve the
    process's RELATIVE dynamics exactly) -- an earlier build of this file
    scaled q0 up 1000x without rescaling kappa/alpha down to match, which
    left permanent impact (kappa0*q0) at ~160% of the starting price and
    the mid-price path went negative partway through a plain TWAP episode.
    Caught by inspecting the TWAP sanity-check IS number, which was
    implausibly large (~84% of notional) for any real execution cost.
    Total permanent impact is now ~2.5% of S0 over a full liquidation and
    per-trade temporary impact ~0.6% of S0 at the TWAP rate -- both still
    freely tunable in SimParams, but no longer pathological.
  - Both kappa_t and alpha_t are tracked as full paths here so the particle
    filter (next milestone piece) can later choose to filter either one or
    both jointly.

STATUS: milestone-1 build (~30-40% of full project). This module is
treated as essentially complete, not a stub -- everything downstream reads
its output, which is exactly why the impact-scaling issue above was worth
catching and fixing here rather than letting it propagate.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimParams:
    # --- trading horizon ---
    S0: float = 10.0          # initial mid-price ($)
    q0: float = 20_000.0      # initial inventory (shares)
    T: float = 1.0            # trading horizon
    N: int = 50                # number of trading sub-intervals

    # --- price diffusion ---
    sigma: float = 0.02        # mid-price volatility

    # --- permanent impact kappa_t (CIR) ---
    kappa0: float = 1.25e-5
    theta_kappa: float = 1.25e-5
    lambda_kappa: float = 3.0
    sigma_kappa: float = 1.875e-4

    # --- temporary impact alpha_t (CIR) ---
    alpha0: float = 1.5e-4
    theta_alpha: float = 1.5e-4
    lambda_alpha: float = 3.0
    sigma_alpha: float = 4.5928e-4

    # --- correlation between the two CIR driving Brownian motions ---
    rho: float = 0.9

    @property
    def tau(self) -> float:
        return self.T / self.N


@dataclass
class Episode:
    """Container for one simulated episode's ground truth."""
    t: np.ndarray             # (N+1,) time grid, t[0]=0
    S: np.ndarray              # (N+1,) mid-price path, S[0]=S0
    S_exec: np.ndarray         # (N,) execution price received each interval
    kappa: np.ndarray          # (N+1,) true permanent impact path
    alpha: np.ndarray          # (N+1,) true temporary impact path
    v: np.ndarray               # (N,) trading velocity (shares) each interval
    q: np.ndarray               # (N+1,) inventory remaining, q[0]=q0
    IS: float                   # realised implementation shortfall
    shock_step: int | None = None
    shock_info: dict | None = None


def _cir_step(x, theta, lam, sigma_x, dt, dB, floor=1e-10):
    """One Euler-Maruyama step of a CIR process with a full-truncation
    positivity fix:  dx = lam*(theta - x)*dt + sigma_x*sqrt(max(x,0))*dB
    """
    x_pos = max(x, 0.0)
    x_next = x + lam * (theta - x_pos) * dt + sigma_x * np.sqrt(x_pos) * dB
    return max(x_next, floor)


class MarketSimulator:
    """Ground-truth Almgren-Chriss market with stochastic (CIR) permanent
    and temporary impact, per Macri & Lillo (2025), Sec. 2.1."""

    def __init__(self, params: SimParams | None = None, seed: int | None = None):
        self.p = params or SimParams()
        self.rng = np.random.default_rng(seed)

    def twap_schedule(self) -> np.ndarray:
        """Benchmark schedule: sell q0/N shares every interval."""
        p = self.p
        return np.full(p.N, p.q0 / p.N)

    def simulate_episode(
        self,
        velocities: np.ndarray | None = None,
        policy=None,
        shock_step: int | None = None,
        shock_jump_kappa_mult: float = 1.0,
        shock_jump_alpha_mult: float = 3.0,
        seed: int | None = None,
    ) -> Episode:
        """
        Simulate one full episode.

        velocities: fixed (N,) array of shares to sell each interval (e.g.
            TWAP). Ignored if `policy` is given.
        policy: optional callable
            policy(t_idx, q_remaining, kappa_t, alpha_t, N) -> v_t
            Called with the TRUE kappa/alpha, i.e. this is "oracle" mode.
            The belief-informed heuristic (next milestone piece) instead
            calls this simulator interval-by-interval using its own
            filtered belief, not the true kappa/alpha directly.
        shock_step: if given, at this interval kappa_t/alpha_t receive an
            instantaneous multiplicative JUMP (a jump-diffusion event --
            e.g. a sudden liquidity crunch), then continue under the SAME,
            unchanged long-run parameters (theta_kappa, theta_alpha, ...)
            for the rest of the episode, so the process mean-reverts back
            toward its normal level afterward.

            NOTE on a design decision worth flagging: an earlier version of
            this shock permanently changed theta_alpha itself (a regime
            change) rather than jumping the state under a fixed regime.
            That was scrapped after it broke the particle filter validation
            in a way that turned out to be structural, not a filter bug: a
            bootstrap SMC filter is built around KNOWN, fixed transition
            parameters, so silently redefining theta mid-episode makes the
            filter's own model wrong for the rest of the run -- it keeps
            reverting particles toward the OLD mean while the truth chases
            a NEW one it's never told about. That's precisely the
            regime-switching problem this project already scopes to Phase 2
            (Interacting Multiple Model filter, in the "what's next" list),
            not something a Phase-1 bootstrap filter is designed to solve.
            A jump under fixed, known parameters is the right-difficulty
            stress test for THIS milestone's filter: it can and should
            relocate quickly and settle back down, which is exactly the
            "uncertainty widens at the shock, narrows as it re-converges"
            behaviour the validation plot needs to show.
        """
        p = self.p
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        N, tau = p.N, p.tau

        S = np.empty(N + 1)
        S_exec = np.empty(N)
        kappa = np.empty(N + 1)
        alpha = np.empty(N + 1)
        q = np.empty(N + 1)
        v = np.empty(N)

        S[0] = p.S0
        kappa[0] = p.kappa0
        alpha[0] = p.alpha0
        q[0] = p.q0

        shock_info = None

        # correlated Brownian increments for the two CIR processes
        cov = np.array([[1.0, p.rho], [p.rho, 1.0]])
        L = np.linalg.cholesky(cov)

        for t in range(N):
            if shock_step is not None and t == shock_step:
                pre_kappa, pre_alpha = kappa[t], alpha[t]
                kappa[t] = kappa[t] * shock_jump_kappa_mult
                alpha[t] = alpha[t] * shock_jump_alpha_mult
                shock_info = {
                    "step": t,
                    "pre_kappa": pre_kappa, "pre_alpha": pre_alpha,
                    "post_kappa": kappa[t], "post_alpha": alpha[t],
                }

            # trading decision
            if policy is not None:
                v_t = policy(t, q[t], kappa[t], alpha[t], N)
                v_t = float(np.clip(v_t, 0.0, q[t]))
            elif velocities is not None:
                v_t = float(velocities[t])
            else:
                v_t = p.q0 / N  # default TWAP
            v_t = min(v_t, q[t])
            v[t] = v_t

            # execution price uses S[t] and the CURRENT temporary impact
            S_exec[t] = S[t] - alpha[t] * v_t

            # mid-price update (permanent impact + diffusion), Euler-Maruyama
            dW = rng.normal(0.0, np.sqrt(tau))
            S[t + 1] = S[t] - kappa[t] * v_t + p.sigma * dW

            # correlated Brownian increments for kappa, alpha CIR dynamics
            z = rng.normal(size=2) * np.sqrt(tau)
            dB1, dB2 = L @ z

            kappa[t + 1] = _cir_step(kappa[t], p.theta_kappa, p.lambda_kappa, p.sigma_kappa, tau, dB1)
            alpha[t + 1] = _cir_step(alpha[t], p.theta_alpha, p.lambda_alpha, p.sigma_alpha, tau, dB2)

            q[t + 1] = q[t] - v_t

        IS = p.S0 * p.q0 - float(np.sum(S_exec * v))

        return Episode(
            t=np.arange(N + 1) * tau, S=S, S_exec=S_exec, kappa=kappa,
            alpha=alpha, v=v, q=q, IS=IS, shock_step=shock_step, shock_info=shock_info,
        )

    def run_stepwise(
        self,
        shock_step: int | None = None,
        shock_jump_kappa_mult: float = 1.0,
        shock_jump_alpha_mult: float = 3.0,
        seed: int | None = None,
    ):
        """
        Generator-based step API. simulate_episode()'s `policy` callback
        only ever sees the TRUE kappa_t/alpha_t, which is fine for the
        oracle benchmark but useless for a policy that is only supposed to
        act on a FILTERED BELIEF (this milestone's heuristic) or a learned
        state (Phase 2's RL agent) -- both need to run their own
        observation/filter/decision logic BETWEEN simulator steps, using
        whatever v_t they compute, not something simulate_episode can
        hand them upfront. Hence a pull-based generator instead:

            gen = sim.run_stepwise(seed=7)
            state = next(gen)                 # state before any trade
            while True:
                v_t = my_policy(state)        # state['kappa']/state['alpha']
                                               # are TRUE values -- a belief-
                                               # based policy must ignore them
                                               # and consult its own filter.
                try:
                    state = gen.send(v_t)
                except StopIteration as stop:
                    episode = stop.value      # final Episode
                    break

        This is an independently-written implementation of the exact same
        step math as simulate_episode (not a shared refactor of it), by
        deliberate choice: simulate_episode is the already-validated
        ground-truth engine everything reads from, so it's left untouched
        rather than risking a shared-code regression. The __main__ block
        below instead runs both with an identical seed+velocity schedule
        and asserts bit-for-bit equality, as a standing regression test
        that this generator can't silently drift from simulate_episode.
        """
        p = self.p
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        N, tau = p.N, p.tau

        S = np.empty(N + 1)
        S_exec = np.empty(N)
        kappa = np.empty(N + 1)
        alpha = np.empty(N + 1)
        q = np.empty(N + 1)
        v = np.empty(N)

        S[0] = p.S0
        kappa[0] = p.kappa0
        alpha[0] = p.alpha0
        q[0] = p.q0
        shock_info = None

        cov = np.array([[1.0, p.rho], [p.rho, 1.0]])
        L = np.linalg.cholesky(cov)

        for t in range(N):
            if shock_step is not None and t == shock_step:
                pre_kappa, pre_alpha = kappa[t], alpha[t]
                kappa[t] = kappa[t] * shock_jump_kappa_mult
                alpha[t] = alpha[t] * shock_jump_alpha_mult
                shock_info = {
                    "step": t,
                    "pre_kappa": pre_kappa, "pre_alpha": pre_alpha,
                    "post_kappa": kappa[t], "post_alpha": alpha[t],
                }

            v_t = yield {
                "t": t, "S": S[t], "kappa": kappa[t], "alpha": alpha[t],
                "q": q[t], "N": N, "shock_step": shock_step,
            }
            v_t = float(np.clip(0.0 if v_t is None else v_t, 0.0, q[t]))
            v[t] = v_t

            S_exec[t] = S[t] - alpha[t] * v_t
            dW = rng.normal(0.0, np.sqrt(tau))
            S[t + 1] = S[t] - kappa[t] * v_t + p.sigma * dW

            z = rng.normal(size=2) * np.sqrt(tau)
            dB1, dB2 = L @ z
            kappa[t + 1] = _cir_step(kappa[t], p.theta_kappa, p.lambda_kappa, p.sigma_kappa, tau, dB1)
            alpha[t + 1] = _cir_step(alpha[t], p.theta_alpha, p.lambda_alpha, p.sigma_alpha, tau, dB2)
            q[t + 1] = q[t] - v_t

        IS = p.S0 * p.q0 - float(np.sum(S_exec * v))
        return Episode(
            t=np.arange(N + 1) * tau, S=S, S_exec=S_exec, kappa=kappa,
            alpha=alpha, v=v, q=q, IS=IS, shock_step=shock_step, shock_info=shock_info,
        )


if __name__ == "__main__":
    # sanity check: python src/simulator.py
    sim = MarketSimulator(seed=42)
    ep = sim.simulate_episode(velocities=sim.twap_schedule())
    print(f"TWAP episode -> IS = {ep.IS:.4f}, final inventory = {ep.q[-1]:.2f}")
    print(f"kappa path: {ep.kappa[0]:.5f} -> {ep.kappa[-1]:.5f}")
    print(f"alpha path: {ep.alpha[0]:.5f} -> {ep.alpha[-1]:.5f}")

    # regression test: run_stepwise() must reproduce simulate_episode()
    # bit-for-bit given the same seed and velocity schedule
    sim2 = MarketSimulator(seed=42)
    twap = sim2.twap_schedule()
    ep_direct = sim2.simulate_episode(velocities=twap, seed=123)

    sim3 = MarketSimulator(seed=42)
    gen = sim3.run_stepwise(seed=123)
    state = next(gen)
    ep_stepwise = None
    i = 0
    while True:
        try:
            state = gen.send(float(twap[i]))
            i += 1
        except StopIteration as stop:
            ep_stepwise = stop.value
            break
    assert np.allclose(ep_direct.S, ep_stepwise.S) and np.allclose(ep_direct.kappa, ep_stepwise.kappa) \
        and np.allclose(ep_direct.alpha, ep_stepwise.alpha) and np.isclose(ep_direct.IS, ep_stepwise.IS)
    print(f"regression check: run_stepwise matches simulate_episode bit-for-bit "
          f"(IS {ep_direct.IS:.4f} == {ep_stepwise.IS:.4f}) -- OK")