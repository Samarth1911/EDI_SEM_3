"""
observation_model.py
---------------------
Generates noisy market proxies from the TRUE hidden impact state
(kappa_t, alpha_t) produced by simulator.py.

Proxies are defined RELATIVE to a reference scale (kappa_ref, alpha_ref --
by default the simulator's baseline kappa0/alpha0), rather than as
absolute-scale linear functions. This matters because kappa_t/alpha_t live
at ~1e-3 scale with natural (stationary CIR) fluctuations of only ~2e-5 --
absolute-scale sensitivities are very easy to mis-calibrate against noise
levels picked independently (this file's first version did exactly that;
see particle_filter.py's module docstring / the validation writeup for
what that looked like and how it was caught).

  alpha_rel = alpha_t / alpha_ref     (~1.0 in normal conditions)
  kappa_rel = kappa_t / kappa_ref     (~1.0 in normal conditions)

  spread_t  = spread_base * alpha_rel + noise      (wider when alpha up)
  volume_t  = volume_base / alpha_rel + noise      (thinner when alpha up)
  realvol_t = vol_base * kappa_rel + noise          (more churn when kappa up)

Noise levels are set so a normal one-std fluctuation in alpha_rel/kappa_rel
(~1.5-2%) is only marginally detectable per-step (SNR ~1, so the filter
genuinely needs to accumulate evidence over several steps to track it),
while a liquidity shock (which multiplies the long-run mean by e.g. 3x)
produces a signal tens of std's above the noise floor -- unambiguous.

STATUS: milestone-1 build. Simple/linear-in-relative-deviation forms by
design; richer (e.g. Hawkes-driven order flow) proxies are next-phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from simulator import Episode

@dataclass
class ObsParams:
    kappa_ref: float = 1.25e-5    # reference kappa scale (matches SimParams.kappa0)
    alpha_ref: float = 1.5e-4     # reference alpha scale (matches SimParams.alpha0)

    spread_base: float = 0.05          # $ spread when alpha_t == alpha_ref
    spread_noise_std: float = 0.0018   # loosened from 0.0008 -- see particle_filter.py notes

    volume_base: float = 5000.0        # market volume when alpha_t == alpha_ref
    volume_floor: float = 50.0
    volume_noise_std: float = 120.0    # loosened from 60

    vol_base: float = 0.02             # realised-vol proxy when kappa_t == kappa_ref
    vol_noise_std: float = 0.0008      # loosened from 0.0004


class ObservationModel:
    """Maps a true (kappa_t, alpha_t) state to noisy observable proxies."""

    def __init__(self, params: ObsParams | None = None, seed: int | None = None):
        self.p = params or ObsParams()
        self.rng = np.random.default_rng(seed)

    def observe_step(self, kappa_t: float, alpha_t: float, rng=None) -> dict:
        p = self.p
        r = rng or self.rng

        alpha_rel = alpha_t / p.alpha_ref
        kappa_rel = kappa_t / p.kappa_ref

        spread = p.spread_base * alpha_rel + r.normal(0, p.spread_noise_std)
        spread = max(spread, 1e-5)

        volume = p.volume_base / max(alpha_rel, 1e-3) + r.normal(0, p.volume_noise_std)
        volume = max(volume, p.volume_floor)

        realvol = p.vol_base * kappa_rel + r.normal(0, p.vol_noise_std)
        realvol = max(realvol, 1e-5)

        return {"spread": spread, "volume": volume, "realvol": realvol}

    def observe_episode(self, episode: Episode, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        N = len(episode.v)
        rows = [self.observe_step(episode.kappa[t], episode.alpha[t], rng=rng) for t in range(N)]
        df = pd.DataFrame(rows)
        df.insert(0, "t", episode.t[:N])
        df["true_kappa"] = episode.kappa[:N]
        df["true_alpha"] = episode.alpha[:N]
        return df


if __name__ == "__main__":
    # sanity check: python src/observation_model.py
    from simulator import MarketSimulator

    sim = MarketSimulator(seed=7)
    ep = sim.simulate_episode(
        velocities=sim.twap_schedule(),
        shock_step=25,
        shock_jump_alpha_mult=3.0,
    )
    obs_model = ObservationModel(seed=7)
    df = obs_model.observe_episode(ep)

    print(df.head(8).to_string(index=False))
    print("...")
    print(df.iloc[23:28].to_string(index=False))  # around the shock at t=25
    print(f"\nspread: pre-shock mean ~{df.spread.iloc[:25].mean():.4f}, "
          f"post-shock mean ~{df.spread.iloc[25:].mean():.4f} (should be clearly higher)")
    print(f"volume: pre-shock mean ~{df.volume.iloc[:25].mean():.1f}, "
          f"post-shock mean ~{df.volume.iloc[25:].mean():.1f} (should be clearly lower)")