"""
belief_mdp_env.py -- Stage 3A (Person 3): belief-MDP environment.
PROVISIONAL / UNTESTED until validation/selftest_env.py has been run by the team.

Wraps RegimeSwitchingSimulator.run_stepwise + an observation model + IMMParticleFilter.
Action  : index into cfg.action_multipliers; v_t = mult * q_t/(N-t)  (multiplier 1.0 == TWAP).
Reward  : -1e4 * cost_t / (S0*q0)  (bps of initial value); cost per cfg.reward_mode.
Obs     : see _obs(); layout is controlled by obs_mode / use_uncertainty / use_regime.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from interfaces import Observation

OBS_MODES = ("oracle", "raw_proxy", "filtered")


@dataclass
class EnvConfig:
    obs_mode: str = "filtered"        # ablation (a)
    use_uncertainty: bool = True      # ablation (b)
    use_regime: bool = True           # ablation (c)
    include_kappa: bool = True        # also expose permanent-impact kappa features
    n_regimes: int = 2                # must equal the filter's K (1 for a single-regime filter)
    action_multipliers: tuple = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    reward_mode: str = "decomp"       # "decomp" (low variance) | "is" (exact IS increment)
    reward_scale: float = 1.0
    std_scale: float = 10.0           # posterior std / ref is small; scaled up. NOT TUNED.
    # normalisers = ObsParams / SimParams defaults
    spread_ref: float = 0.05
    volume_ref: float = 5000.0
    vol_ref: float = 0.02
    alpha_ref: float = 1.5e-4
    kappa_ref: float = 1.25e-5
    sim_kwargs: dict = field(default_factory=dict)   # e.g. shock_step, shock_jump_alpha_mult


class BeliefMDPEnv:
    def __init__(self, cfg: EnvConfig, sim, obs_model, filt):
        assert cfg.obs_mode in OBS_MODES and cfg.reward_mode in ("decomp", "is")
        assert 1.0 in cfg.action_multipliers
        self.cfg, self.sim, self.om, self.filt = cfg, sim, obs_model, filt
        self.p = sim.p
        self.N = self.p.N
        self.n_actions = len(cfg.action_multipliers)
        self.twap_action = list(cfg.action_multipliers).index(1.0)
        self.obs_dim = self._obs_dim()
        self._done = True

    # ---------------- observation layout ----------------
    def _obs_dim(self) -> int:
        c, K = self.cfg, self.cfg.n_regimes
        ch = 2 if c.include_kappa else 1
        d = 2                                        # inventory frac, time remaining frac
        if c.obs_mode == "raw_proxy":
            return d + 3                             # spread, volume, realvol
        d += 1                                       # spread proxy
        if c.obs_mode == "oracle":
            return d + ch + (K if c.use_regime else 0)
        if c.use_regime:
            return d + K * ch * (2 if c.use_uncertainty else 1) + K
        return d + ch * (2 if c.use_uncertainty else 1)

    def _obs(self) -> np.ndarray:
        c, s, px = self.cfg, self._s, self._px
        f = [s["q"] / self.p.q0, (self.N - s["t"]) / self.N]
        if c.obs_mode == "raw_proxy":
            f += [px["spread"] / c.spread_ref, px["volume"] / c.volume_ref, px["realvol"] / c.vol_ref]
        else:
            f.append(px["spread"] / c.spread_ref)
            if c.obs_mode == "oracle":
                f.append(s["alpha"] / c.alpha_ref)
                if c.include_kappa:
                    f.append(s["kappa"] / c.kappa_ref)
                if c.use_regime:
                    if "regime" not in s:
                        raise KeyError("oracle+use_regime needs state['regime'] (RegimeSwitchingSimulator)")
                    f += list(np.eye(c.n_regimes)[int(s["regime"])])
            else:
                b, ss = self.belief, c.std_scale
                if c.use_regime:
                    per = b.per_regime
                    if len(per) != c.n_regimes:
                        raise ValueError(f"filter K={len(per)} != cfg.n_regimes={c.n_regimes}")
                    f += [r.alpha_mean / c.alpha_ref for r in per]
                    if c.use_uncertainty:
                        f += [r.alpha_std / c.alpha_ref * ss for r in per]
                    if c.include_kappa:
                        f += [r.kappa_mean / c.kappa_ref for r in per]
                        if c.use_uncertainty:
                            f += [r.kappa_std / c.kappa_ref * ss for r in per]
                    f += list(b.regime_probs)
                else:                                # pooled moments only, regime probs hidden
                    f.append(b.alpha_mean / c.alpha_ref)
                    if c.use_uncertainty:
                        f.append(b.alpha_std / c.alpha_ref * ss)
                    if c.include_kappa:
                        f.append(b.kappa_mean / c.kappa_ref)
                        if c.use_uncertainty:
                            f.append(b.kappa_std / c.kappa_ref * ss)
        out = np.asarray(f, dtype=np.float32)
        assert out.shape == (self.obs_dim,), (out.shape, self.obs_dim)
        return out

    # ---------------- sensing ----------------
    def _sense(self):
        s = self._s
        try:
            px = self.om.observe_step(s["kappa"], s["alpha"], regime=int(s.get("regime", 0)))
        except TypeError:                            # Gaussian model without regime kwarg
            px = self.om.observe_step(s["kappa"], s["alpha"])
        self._px = px
        if self.cfg.obs_mode == "filtered":
            self.belief = self.filt.update(Observation.from_dict(int(s["t"]), px))

    # ---------------- Gym-style API ----------------
    def reset(self, seed: int = 0):
        self._gen = self.sim.run_stepwise(seed=seed, **self.cfg.sim_kwargs)
        self._s = next(self._gen)
        if hasattr(self.om, "reset"):
            self.om.reset(seed + 10_000)
        elif hasattr(self.om, "rng"):
            self.om.rng = np.random.default_rng(seed + 10_000)
        if self.cfg.obs_mode == "filtered":
            self.filt.reset(seed + 20_000)
        self.belief = None
        self._sense()
        self._done, self.log, self._is_sum, self._dec_sum = False, [], 0.0, 0.0
        self.episode, self._seed = None, seed
        return self._obs(), {}

    def step(self, action: int):
        assert not self._done, "call reset()"
        c, p, s = self.cfg, self.p, self._s
        t, q = int(s["t"]), float(s["q"])
        v = q if t >= self.N - 1 else min(c.action_multipliers[action] * q / (self.N - t), q)
        q_after = q - v
        is_inc = v * (p.S0 - s["S"] + s["alpha"] * v)              # exact IS increment
        dec = s["alpha"] * v * v + s["kappa"] * v * q_after        # drift-only cost
        cost = dec if c.reward_mode == "decomp" else is_inc
        reward = -1e4 * cost / (p.S0 * p.q0) * c.reward_scale
        self._is_sum += is_inc
        self._dec_sum += dec

        b = self.belief
        row = dict(t=t, action=int(action), v=float(v), q_before=q, S=float(s["S"]),
                   reward=float(reward), cost_is=float(is_inc), cost_decomp=float(dec),
                   true_alpha=float(s["alpha"]), true_kappa=float(s["kappa"]),
                   true_regime=int(s["regime"]) if "regime" in s else None)
        info = {"v": v}
        if b is not None:
            row.update(alpha_mean=b.alpha_mean, alpha_std=b.alpha_std, kappa_mean=b.kappa_mean,
                       kappa_std=b.kappa_std, regime_probs=list(b.regime_probs))
            info["alpha_std_norm"] = b.alpha_std / c.alpha_ref     # hook for 3C exploration bonus
            info["kappa_std_norm"] = b.kappa_std / c.kappa_ref
        self.log.append(row)

        try:
            self._s = self._gen.send(v)
            terminated = False
        except StopIteration as stop:
            self.episode, terminated = stop.value, True
        if terminated:
            self._done = True
            info.update(IS=float(self.episode.IS), IS_from_steps=float(self._is_sum))
            return np.zeros(self.obs_dim, dtype=np.float32), reward, True, False, info
        self._sense()
        return self._obs(), reward, False, False, info

    def get_episode(self) -> dict:
        """Per-episode log. PROVISIONAL: not yet matched to the Stage 0 trajectory schema."""
        assert self._done and self.episode is not None
        return dict(seed=self._seed, config=asdict(self.cfg), IS=float(self.episode.IS),
                    IS_from_steps=float(self._is_sum), sum_decomp=float(self._dec_sum),
                    q0=self.p.q0, N=self.N, steps=self.log)