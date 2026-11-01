"""
agent_policies.py -- Stage 4C: trained DDQN agents / ensembles behind the Stage 0 Policy protocol.  NOT RUN by its author.

Names:   ens_<arm>            mean-Q ensemble of the arm's 5 validation-selected members (Gaussian: all 7 hosted arms;
                              spread_hawkes: oracle/raw_proxy/filt_full/filt_mean_only, once all 5 Hawkes checkpoints exist)
         ddqn_<arm>_s<k>      one member (k 0..4 in both families; spread_hawkes arms oracle/raw_proxy/filt_full/filt_mean_only)
Checkpoints: <arm>_base1<tag>_s<k>_valsel.npz in CKPT_DIR (tag _abl, _abl_b1 for bonus arms, _hawkes). The agent family must
match the episode's obs_kind. filt_single is not hosted (needs a K=1 filter). build_obs mirrors BeliefMDPEnv._obs exactly.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np

from belief_mdp_env import EnvConfig
from ddqn import DDQNAgent
import interfaces as I

CKPT_DIR = os.environ.get("EDI_CKPT_DIR", os.path.join(HERE, "..", "validation", "outputs", "stage3c", "ckpt"))
AGENT_TRAIN_PARTICLES = 2000          # score_ensemble.py default; filtered agents were evaluated with this filter setting

ARMS = {
    "oracle": dict(obs_mode="oracle"),
    "raw_proxy": dict(obs_mode="raw_proxy"),
    "filt_full": dict(obs_mode="filtered"),
    "filt_mean_only": dict(obs_mode="filtered", use_uncertainty=False),
    "filt_no_regime": dict(obs_mode="filtered", use_regime=False),
    "filt_full_bonus": dict(obs_mode="filtered"),
    "filt_mean_bonus": dict(obs_mode="filtered", use_uncertainty=False),
}
HAWKES_ARMS = ("oracle", "raw_proxy", "filt_full", "filt_mean_only")
N_SEEDS = {"gaussian": 5, "spread_hawkes": 5}      # Hawkes was 3 seeds until the arm was completed to 5; missing files raise "checkpoint missing"


def arm_config(arm: str) -> EnvConfig:
    return EnvConfig(**ARMS[arm], n_regimes=2, reward_scale=0.1)


def expected_obs_dim(c: EnvConfig) -> int:
    K, ch, d = c.n_regimes, (2 if c.include_kappa else 1), 2
    if c.obs_mode == "raw_proxy":
        return d + 3
    d += 1
    if c.obs_mode == "oracle":
        return d + ch + (K if c.use_regime else 0)
    if c.use_regime:
        return d + K * ch * (2 if c.use_uncertainty else 1) + K
    return d + ch * (2 if c.use_uncertainty else 1)


def build_obs(c: EnvConfig, *, q: float, q0: float, t: int, N: int, px, belief, truth) -> np.ndarray:
    """Same features, order and scaling as BeliefMDPEnv._obs. px: interfaces.Observation; truth only for the oracle arm."""
    f = [q / q0, (N - t) / N]
    if c.obs_mode == "raw_proxy":
        f += [px.spread / c.spread_ref, px.volume / c.volume_ref, px.realvol / c.vol_ref]
    else:
        f.append(px.spread / c.spread_ref)
        if c.obs_mode == "oracle":
            f.append(truth.alpha / c.alpha_ref)
            if c.include_kappa:
                f.append(truth.kappa / c.kappa_ref)
            if c.use_regime:
                f += list(np.eye(c.n_regimes)[int(truth.regime)])
        else:
            ss = c.std_scale
            if c.use_regime:
                per = belief.per_regime
                if len(per) != c.n_regimes:
                    raise ValueError(f"filter K={len(per)} != n_regimes={c.n_regimes}")
                f += [r.alpha_mean / c.alpha_ref for r in per]
                if c.use_uncertainty:
                    f += [r.alpha_std / c.alpha_ref * ss for r in per]
                if c.include_kappa:
                    f += [r.kappa_mean / c.kappa_ref for r in per]
                    if c.use_uncertainty:
                        f += [r.kappa_std / c.kappa_ref * ss for r in per]
                f += list(belief.regime_probs)
            else:
                f.append(belief.alpha_mean / c.alpha_ref)
                if c.use_uncertainty:
                    f.append(belief.alpha_std / c.alpha_ref * ss)
                if c.include_kappa:
                    f.append(belief.kappa_mean / c.kappa_ref)
                    if c.use_uncertainty:
                        f.append(belief.kappa_std / c.kappa_ref * ss)
    out = np.asarray(f, dtype=np.float32)
    if out.shape != (expected_obs_dim(c),):
        raise ValueError(f"observation shape {out.shape} != {(expected_obs_dim(c),)}")
    return out


def _tag(arm: str, obs_kind: str) -> str:
    return "_hawkes" if obs_kind == "spread_hawkes" else ("_abl_b1" if arm.endswith("_bonus") else "_abl")


def _path(arm: str, obs_kind: str, k: int) -> str:
    return os.path.join(CKPT_DIR, f"{arm}_base1{_tag(arm, obs_kind)}_s{k}_valsel.npz")


def parse_agent_name(name: str):
    """-> ('ens', arm, None) | ('single', arm, k) | None"""
    for arm in ARMS:
        if name == f"ens_{arm}":
            return ("ens", arm, None)
        pre = f"ddqn_{arm}_s"
        if name.startswith(pre) and name[len(pre):].isdigit():
            return ("single", arm, int(name[len(pre):]))
    return None


def is_agent_name(name: str) -> bool:
    return name.startswith(("ens_", "ddqn_"))


def agent_names(obs_kind: str) -> list:
    """Names whose checkpoint files exist for this observation kind (an ensemble needs all N_SEEDS members)."""
    if obs_kind not in N_SEEDS:
        return []
    out, n = [], N_SEEDS[obs_kind]
    for arm in (ARMS if obs_kind == "gaussian" else HAWKES_ARMS):
        if all(os.path.exists(_path(arm, obs_kind, k)) for k in range(n)):
            out.append(f"ens_{arm}")
        out += [f"ddqn_{arm}_s{k}" for k in range(n) if os.path.exists(_path(arm, obs_kind, k))]
    return out


_CACHE: dict = {}


def _load(path: str) -> DDQNAgent:
    if path not in _CACHE:
        if not os.path.exists(path):
            raise ValueError(f"checkpoint missing: {path}")
        _CACHE[path] = DDQNAgent.load(path)
    return _CACHE[path]


class AgentPolicy:
    needs_obs = True

    def __init__(self, name: str, arm: str, members: list):
        self.name, self.arm, self.members = name, arm, members
        self.cfg = arm_config(arm)
        self.obs_dim = expected_obs_dim(self.cfg)
        self.mults = [float(m) for m in self.cfg.action_multipliers]
        self.uses_oracle_truth = self.cfg.obs_mode == "oracle"
        self.uses_belief = self.cfg.obs_mode == "filtered"
        self.last_action = None
        for m in members:
            if m.obs_dim != self.obs_dim or m.n_actions != len(self.mults):
                raise ValueError(f"{name}: checkpoint (obs_dim {m.obs_dim}, actions {m.n_actions}) does not match "
                                 f"arm config ({self.obs_dim}, {len(self.mults)})")

    def reset(self, seed=None) -> None:
        self.last_action = None

    def act(self, x: I.PolicyInput) -> float:
        if getattr(x, "obs", None) is None:
            raise ValueError(f"{self.name} needs PolicyInput.obs (add the field to interfaces.PolicyInput)")
        obs = np.asarray(x.obs)
        if obs.shape != (self.obs_dim,) or not np.all(np.isfinite(obs)):
            raise ValueError(f"{self.name}: bad observation vector {obs.shape}")
        a = int(np.argmax(np.mean([m.q_values(obs)[0] for m in self.members], axis=0)))
        self.last_action = a
        v = x.q if x.t >= x.N - 1 else min(self.mults[a] * x.q / (x.N - x.t), x.q)
        return float(min(max(v, 0.0), x.q))


def make_agent_policy(name: str, obs_kind: str) -> AgentPolicy:
    p = parse_agent_name(name)
    if p is None:
        raise ValueError(f"unknown agent policy {name!r} (filt_single is not hosted); see /api/policies")
    kind, arm, k = p
    if obs_kind == "spread_hawkes":
        if arm not in HAWKES_ARMS:
            raise ValueError(f"no spread_hawkes agents for arm {arm}; available arms: {list(HAWKES_ARMS)}")
    elif obs_kind != "gaussian":
        raise ValueError(f"no agents for obs_kind {obs_kind}")
    n = N_SEEDS[obs_kind]
    if kind == "single" and not 0 <= k < n:
        raise ValueError(f"member index must be in [0, {n - 1}] for {obs_kind}")
    members = [_load(_path(arm, obs_kind, j)) for j in (range(n) if kind == "ens" else [k])]
    return AgentPolicy(name, arm, members)