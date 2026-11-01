"""
interfaces.py  (STAGE 0 -- shared data contracts)
--------------------------------------------------
Single source of truth for the data that crosses module boundaries. Every stage
imports these types instead of inventing its own dicts. Nothing here changes the
validated simulator / filter code: the `from_*` adapters wrap what those modules
ALREADY produce, and `validation/validate_interfaces.py` proves the fit end to end.

STATUS: DRAFT for team ratification. Items marked  [DECISION]  in
docs/STAGE0_INTERFACES.md must be agreed before Stages 2-4 build against this.

Conventions (see the doc for the reasoning):
  * Discrete time t = 0..N-1 are DECISION steps; state arrays (kappa, alpha, S,
    regime, q) have N+1 entries, observations/beliefs/actions have N.
  * y_t is measured over the window ((t-1)*tau, t*tau] and depends only on x_t
    and regime[t]. The decision v_t uses information I_t = {y_0..y_t}.
    v_t is executed over [t*tau, (t+1)*tau).
  * TRUTH (kappa, alpha, regime) is oracle-only. Belief-based policies never see it.
  * Units: SI-like as in simulator.py: time in units of the horizon T (tau = T/N),
    prices in $, kappa/alpha in $/share^2-type impact units (~1e-5 .. 1e-3).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Protocol, Sequence, runtime_checkable
import json
import math

import numpy as np

SCHEMA_VERSION = "0.1-draft"

# Proxy channels the particle filter is allowed to read (by name).
CORE_CHANNELS = ("spread", "volume", "realvol")
# Optional extra channels some observation models add (a count/tick-based
# likelihood in Stage 2 may use them; the Gaussian PF ignores them).
EXTRA_CHANNELS = ("spread_ticks", "n_open", "n_close", "n_events", "hawkes_excitation")

# Regime ids. K = number of regimes; calm/stressed in Stage 1.
REGIME_CALM, REGIME_STRESSED = 0, 1


# =============================================================================
# C0. Reproducibility: one seed -> independent named RNG streams
# =============================================================================
def seed_streams(seed: int) -> dict:
    """Standard seed offsets so every policy sees identical market noise
    (common random numbers). Matches heuristic_policy.run_policy_episode
    (obs = seed+10_000, filter = seed+20_000); the regime chain already derives
    its own stream from [seed, REGIME_RNG_TAG] inside regime_simulator.py."""
    return {"sim": seed, "obs": seed + 10_000, "filter": seed + 20_000, "policy": seed + 30_000}


# =============================================================================
# C1. Simulator -> Observation model / oracle  (ground truth, one timestep)
# =============================================================================
@dataclass
class TruthStep:
    """Ground truth at decision step t, BEFORE the trade v_t is chosen.
    Mirrors the `state` dict yielded by (Regime)MarketSimulator.run_stepwise."""
    t: int
    S: float                    # mid-price
    kappa: float                # TRUE permanent impact      (oracle only)
    alpha: float                # TRUE temporary impact      (oracle only)
    regime: int                 # TRUE regime id in force    (oracle only); 0 for single-regime sims
    q: float                    # inventory remaining
    N: int                      # number of decision steps
    shock_step: Optional[int] = None   # scheduled jump shock (within-regime), if any

    @classmethod
    def from_state(cls, state: dict) -> "TruthStep":
        return cls(t=int(state["t"]), S=float(state["S"]), kappa=float(state["kappa"]),
                   alpha=float(state["alpha"]), regime=int(state.get("regime", 0)),
                   q=float(state["q"]), N=int(state["N"]), shock_step=state.get("shock_step"))


@dataclass
class WorldConfig:
    """Everything needed to regenerate an episode. JSON-serialisable.
    `sim`/`regime`/`obs`/`hawkes` are plain dicts (dataclasses.asdict of SimParams,
    RegimeParams, ObsParams, Hawkes params) so the config can be logged and hashed."""
    sim: dict
    regime: Optional[dict] = None       # None => single regime
    obs_kind: str = "gaussian"          # 'gaussian' | 'hawkes' | 'spread_hawkes'
    obs: Optional[dict] = None
    hawkes: Optional[dict] = None
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        def default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            raise TypeError(type(o))
        return json.dumps(asdict(self), default=default, sort_keys=True)


@runtime_checkable
class Simulator(Protocol):
    """Pull-based generator protocol (already implemented by MarketSimulator and
    RegimeSwitchingSimulator): state = next(gen); state = gen.send(v_t);
    StopIteration.value = final Episode."""
    def run_stepwise(self, *args, **kwargs): ...


# =============================================================================
# C2. Observation model -> Particle filter  (noisy proxy vector at step t)
# =============================================================================
@dataclass
class Observation:
    t: int
    spread: float               # $ quoted spread proxy
    volume: float               # market volume proxy
    realvol: float              # realised-volatility proxy
    extras: dict = field(default_factory=dict)   # subset of EXTRA_CHANNELS, if the model provides them

    @classmethod
    def from_dict(cls, t: int, d: dict) -> "Observation":
        return cls(t=int(t), spread=float(d["spread"]), volume=float(d["volume"]),
                   realvol=float(d["realvol"]),
                   extras={k: float(d[k]) for k in EXTRA_CHANNELS if k in d})

    def core_dict(self) -> dict:
        """The dict the existing ParticleFilter.step() reads (by name)."""
        return {"spread": self.spread, "volume": self.volume, "realvol": self.realvol}


@runtime_checkable
class ObservationModel(Protocol):
    """All three models (gaussian, hawkes, spread_hawkes) satisfy this. `regime`
    is a GENERATOR-side input (the true regime); it must never be forwarded to a
    filter. Stateful models (hawkes*) must be called in time order after reset()."""
    kind: str
    def reset(self, seed: Optional[int] = None) -> None: ...
    def observe_step(self, kappa_t: float, alpha_t: float, regime: int = 0, rng=None) -> dict: ...


# =============================================================================
# C3. Particle filter -> Decision layer  (belief state)
# =============================================================================
@dataclass
class RegimeBelief:
    """Posterior of the continuous state CONDITIONAL on one regime."""
    regime: int
    prob: float                 # p(s_t = regime | y_0..y_t)
    kappa_mean: float
    kappa_std: float
    alpha_mean: float
    alpha_std: float
    ess: Optional[float] = None


@dataclass
class BeliefState:
    """Filtering posterior p(x_t, s_t | y_0..y_t) (REWEIGHTED, PRE-resample, as in
    the validated bootstrap PF). Pooled fields are the moments of the MIXTURE over
    regimes and are what the existing heuristic policy reads. A single-regime
    filter fills K = 1, regime_probs = [1.0], per_regime = [that one belief]."""
    t: int
    regime_probs: list                     # length K, sums to 1
    kappa_mean: float                      # pooled (mixture) moments
    kappa_std: float
    alpha_mean: float
    alpha_std: float
    ess: float                             # total effective sample size (see doc for K>1 definition)
    per_regime: list = field(default_factory=list)   # list[RegimeBelief], length K (may be [] for pooled-only filters)
    n_particles: Optional[int] = None
    loglik_increment: Optional[float] = None         # log p(y_t | y_0..t-1), for calibration checks

    @property
    def K(self) -> int:
        return len(self.regime_probs)

    def to_vector(self) -> np.ndarray:
        """Fixed-order feature vector for learned policies:
        [alpha_mean, alpha_std, kappa_mean, kappa_std, p_regime_0..p_regime_{K-1}]
        (raw units; scaling is the policy's job). Length 4 + K."""
        return np.array([self.alpha_mean, self.alpha_std, self.kappa_mean, self.kappa_std,
                         *self.regime_probs], dtype=float)


@runtime_checkable
class Estimator(Protocol):
    """Bootstrap PF, IMM-PF and any later filter implement this."""
    def reset(self, seed: Optional[int] = None) -> None: ...
    def update(self, obs: Observation) -> BeliefState: ...


def belief_from_bootstrap_pf(pf, t: int) -> BeliefState:
    """Adapter: read the latest posterior out of the validated single-regime
    ParticleFilter (particle_filter.py) WITHOUT modifying it."""
    h = pf.history
    a_m, a_s = float(h["alpha_mean"][-1]), float(h["alpha_std"][-1])
    k_m, k_s = float(h["kappa_mean"][-1]), float(h["kappa_std"][-1])
    ess = float(h["ess"][-1])
    rb = RegimeBelief(0, 1.0, k_m, k_s, a_m, a_s, ess)
    return BeliefState(t=t, regime_probs=[1.0], kappa_mean=k_m, kappa_std=k_s,
                       alpha_mean=a_m, alpha_std=a_s, ess=ess, per_regime=[rb],
                       n_particles=int(pf.pp.n_particles))


def validate_belief(b: BeliefState, K: Optional[int] = None) -> list:
    """Return a list of contract violations (empty list = OK)."""
    errs = []
    p = np.asarray(b.regime_probs, dtype=float)
    if K is not None and len(p) != K:
        errs.append(f"len(regime_probs)={len(p)} != K={K}")
    if np.any(p < -1e-12) or not math.isclose(float(p.sum()), 1.0, abs_tol=1e-6):
        errs.append(f"regime_probs not a distribution: {p}")
    for name in ("kappa_mean", "kappa_std", "alpha_mean", "alpha_std", "ess"):
        v = getattr(b, name)
        if not math.isfinite(v):
            errs.append(f"{name} not finite")
    if b.alpha_std < 0 or b.kappa_std < 0:
        errs.append("negative std")
    if b.per_regime and len(b.per_regime) != len(p):
        errs.append("per_regime length != K")
    return errs


# =============================================================================
# C4. Decision layer -> Evaluation  (logged trading episode)
# =============================================================================
@dataclass
class PolicyInput:
    """What a policy is allowed to see at step t. `truth` is None unless the
    policy declares uses_oracle_truth = True (harness enforces this).
    `obs` is the learned-policy observation vector (DDQN agents/ensembles); None for TWAP/CE/heuristic."""
    t: int
    N: int
    q: float                              # inventory remaining
    belief: Optional[BeliefState]         # None for belief-free policies (TWAP)
    truth: Optional[TruthStep] = None     # oracle benchmark only
    obs: Optional[np.ndarray] = None      # agent observation vector (see belief_mdp_env); None unless a learned policy is hosted

@runtime_checkable
class Policy(Protocol):
    name: str
    uses_oracle_truth: bool
    def reset(self, seed: Optional[int] = None) -> None: ...
    def act(self, x: PolicyInput) -> float: ...       # shares to sell in interval t (clipped to [0, q] by the harness)


@dataclass
class StepLog:
    t: int
    q_before: float
    v: float
    S: float                              # mid-price before the trade
    S_exec: float                         # execution price received = S - alpha*v
    obs: Optional[dict] = None            # Observation.core_dict() + extras, if logged
    belief: Optional[dict] = None         # asdict(BeliefState), if logged
    # --- truth, EVALUATION ONLY (never fed to a belief-based policy) ---
    true_kappa: Optional[float] = None
    true_alpha: Optional[float] = None
    true_regime: Optional[int] = None


@dataclass
class EpisodeLog:
    """One trading episode under one policy. Uniform across all policies so that
    benchmarks are computed by one function."""
    policy: str
    seed: int
    world: WorldConfig
    steps: list                           # list[StepLog], length N
    IS: float                             # implementation shortfall = S0*q0 - sum(S_exec*v)
    q0: float
    S0: float
    q_final: float                        # inventory left (should be ~0; report it, do not hide it)
    shock_step: Optional[int] = None
    schema_version: str = SCHEMA_VERSION

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for s in self.steps:
            r = {"t": s.t, "q_before": s.q_before, "v": s.v, "S": s.S, "S_exec": s.S_exec,
                 "true_kappa": s.true_kappa, "true_alpha": s.true_alpha, "true_regime": s.true_regime}
            if s.belief:
                r.update({"alpha_hat": s.belief["alpha_mean"], "alpha_std": s.belief["alpha_std"],
                          "kappa_hat": s.belief["kappa_mean"], "kappa_std": s.belief["kappa_std"]})
                for i, p in enumerate(s.belief["regime_probs"]):
                    r[f"p_regime_{i}"] = p
            rows.append(r)
        return pd.DataFrame(rows)


def episode_log_from_episode(policy: str, seed: int, world: WorldConfig, ep, *,
                             obs_rows: Optional[list] = None,
                             beliefs: Optional[list] = None) -> EpisodeLog:
    """Adapter: convert an (Regime)Episode returned by run_stepwise into an EpisodeLog.
    obs_rows: optional list[dict] length N; beliefs: optional list[BeliefState] length N."""
    N = len(ep.v)
    reg = getattr(ep, "regime", None)
    steps = []
    for t in range(N):
        steps.append(StepLog(
            t=t, q_before=float(ep.q[t]), v=float(ep.v[t]), S=float(ep.S[t]), S_exec=float(ep.S_exec[t]),
            obs=None if obs_rows is None else obs_rows[t],
            belief=None if beliefs is None else asdict(beliefs[t]),
            true_kappa=float(ep.kappa[t]), true_alpha=float(ep.alpha[t]),
            true_regime=None if reg is None else int(reg[t])))
    return EpisodeLog(policy=policy, seed=seed, world=world, steps=steps, IS=float(ep.IS),
                      q0=float(ep.q[0]), S0=float(ep.S[0]), q_final=float(ep.q[-1]),
                      shock_step=ep.shock_step)


def validate_episode_log(log: EpisodeLog, tol: float = 1e-6) -> list:
    """Contract checks every logged episode must pass before it is evaluated."""
    errs = []
    N = len(log.steps)
    if [s.t for s in log.steps] != list(range(N)):
        errs.append("steps not indexed 0..N-1 in order")
    v = np.array([s.v for s in log.steps]); q = np.array([s.q_before for s in log.steps])
    if np.any(v < -tol) or np.any(v > q + tol):
        errs.append("trade v outside [0, q]")
    if N and not math.isclose(q[0], log.q0, rel_tol=1e-9):
        errs.append("q_before[0] != q0")
    IS_recomputed = log.S0 * log.q0 - float(np.sum([s.S_exec * s.v for s in log.steps]))
    if not math.isclose(IS_recomputed, log.IS, rel_tol=1e-9, abs_tol=1e-6):
        errs.append(f"IS mismatch: logged {log.IS} vs recomputed {IS_recomputed}")
    if abs(log.q_final) > 1e-3 * max(log.q0, 1.0):
        errs.append(f"inventory not liquidated: q_final={log.q_final:.3f} (IS then understates cost)")
    return errs


# ---- benchmark metrics: ONE implementation for every policy ------------------
def paired_metrics(logs_by_policy: dict, baseline: str = "twap") -> dict:
    """logs_by_policy: {policy: [EpisodeLog,...]} with the SAME seeds in the SAME order.
    Returns mean IS, std, CVaR95 and paired difference vs `baseline` (mean, s.e.)."""
    base = np.array([l.IS for l in logs_by_policy[baseline]])
    out = {}
    for name, logs in logs_by_policy.items():
        seeds = [l.seed for l in logs]
        if seeds != [l.seed for l in logs_by_policy[baseline]]:
            raise ValueError(f"{name}: seeds differ from baseline -> not paired")
        x = np.array([l.IS for l in logs]); d = x - base
        k = max(1, int(math.ceil(0.05 * len(x))))
        out[name] = {"n": len(x), "IS_mean": float(x.mean()), "IS_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                     "IS_cvar95": float(np.sort(x)[-k:].mean()),
                     "paired_diff_mean": float(d.mean()),
                     "paired_diff_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0}
    return out


# =============================================================================
# C5. Everything -> Backend/Frontend  (live-simulation step API)
# =============================================================================
STEP_RESPONSE_EXAMPLE = {
    "schema_version": SCHEMA_VERSION,
    "episode_id": "ep-000123",
    "seed": 1234,
    "t": 17, "N": 50, "done": False,
    "config": {"obs_kind": "spread_hawkes", "n_regimes": 2, "regime_names": ["calm", "stressed"], "q0": 20000, "S0": 10.0},
    "truth": {"S": 9.98, "kappa": 1.3e-5, "alpha": 2.4e-4, "regime": 1},
    "observation": {"spread": 0.071, "volume": 3120.0, "realvol": 0.023,
                    "extras": {"spread_ticks": 5, "n_events": 4}},
    "belief": {"t": 17, "regime_probs": [0.22, 0.78], "alpha_mean": 2.3e-4, "alpha_std": 1.1e-5,
               "kappa_mean": 1.3e-5, "kappa_std": 6e-7, "ess": 2100.0,
               "per_regime": [{"regime": 0, "prob": 0.22, "alpha_mean": 1.6e-4, "alpha_std": 9e-6,
                               "kappa_mean": 1.25e-5, "kappa_std": 5e-7},
                              {"regime": 1, "prob": 0.78, "alpha_mean": 2.4e-4, "alpha_std": 1e-5,
                               "kappa_mean": 1.35e-5, "kappa_std": 6e-7}]},
    "policies": {
        "twap":   {"q": 13600.0, "v": 400.0, "S": 9.98, "S_exec": 9.92, "cum_cost": 812.4},
        "belief": {"q": 14100.0, "v": 310.0, "S": 9.99, "S_exec": 9.93, "cum_cost": 655.0}},
    "events": [{"type": "regime_forced", "t": 15, "regime": 1}]
}

STEP_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "inject_shock": {"type": ["number", "null"], "minimum": 2, "maximum": 5,
                         "description": "multiply alpha by this at the NEXT step (validated range x2..x5)"},
        "force_regime": {"type": ["object", "null"],
                         "properties": {"regime": {"type": "integer", "minimum": 0},
                                        "duration": {"type": "integer", "minimum": 1}},
                         "required": ["regime", "duration"]}},
    "additionalProperties": False}

STEP_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "episode_id", "seed", "t", "N", "done", "config", "truth",
                 "observation", "belief", "policies", "events"],
    "properties": {
        "schema_version": {"type": "string"}, "episode_id": {"type": "string"}, "seed": {"type": "integer"},
        "t": {"type": "integer", "minimum": 0}, "N": {"type": "integer", "minimum": 1}, "done": {"type": "boolean"},
        "config": {"type": "object", "required": ["obs_kind", "n_regimes", "regime_names", "q0", "S0"]},
        "truth": {"type": "object", "required": ["S", "kappa", "alpha", "regime"]},
        "observation": {"type": "object", "required": ["spread", "volume", "realvol"]},
        "belief": {"type": "object", "required": ["t", "regime_probs", "alpha_mean", "alpha_std",
                                                   "kappa_mean", "kappa_std", "ess"]},
        "policies": {"type": "object",
                     "additionalProperties": {"type": "object",
                                              "required": ["q", "v", "S", "S_exec", "cum_cost"]}},
        "events": {"type": "array", "items": {"type": "object", "required": ["type", "t"]}}}}


def validate_step_response(resp: dict) -> list:
    """Dependency-free structural check of a StepResponse against STEP_RESPONSE_SCHEMA
    (required keys at top level and one level down, plus belief sanity)."""
    errs = []
    for k in STEP_RESPONSE_SCHEMA["required"]:
        if k not in resp:
            errs.append(f"missing key: {k}")
    if errs:
        return errs
    for section in ("config", "truth", "observation", "belief"):
        for k in STEP_RESPONSE_SCHEMA["properties"][section].get("required", []):
            if k not in resp[section]:
                errs.append(f"{section}.{k} missing")
    for name, pol in resp["policies"].items():
        for k in ("q", "v", "S", "S_exec", "cum_cost"):
            if k not in pol:
                errs.append(f"policies.{name}.{k} missing")
    p = resp["belief"].get("regime_probs", [])
    if p and not math.isclose(sum(p), 1.0, abs_tol=1e-6):
        errs.append("belief.regime_probs does not sum to 1")
    if len(p) != resp["config"].get("n_regimes", len(p)):
        errs.append("len(belief.regime_probs) != config.n_regimes")
    return errs


def step_response(*, episode_id: str, seed: int, truth: TruthStep, obs: Observation,
                  belief: BeliefState, policies: dict, config: dict, done: bool,
                  events: Optional[list] = None) -> dict:
    """Build a StepResponse dict (JSON-serialisable) from the typed pieces."""
    return {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "seed": int(seed),
            "t": int(truth.t), "N": int(truth.N), "done": bool(done), "config": config,
            "truth": {"S": truth.S, "kappa": truth.kappa, "alpha": truth.alpha, "regime": truth.regime},
            "observation": {**obs.core_dict(), "extras": obs.extras},
            "belief": asdict(belief), "policies": policies, "events": events or []}