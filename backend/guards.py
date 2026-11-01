"""
guards.py -- Stage 4C Tier 1 safeguards: wrappers around ANY policy, no retraining. NOT RUN by its author.

  Circuit breaker: if a guarded policy's mark-to-market cost exceeds the hosted TWAP policy's by more than breaker_usd, the
    policy is switched off for the rest of the episode and trades the TWAP rate q/(N-t). Latched (never switches back).
    Mark-to-market cost = S0*q0 - (proceeds + q_remaining*S), so policies that have sold different amounts compare fairly.
  Dynamic action constraint: when the filter's probability mass on the stress regimes is >= stress_prob, the action multiplier
    m = v / (q/(N-t)) is clipped to [stress_lo, stress_hi]. The final step is never clipped (forced liquidation).
The breaker caps further deterioration, not the loss already taken (it reacts one step late, and impact/price-path differences
persist after the switch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

STRESS_KEYWORDS = ("stress", "crisis")


@dataclass(frozen=True)
class GuardConfig:
    policies: Optional[tuple] = None             # names to guard; None = every hosted policy except 'twap'
    breaker_usd: Optional[float] = None          # None = breaker off
    constrain_in_stress: bool = False
    stress_regimes: Optional[tuple] = None       # regime indices; None = detect by name ('stress'/'crisis')
    stress_prob: float = 0.5
    stress_lo: float = 0.75
    stress_hi: float = 1.25

    def __post_init__(self):
        if self.breaker_usd is not None and not self.breaker_usd > 0:
            raise ValueError("breaker_usd must be > 0")
        if not 0.0 <= self.stress_prob <= 1.0:
            raise ValueError("stress_prob must be in [0, 1]")
        if not 0.0 <= self.stress_lo <= self.stress_hi:
            raise ValueError("need 0 <= stress_lo <= stress_hi")

    @property
    def active(self) -> bool:
        return self.breaker_usd is not None or self.constrain_in_stress


def resolve_stress(cfg: GuardConfig, regime_names, K: int):
    """Stress regime indices, or None when constraining is off."""
    if not cfg.constrain_in_stress:
        return None
    if cfg.stress_regimes is not None:
        idx = tuple(sorted(set(int(i) for i in cfg.stress_regimes)))
    else:
        idx = tuple(i for i, n in enumerate(regime_names) if any(k in str(n).lower() for k in STRESS_KEYWORDS))
    if not idx:
        raise ValueError(f"no stress regime identified from names {list(regime_names)}; pass stress_regimes explicitly")
    if any(not 0 <= i < K for i in idx):
        raise ValueError(f"stress_regimes must be in [0, {K - 1}]")
    return idx


def mtm_cost(S0: float, q0: float, proceeds: float, q_remaining: float, S: float) -> float:
    """Mark-to-market implementation shortfall in dollars: remaining inventory valued at the current price S."""
    return float(S0 * q0 - (proceeds + q_remaining * S))


def _probs(belief):
    rp = belief["regime_probs"] if isinstance(belief, dict) else belief.regime_probs
    return [float(x) for x in rp]


class GuardState:
    def __init__(self, cfg: GuardConfig, names, stress):
        self.cfg, self.names, self.stress = cfg, set(names), stress
        self.tripped: dict = {}                  # policy name -> step at which its breaker tripped
        self.log: list = []

    # rollback support (LiveEpisode snapshots/restores around every step)
    def snapshot(self):
        return (dict(self.tripped), list(self.log))

    def restore(self, s) -> None:
        self.tripped, self.log = dict(s[0]), list(s[1])

    def guarded(self, name: str) -> bool:
        return name in self.names

    def p_stress(self, belief) -> float:
        if not self.stress:
            return 0.0
        p = _probs(belief)
        return float(sum(p[i] for i in self.stress))

    def breaker(self, name: str, t: int, gap_usd: float) -> bool:
        """True if this policy is (now or already) switched to TWAP."""
        if name in self.tripped:
            return True
        if self.cfg.breaker_usd is not None and gap_usd > self.cfg.breaker_usd:
            self.tripped[name] = t
            self.log.append({"type": "breaker_tripped", "policy": name, "t": t,
                             "gap_usd": float(gap_usd), "limit_usd": float(self.cfg.breaker_usd)})
            return True
        return False

    def constrain(self, name: str, t: int, N: int, q: float, v: float, p_stress: float) -> float:
        if self.stress is None or t >= N - 1 or q <= 0.0 or p_stress < self.cfg.stress_prob:
            return v
        base = q / (N - t)
        out = min(max(v, self.cfg.stress_lo * base), self.cfg.stress_hi * base, q)
        if out != v:
            self.log.append({"type": "action_constrained", "policy": name, "t": t, "v_raw": float(v), "v": float(out),
                             "mult_raw": float(v / base), "mult": float(out / base), "p_stress": float(p_stress)})
        return float(out)

    def report(self) -> dict:
        return {"enabled": True, "breaker_usd": self.cfg.breaker_usd, "constrain_in_stress": self.cfg.constrain_in_stress,
                "stress_regimes": list(self.stress) if self.stress is not None else None,
                "stress_prob": self.cfg.stress_prob, "stress_band": [self.cfg.stress_lo, self.cfg.stress_hi],
                "guarded": sorted(self.names), "tripped": dict(self.tripped), "log": list(self.log)}