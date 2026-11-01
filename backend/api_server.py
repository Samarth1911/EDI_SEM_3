"""
api_server.py -- Stage 4C HTTP layer (Stage 0, Decision 4 option A). NOT RUN by its author.
Run from EDI/backend:   uvicorn api_server:app --port 8000        (needs: pip install fastapi uvicorn, pydantic v2)
Use IMM_PARTICLES=2000 when hosting filtered agents (the Stage 3 evaluation setting).

  POST /api/episode              body: {seed, obs_kind, regime_preset, policies?, guards?}  -> episode descriptor (no step run yet)
  POST /api/episode/{id}/step    body: STEP_REQUEST_SCHEMA {inject_shock?, force_regime?} -> STEP_RESPONSE_SCHEMA
  GET  /api/episode/{id}/guards  Tier 1 guard status (exploratory)
  GET  /api/episode/{id}/result  after done: IS per policy vs TWAP + contract errors (+ guard report)
  GET  /api/policies             benchmarks, and the trained agents whose checkpoints exist, per obs_kind

Trained agents: ens_<arm> (Gaussian only) and ddqn_<arm>_s<k>; their policy entries carry extra fields action / mult.
Status codes for /step: 422 invalid request values, 409 episode finished / broken / other runtime failure,
500 the step response violated the contract. A failed step is rolled back inside LiveEpisode, so it can be retried.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from live_session import LiveEpisode, DEFAULT_POLICIES, ContractError   # importing live_session puts ../src and ../validation on sys.path
from guards import GuardConfig
from agent_policies import agent_names
import interfaces as I

MAX_SESSIONS = 50
IMM_PARTICLES = int(os.environ.get("IMM_PARTICLES", "2000"))

app = FastAPI(title="EDI live-step API", version=I.SCHEMA_VERSION)
# Local dashboard (serve.py) runs on another port. Wide open for development; tighten before any deployment.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_sessions: "OrderedDict[str, LiveEpisode]" = OrderedDict()


class ForceRegime(BaseModel):
    model_config = ConfigDict(extra="forbid")
    regime: int = Field(ge=0)
    duration: int = Field(ge=1)


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")          # STEP_REQUEST_SCHEMA: additionalProperties False
    inject_shock: Optional[float] = Field(None, ge=2, le=5)
    force_regime: Optional[ForceRegime] = None


class GuardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policies: Optional[list[str]] = None               # None = every hosted policy except 'twap'
    breaker_usd: Optional[float] = Field(None, gt=0)   # exploratory: did not work in the offline study
    constrain_in_stress: bool = False
    stress_regimes: Optional[list[int]] = None         # None = detect by regime name ('stress' / 'crisis')
    stress_prob: float = Field(0.5, ge=0, le=1)
    stress_lo: float = Field(0.75, ge=0)
    stress_hi: float = Field(1.25, ge=0)


class EpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    obs_kind: Literal["gaussian", "spread_hawkes"] = "gaussian"
    regime_preset: Literal["default", "subtle", "single"] = "default"
    policies: Optional[list[str]] = None
    guards: Optional[GuardRequest] = None
    particle_sample: int = Field(0, ge=0, le=2000)   # 0 = off; else particles per regime in each step response


def _get(eid: str) -> LiveEpisode:
    if eid not in _sessions:
        raise HTTPException(404, f"unknown episode {eid}")
    return _sessions[eid]


@app.get("/api/policies")
def policies():
    return {"default": list(DEFAULT_POLICIES), "schema_version": I.SCHEMA_VERSION,
            "agents": {"gaussian": agent_names("gaussian"), "spread_hawkes": agent_names("spread_hawkes")},
            "note": "agents need the matching obs_kind; ensembles are Gaussian-only; filt_single is not hosted; "
                    "run the server with IMM_PARTICLES=2000 for filtered agents"}


@app.post("/api/episode")
def create_episode(req: EpisodeRequest):
    try:
        gcfg = None
        if req.guards is not None:
            g = req.guards
            gcfg = GuardConfig(policies=tuple(g.policies) if g.policies is not None else None,
                               breaker_usd=g.breaker_usd, constrain_in_stress=g.constrain_in_stress,
                               stress_regimes=tuple(g.stress_regimes) if g.stress_regimes is not None else None,
                               stress_prob=g.stress_prob, stress_lo=g.stress_lo, stress_hi=g.stress_hi)
        ep = LiveEpisode(req.seed, req.obs_kind, req.regime_preset, tuple(req.policies or DEFAULT_POLICIES),
                         imm_particles=IMM_PARTICLES, guards=gcfg, particle_sample=req.particle_sample)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _sessions[ep.id] = ep
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return {"schema_version": I.SCHEMA_VERSION, "episode_id": ep.id, "seed": ep.seed, "N": ep.N, "t": 0,
            "done": False, "config": ep.config(), "policies": [p.name for p in ep.policies],
            "guards": ep.guard_report(), "notes": ep.notes}


@app.post("/api/episode/{eid}/step")
def step(eid: str, req: StepRequest):
    ep = _get(eid)
    try:
        return ep.step(inject_shock=req.inject_shock,
                       force_regime=req.force_regime.model_dump() if req.force_regime else None)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except ContractError as e:                         # must come before RuntimeError (it is a subclass)
        raise HTTPException(500, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/episode/{eid}/guards")
def guards(eid: str):
    return _get(eid).guard_report()


@app.get("/api/episode/{eid}/result")
def result(eid: str):
    ep = _get(eid)
    try:
        return ep.result()
    except RuntimeError as e:
        raise HTTPException(409, str(e))