"""env_factory.py -- builds the real simulator + observation model + IMM-PF for BeliefMDPEnv."""
from simulator import SimParams
from observation_model import ObsParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from spread_hawkes import make_observation_model_v2
from imm_particle_filter import IMMParticleFilter, IMMParams
from belief_mdp_env import BeliefMDPEnv, EnvConfig


def make_env(cfg: EnvConfig, obs_kind: str = "gaussian", filter_regimes: str = "imm",
             seed: int = 0, imm_params: IMMParams | None = None, *,
             stationary_init: bool = False, sim_params: SimParams | None = None) -> BeliefMDPEnv:
    """stationary_init=True: world AND filter prior start from the chain's stationary distribution
    (Stage 0 Decision 5). Default False keeps 3A behaviour (every episode starts calm).
    sim_params: override SimParams (used by the sigma=0 exactness test)."""
    sp, op = sim_params or SimParams(), ObsParams()
    world_rp = RegimeParams.default()
    if stationary_init:
        world_rp = RegimeParams(specs=world_rp.specs, P=world_rp.P.copy(),
                                init_probs=world_rp.stationary_distribution())
    filt_rp = world_rp if filter_regimes == "imm" else RegimeParams.single()
    sim = RegimeSwitchingSimulator(sp, world_rp, seed=seed)
    om = make_observation_model_v2(obs_kind, seed=seed)
    filt = IMMParticleFilter(sp, op, filt_rp, imm_params or IMMParams(), seed=seed)
    return BeliefMDPEnv(cfg, sim, om, filt)