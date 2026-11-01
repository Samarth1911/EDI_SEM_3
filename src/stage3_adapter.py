"""
stage3_adapter.py -- Stage 3 -> Stage 4 interface (Person 3).  NOT RUN by its author.  Self-test:  python stage3_adapter.py
(run from src/, needs the project on the path like the other modules).

PROBLEM IT SOLVES
    Stage 0 contract C4 (interfaces.EpisodeLog / StepLog) is what Stage 4 scores every policy with
    (interfaces.paired_metrics). BeliefMDPEnv.get_episode() returns a different, flat dict (action index, reward, cost
    decomposition, belief fields inline, no S_exec / world / q_final), and its own docstring says it was never matched to
    the schema. This module converts the env's last finished episode into a real interfaces.EpisodeLog, so TWAP, the
    DDQN agents and the heuristic can go through ONE scoring function.

WHAT IS AND IS NOT IN THE LOG
    * From the simulator's Episode (exact): q_before, v, S, S_exec, IS, q0, S0, q_final, shock_step, true kappa / alpha /
      regime per step.
    * StepLog.belief is a PARTIAL dict {t, regime_probs, alpha_mean, alpha_std, kappa_mean, kappa_std}: the env keeps
      only these pooled fields per step, not the BeliefState objects, so per_regime, ess and loglik_increment are absent.
      interfaces.EpisodeLog.to_dataframe() only needs these keys and works. Belief is None for the oracle / raw_proxy arms.
    * StepLog.obs is None (the env does not keep the proxy dict per step).
    * The agent's action index / multiplier is NOT part of C4 and is not logged; v already encodes what was done.
    * WorldConfig.obs_kind is passed in by the caller (the env does not remember it); default 'gaussian'.

HOW TO SCORE A TRAINED AGENT (the Stage 4 harness's Policy protocol cannot host the DDQN for filtered arms, because
    PolicyInput carries no proxy vector and the observation includes the spread proxy; so run the agent INSIDE the env):
        from ddqn import DDQNAgent; from env_factory import make_env; from belief_mdp_env import EnvConfig
        env   = make_env(EnvConfig(obs_mode="oracle", reward_scale=0.1), stationary_init=True)   # same cfg as training!
        agent = DDQNAgent.load("outputs/stage3c/ckpt/oracle_base1_ts2000_e30k_s0_valsel.npz")
        logs  = run_logs(env, range(900000, 900200), agent, "ddqn_oracle")
        twap  = run_logs(env, range(900000, 900200), None, "twap")
        interfaces.paired_metrics({"twap": twap, "ddqn_oracle": logs})
    The env config used for scoring MUST equal the training config (obs_mode, use_uncertainty, use_regime, n_regimes,
    include_kappa, action_multipliers): the network's input size and action meaning depend on it. DDQNAgent.load rebuilds
    the network from the saved sizes only; it cannot check the rest.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from interfaces import EpisodeLog, StepLog, WorldConfig, validate_episode_log


def world_config(env, obs_kind: str = "gaussian") -> WorldConfig:
    """WorldConfig for the env's simulator, regime model and observation parameters (JSON-serialisable via to_json)."""
    return WorldConfig(sim=asdict(env.p), regime=asdict(env.sim.rp), obs_kind=obs_kind, obs=asdict(env.filt.op))


def env_episode_to_log(env, policy_name: str, world: WorldConfig | None = None) -> EpisodeLog:
    """Convert the env's LAST FINISHED episode (env.episode + env.log) into an interfaces.EpisodeLog."""
    ep, rows = env.episode, env.log
    if ep is None or not env._done:
        raise RuntimeError("the env's episode is not finished; run reset()/step() to the end first")
    N = len(ep.v)
    reg = getattr(ep, "regime", None)
    steps = []
    for t in range(N):
        r = rows[t]
        belief = None
        if "alpha_mean" in r:
            belief = dict(t=t, regime_probs=[float(x) for x in r["regime_probs"]],
                          alpha_mean=float(r["alpha_mean"]), alpha_std=float(r["alpha_std"]),
                          kappa_mean=float(r["kappa_mean"]), kappa_std=float(r["kappa_std"]))
        steps.append(StepLog(t=t, q_before=float(ep.q[t]), v=float(ep.v[t]), S=float(ep.S[t]),
                             S_exec=float(ep.S_exec[t]), obs=None, belief=belief,
                             true_kappa=float(ep.kappa[t]), true_alpha=float(ep.alpha[t]),
                             true_regime=None if reg is None else int(reg[t])))
    return EpisodeLog(policy=policy_name, seed=int(env._seed), world=world or world_config(env), steps=steps,
                      IS=float(ep.IS), q0=float(ep.q[0]), S0=float(ep.S[0]), q_final=float(ep.q[-1]),
                      shock_step=ep.shock_step)


def run_logs(env, seeds, agent=None, name: str | None = None, obs_kind: str = "gaussian") -> list:
    """Run TWAP (agent=None) or the greedy DDQN agent on each world seed and return list[EpisodeLog] in seed order
    (common random numbers: the same seed gives the same world for every policy)."""
    name = name or ("twap" if agent is None else "ddqn")
    world = world_config(env, obs_kind)
    logs = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done = False
        while not done:
            a = env.twap_action if agent is None else agent.greedy(obs)
            obs, _, done, _, _ = env.step(a)
        logs.append(env_episode_to_log(env, name, world))
    return logs


def _selftest():
    from env_factory import make_env
    from belief_mdp_env import EnvConfig
    errs = []
    env = make_env(EnvConfig(obs_mode="oracle", reward_scale=0.1), stationary_init=True)
    seeds = [900_000, 900_001, 900_002]
    logs = run_logs(env, seeds, None, "twap")
    if [l.seed for l in logs] != seeds:
        errs.append("seeds not preserved in order")
    for l in logs:
        e = validate_episode_log(l)
        if e:
            errs.append(f"seed {l.seed}: {e}")
        if l.steps[0].belief is not None:
            errs.append("oracle arm should have belief=None")
    # the log's IS must equal the env's own IS, and the env's exact step-sum
    env.reset(seed=seeds[0])
    done = False
    while not done:
        _, _, done, _, info = env.step(env.twap_action)
    if not np.isclose(logs[0].IS, info["IS"], rtol=1e-12, atol=1e-9):
        errs.append(f"IS mismatch: log {logs[0].IS} vs env {info['IS']}")
    try:
        logs[0].world.to_json()
    except Exception as ex:                                            # noqa: BLE001
        errs.append(f"WorldConfig.to_json failed: {type(ex).__name__}: {ex}")
    # a filtered arm: belief dicts present, to_dataframe works (uses the live IMM particle filter; a few seconds)
    envf = make_env(EnvConfig(obs_mode="filtered", reward_scale=0.1), stationary_init=True)
    lf = run_logs(envf, seeds[:1], None, "twap_filtered")[0]
    ef = validate_episode_log(lf)
    if ef:
        errs.append(f"filtered log: {ef}")
    if lf.steps[0].belief is None or len(lf.steps[0].belief["regime_probs"]) != 2:
        errs.append("filtered arm should carry a 2-regime belief dict")
    else:
        df = lf.to_dataframe()
        if "p_regime_1" not in df.columns:
            errs.append("to_dataframe lost the regime probability columns")
    print("stage3_adapter selftest:", "PASS" if not errs else f"FAIL {errs}")


if __name__ == "__main__":
    _selftest()