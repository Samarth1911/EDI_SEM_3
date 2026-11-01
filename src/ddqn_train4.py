"""
ddqn_train4.py -- Stage 3D: ddqn_train3.train_v3 (same logic) + an OPTIONAL exponential moving average (EMA) of the
online weights.  NOT RUN by its author.  ddqn.py / ddqn_train*.py / stage3c_bonus.py are untouched.

WHY (a hypothesis, not a finding)
    In the 15000- and 30000-episode oracle runs the greedy policy of the ONLINE network jumps from one validation
    checkpoint to the next on FIXED validation worlds (e.g. -62.6 -> +126.6 -> -3.8 $ vs TWAP within 1600 episodes), so
    the FINAL checkpoint is a noisy draw while the validation-selected one is good. The achievable gain (~90 $ per
    episode = 0.45 reward units at reward_scale 0.1, since 1 $ = 1e4/(S0*q0)*0.1 = 0.005 units for S0=10, q0=20000) is
    small next to a TD error of order 0.1 per step (loss ~0.005-0.02 with a 0.5*err^2 Huber core), so argmax flips
    between near-tied actions are plausible. Averaging the weights over recent gradient steps is a standard cheap way
    to damp that. Whether it helps here is empirical.

WHAT CHANGES vs train_v3   (ema_decay > 0)
    after every gradient step:   ema <- decay*ema + (1-decay)*online     (in place, every parameter array)
    * training (behaviour policy, TD targets, target-network sync) still uses the ONLINE network only;
    * validation, best-checkpoint selection (snapshot) and the RETURNED agent use the EMA weights: on return the agent's
      online network is overwritten with the final EMA weights, so agent.greedy / agent.save give the EMA policy.
    ema_decay = 0 -> the same behaviour as train_v3 (use ddqn_train3.train_v3 directly to reproduce old runs bit for bit).
    decay 0.999 averages over ~1000 gradient steps = ~4000 env steps = ~80 episodes (update_every=4, N=50).
    The EMA starts at the initial online weights and does not move before warm-up ends, so the first validation
    readings are of a lagging network and will differ from train_v3's.
Everything else is copied from train_v3: TWAP reward baseline, optional bonus at action selection only, seeds
(train 100000*(seed+1)+ep, validation 800000+i, test 900000+i, disjoint by the asserts).
"""
from __future__ import annotations
import numpy as np
from ddqn import DDQNAgent, DDQNConfig
from ddqn_train import epsilon_at, run_episodes, paired, TRAIN_BASE, EVAL_BASE
from ddqn_train2 import snapshot, VAL_BASE
from ddqn_train3 import _baseline_reward
from stage3c_bonus import bonus_vector, beta_at


def _load_params(agent, arrays):
    """Copy arrays into the ONLINE network only (ddqn_train2.restore would also resync the target network)."""
    for p, s in zip(agent.online.params(), arrays):
        np.copyto(p, s)


def _ema_step(ema, agent, decay: float):
    for e, p in zip(ema, agent.online.params()):
        e *= decay
        e += (1.0 - decay) * p


def train_v4(env, cfg: DDQNConfig, n_episodes: int, seed: int, *, val_every: int = 200, n_val: int = 60,
             explore_radius=None, bonus_beta: float = 0.0, bonus_anneal: float = 0.8, unc_source: str = "pooled",
             baseline: bool = True, ema_decay: float = 0.0, log_every: int = 200, verbose: bool = True):
    if baseline and env.cfg.reward_mode != "decomp":
        raise ValueError("TWAP reward baseline is only defined for reward_mode='decomp'")
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("ema_decay must be in [0, 1)")
    use_ema = ema_decay > 0.0
    base = TRAIN_BASE * (seed + 1)
    assert base + n_episodes < VAL_BASE, "train seeds would reach the validation range"
    assert VAL_BASE + n_val <= EVAL_BASE, "validation seeds would reach the test range"
    val_seeds = list(range(VAL_BASE, VAL_BASE + n_val))

    agent = DDQNAgent(env.obs_dim, env.n_actions, cfg, seed)
    ema = [p.copy() for p in agent.online.params()] if use_ema else None
    val_twap = run_episodes(env, val_seeds, None)
    log, curve, losses, steps = [], [], [], 0
    best = dict(score=np.inf, ep=-1, snap=None)

    def evaluate(ep_done: int):
        live = None
        if use_ema:                                   # evaluate the EMA weights, then put the live weights back
            live = snapshot(agent)
            _load_params(agent, ema)
        res = run_episodes(env, val_seeds, agent)
        pf, pn = paired(res, val_twap, "IS"), paired(res, val_twap, "IS_decomp")
        row = dict(ep=ep_done, val_diff_full=pf["diff"], val_se_full=pf["se"],
                   val_diff_nf=pn["diff"], val_se_nf=pn["se"], max_q_left=float(res["q_left"].max()))
        curve.append(row)
        if row["val_diff_nf"] < best["score"]:
            best.update(score=row["val_diff_nf"], ep=ep_done, snap=snapshot(agent))   # EMA weights when use_ema
        if verbose:
            print(f"  [val @ {ep_done:5d}] greedy - TWAP: full {row['val_diff_full']:+8.1f} (se {row['val_se_full']:5.1f})  "
                  f"noise-free {row['val_diff_nf']:+8.1f} (se {row['val_se_nf']:5.1f})   best so far {best['score']:+.1f} @ {best['ep']}",
                  flush=True)
        if use_ema:
            _load_params(agent, live)
        return row

    evaluate(0)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base + ep)
        eps, done, ret = epsilon_at(ep, n_episodes, cfg), False, 0.0
        beta = beta_at(ep / n_episodes, bonus_beta, bonus_anneal) if bonus_beta > 0 else 0.0
        while not done:
            if not np.all(np.isfinite(obs)):
                raise FloatingPointError(f"non-finite observation in episode {ep}")
            if agent.rng.random() < eps:
                if explore_radius is None:
                    a = int(agent.rng.integers(env.n_actions))
                else:
                    a = int(np.clip(env.twap_action + agent.rng.integers(-explore_radius, explore_radius + 1),
                                    0, env.n_actions - 1))
            elif beta > 0:
                a = int(np.argmax(agent.q_values(obs)[0] + beta * bonus_vector(env, unc_source)))
            else:
                a = agent.greedy(obs)
            obs2, r, done, _, info = env.step(a)
            if baseline:
                r = r - _baseline_reward(env, env.log[-1])
            agent.buf.add(obs, a, r, obs2, float(done))
            ret += r
            steps += 1
            if len(agent.buf) >= cfg.warmup_steps and steps % cfg.update_every == 0:
                losses.append(agent.update())
                if use_ema:
                    _ema_step(ema, agent, ema_decay)
            obs = obs2
        log.append(dict(ep=ep, eps=eps, ret=ret, IS=info["IS"],
                        loss=float(np.mean(losses[-50:])) if losses else None))
        if verbose and ((ep + 1) % log_every == 0 or ep == n_episodes - 1):
            w = log[-log_every:]
            ls = [x["loss"] for x in w if x["loss"] is not None]
            print(f"  ep {ep + 1}/{n_episodes} eps={eps:.2f} beta={beta:.2f} train IS(last {len(w)})="
                  f"{np.mean([x['IS'] for x in w]):.0f} loss={np.mean(ls) if ls else float('nan'):.4f}", flush=True)
        if (ep + 1) % val_every == 0 or ep == n_episodes - 1:
            evaluate(ep + 1)

    if use_ema:                                       # the returned agent carries the EMA policy
        _load_params(agent, ema)
    if not all(np.all(np.isfinite(p)) for p in agent.online.params()):
        raise FloatingPointError("non-finite network weights after training")
    return agent, log, curve, best