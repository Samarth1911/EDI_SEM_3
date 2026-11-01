"""
ddqn_train2.py -- Stage 3B-fix: training loop with a proper validation protocol.  NOT RUN by its author.

What changes relative to ddqn_train.train (which stays untouched):
  1. GREEDY validation curve. Every `val_every` episodes the greedy policy is evaluated on VALIDATION worlds
     (seeds 800000+i, disjoint from training [100000*(seed+1)+ep] and from the held-out TEST worlds [900000+i])
     and compared paired against TWAP. The old log only showed epsilon-greedy TRAINING returns, which mix
     exploration cost with policy quality and hid whether the greedy policy was converging.
  2. BEST-CHECKPOINT selection on validation (criterion: paired noise-free IS difference vs TWAP, lower is
     better). The test worlds are never used for selection. Selection on a finite validation set is itself
     slightly optimistic on validation; the TEST number is the honest one.
  3. Evaluation at episode 0 (untrained network) so the curve has a reference point.
  4. Optional LOCAL exploration (explore_radius=r): the epsilon-random action is TWAP's action +/- r grid steps
     instead of uniform over the grid. Rationale (a hypothesis, not a finding): uniform-random play over
     multipliers 0.25..3.0 produces states the greedy policy never reaches, and the Q-function is then poorly
     trained where it matters. Off by default (explore_radius=None reproduces uniform exploration).
Everything else (network, Double-DQN target, replay, optimiser) is ddqn.py, unchanged.
"""
from __future__ import annotations
import numpy as np
from ddqn import DDQNAgent, DDQNConfig
from ddqn_train import epsilon_at, run_episodes, paired, TRAIN_BASE, EVAL_BASE

VAL_BASE = 800_000


def snapshot(agent):
    return [p.copy() for p in agent.online.params()]


def restore(agent, snap):
    for p, s in zip(agent.online.params(), snap):
        np.copyto(p, s)
    agent.target.copy_from(agent.online)


def train_v2(env, cfg: DDQNConfig, n_episodes: int, seed: int, *, val_every: int = 200, n_val: int = 60,
             explore_radius=None, log_every: int = 200, verbose: bool = True):
    base = TRAIN_BASE * (seed + 1)
    assert base + n_episodes < VAL_BASE, "train seeds would reach the validation range"
    assert VAL_BASE + n_val <= EVAL_BASE, "validation seeds would reach the test range"
    val_seeds = list(range(VAL_BASE, VAL_BASE + n_val))

    agent = DDQNAgent(env.obs_dim, env.n_actions, cfg, seed)
    val_twap = run_episodes(env, val_seeds, None)
    log, curve, losses, steps = [], [], [], 0
    best = dict(score=np.inf, ep=-1, snap=None)

    def evaluate(ep_done: int):
        res = run_episodes(env, val_seeds, agent)
        pf, pn = paired(res, val_twap, "IS"), paired(res, val_twap, "IS_decomp")
        row = dict(ep=ep_done, val_diff_full=pf["diff"], val_se_full=pf["se"],
                   val_diff_nf=pn["diff"], val_se_nf=pn["se"], max_q_left=float(res["q_left"].max()))
        curve.append(row)
        if row["val_diff_nf"] < best["score"]:
            best.update(score=row["val_diff_nf"], ep=ep_done, snap=snapshot(agent))
        if verbose:
            print(f"  [val @ {ep_done:5d}] greedy - TWAP: full {row['val_diff_full']:+8.1f} (se {row['val_se_full']:5.1f})  "
                  f"noise-free {row['val_diff_nf']:+8.1f} (se {row['val_se_nf']:5.1f})   best so far {best['score']:+.1f} @ {best['ep']}",
                  flush=True)
        return row

    evaluate(0)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base + ep)
        eps, done, ret = epsilon_at(ep, n_episodes, cfg), False, 0.0
        while not done:
            if not np.all(np.isfinite(obs)):
                raise FloatingPointError(f"non-finite observation in episode {ep}")
            if agent.rng.random() < eps:
                if explore_radius is None:
                    a = int(agent.rng.integers(env.n_actions))
                else:
                    a = int(np.clip(env.twap_action + agent.rng.integers(-explore_radius, explore_radius + 1),
                                    0, env.n_actions - 1))
            else:
                a = agent.greedy(obs)
            obs2, r, done, _, info = env.step(a)
            agent.buf.add(obs, a, r, obs2, float(done))
            ret += r
            steps += 1
            if len(agent.buf) >= cfg.warmup_steps and steps % cfg.update_every == 0:
                losses.append(agent.update())
            obs = obs2
        log.append(dict(ep=ep, eps=eps, ret=ret, IS=info["IS"],
                        loss=float(np.mean(losses[-50:])) if losses else None))
        if verbose and ((ep + 1) % log_every == 0 or ep == n_episodes - 1):
            w = log[-log_every:]
            ls = [x["loss"] for x in w if x["loss"] is not None]
            print(f"  ep {ep + 1}/{n_episodes} eps={eps:.2f} train IS(last {len(w)})={np.mean([x['IS'] for x in w]):.0f} "
                  f"loss={np.mean(ls) if ls else float('nan'):.4f}", flush=True)
        if (ep + 1) % val_every == 0 or ep == n_episodes - 1:
            evaluate(ep + 1)

    if not all(np.all(np.isfinite(p)) for p in agent.online.params()):
        raise FloatingPointError("non-finite network weights after training")
    return agent, log, curve, best