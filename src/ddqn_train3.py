"""
ddqn_train3.py -- Stage 3C training loop = ddqn_train2.train_v2 (unchanged logic) plus two OPTIONAL additions.
NOT RUN by its author. ddqn.py / ddqn_train.py / ddqn_train2.py are untouched.

1. baseline=True  (TWAP reward baseline, a variance-reduction change -- a HYPOTHESIS, test it)
   Diagnosis from the 3B logs: the achievable gain (~90 $ on ~4850, i.e. ~2%) is tiny relative to the return the
   network must fit, so action gaps are lost and the greedy policy flips/collapses. Here each step's reward has
   the cost TWAP would have paid in that same step subtracted:
       cost_twap_t = alpha_t*v^2 + kappa_t*v*(q0 - (t+1)*v),   v = q0/N
   using the TRUE alpha_t, kappa_t of that step, in the REWARD only (never in the observation). It depends only on
   the exogenous world and on t, not on the agent's actions, so it cannot change which policy is optimal; it only
   re-centres returns around 0 for TWAP. Requires env.cfg.reward_mode == "decomp". twap_baseline_residual() checks
   that a pure-TWAP episode then has total shaped reward ~0.
   NOTE: training returns / 'ret' in the log are now baseline-adjusted; validation and TEST use the plain IS.
2. bonus_beta > 0  (stage3c_bonus.py): optimism bonus added to Q at ACTION SELECTION only (never in the TD target),
   annealed to 0 over the first `bonus_anneal` fraction of training. See stage3c_bonus.py for the caveats
   (the belief is action-independent, so this is a heuristic, not information gathering; beta untuned).
"""
from __future__ import annotations
import numpy as np
from ddqn import DDQNAgent, DDQNConfig
from ddqn_train import epsilon_at, run_episodes, paired, TRAIN_BASE, EVAL_BASE
from ddqn_train2 import snapshot, restore, VAL_BASE
from stage3c_bonus import bonus_vector, beta_at


def _baseline_reward(env, row) -> float:
    """Reward the TWAP schedule would have earned in step row['t'] (added back as r - r_twap)."""
    p = env.p
    v = p.q0 / env.N
    cb = row["true_alpha"] * v * v + row["true_kappa"] * v * (p.q0 - (row["t"] + 1) * v)
    return -1e4 * cb / (p.S0 * p.q0) * env.cfg.reward_scale


def twap_baseline_residual(env, seed: int = 0) -> float:
    """|sum of baseline-adjusted rewards| for a pure TWAP episode. Should be ~1e-9 (assert < 1e-6)."""
    obs, _ = env.reset(seed=seed)
    done, tot = False, 0.0
    while not done:
        obs, r, done, _, _ = env.step(env.twap_action)
        tot += r - _baseline_reward(env, env.log[-1])
    return abs(tot)


def train_v3(env, cfg: DDQNConfig, n_episodes: int, seed: int, *, val_every: int = 200, n_val: int = 60,
             explore_radius=None, bonus_beta: float = 0.0, bonus_anneal: float = 0.8, unc_source: str = "pooled",
             baseline: bool = True, log_every: int = 200, verbose: bool = True):
    if baseline and env.cfg.reward_mode != "decomp":
        raise ValueError("TWAP reward baseline is only defined for reward_mode='decomp'")
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

    if not all(np.all(np.isfinite(p)) for p in agent.online.params()):
        raise FloatingPointError("non-finite network weights after training")
    return agent, log, curve, best