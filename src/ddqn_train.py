"""ddqn_train.py -- Stage 3B: training loop + evaluation harness for BeliefMDPEnv. UNTESTED until run.
Train world seeds: 100000*(agent_seed+1)+episode. Eval world seeds: 900000+i. Disjoint by construction."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from ddqn import DDQNAgent, DDQNConfig

TRAIN_BASE, EVAL_BASE = 100_000, 900_000


def epsilon_at(ep, n_ep, cfg: DDQNConfig) -> float:
    span = max(1.0, cfg.eps_decay_frac * n_ep)
    return cfg.eps_start + min(1.0, ep / span) * (cfg.eps_end - cfg.eps_start)


def train(env, cfg: DDQNConfig, n_episodes: int, seed: int, log_every: int = 50, verbose: bool = True):
    base = TRAIN_BASE * (seed + 1)
    assert base + n_episodes < EVAL_BASE, "train/eval seed ranges would overlap"
    agent = DDQNAgent(env.obs_dim, env.n_actions, cfg, seed)
    log, losses, steps = [], [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base + ep)
        eps, done, ret = epsilon_at(ep, n_episodes, cfg), False, 0.0
        while not done:
            if not np.all(np.isfinite(obs)):
                raise FloatingPointError(f"non-finite observation in episode {ep}")
            a = agent.act(obs, eps)
            obs2, r, done, _, info = env.step(a)
            agent.buf.add(obs, a, r, obs2, float(done))
            ret += r
            steps += 1
            if len(agent.buf) >= cfg.warmup_steps and steps % cfg.update_every == 0:
                losses.append(agent.update())
            obs = obs2
        e = env.get_episode()
        log.append(dict(ep=ep, eps=eps, ret=ret, IS=info["IS"], sum_decomp=e["sum_decomp"],
                        loss=float(np.mean(losses[-50:])) if losses else None))
        if verbose and ((ep + 1) % log_every == 0 or ep == n_episodes - 1):
            w = log[-log_every:]
            ls = [x["loss"] for x in w if x["loss"] is not None]
            print(f"  ep {ep+1}/{n_episodes} eps={eps:.2f} mean IS(last {len(w)})={np.mean([x['IS'] for x in w]):.0f} "
                  f"mean ret={np.mean([x['ret'] for x in w]):.1f} loss={np.mean(ls) if ls else float('nan'):.4f}",
                  flush=True)
    if not all(np.all(np.isfinite(p)) for p in agent.online.params()):
        raise FloatingPointError("non-finite network weights after training")
    return agent, log


def run_episodes(env, seeds, agent=None) -> dict:
    """agent=None => TWAP (action multiplier 1.0). Greedy policy otherwise (no exploration)."""
    mults = env.cfg.action_multipliers
    IS, dec, qleft, by = [], [], [], defaultdict(list)
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done = False
        while not done:
            a = env.twap_action if agent is None else agent.greedy(obs)
            obs, _, done, _, _ = env.step(a)
        e = env.get_episode()
        st = e["steps"]
        IS.append(e["IS"]); dec.append(e["sum_decomp"])
        qleft.append(e["q0"] - sum(x["v"] for x in st))
        for x in st[:-1]:                                   # last step is a forced sale, not a choice
            if x["true_regime"] is not None:
                by[x["true_regime"]].append(mults[x["action"]])
    return dict(IS=np.array(IS), IS_decomp=np.array(dec), q_left=np.array(qleft),
                mult_by_regime={int(k): (float(np.mean(v)), len(v)) for k, v in by.items()})


def cvar95(x) -> float:
    k = max(1, int(np.ceil(0.05 * len(x))))
    return float(np.sort(x)[-k:].mean())


def paired(res: dict, base: dict, key: str = "IS") -> dict:
    d = res[key] - base[key]
    return dict(mean=float(res[key].mean()), base_mean=float(base[key].mean()), diff=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))), cvar95=cvar95(res[key]),
                base_cvar95=cvar95(base[key]), n=int(len(d)))