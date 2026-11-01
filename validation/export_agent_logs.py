"""
export_agent_logs.py -- run saved checkpoints (and TWAP) through stage3_adapter.run_logs -> interfaces.EpisodeLog
for Person 4. UNTESTED. Assumes run_logs(env, seeds, agent, name) -> list of EpisodeLog and uses agent.greedy(obs).
"""
import argparse, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import dataclasses
import numpy as np
from score_fresh import build_env_for_rec, collect_records, FRESH_BASE


class TwapAgent:
    def __init__(self, a):
        self.a = a

    def greedy(self, obs):
        return self.a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--baseline", type=int, default=1)
    ap.add_argument("--which", default="valsel", choices=["valsel", "last"])
    ap.add_argument("--n", type=int, default=200, help="number of worlds")
    ap.add_argument("--seed-base", type=int, default=FRESH_BASE)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=None, help="only training seeds < this")
    ap.add_argument("--use-cache", action="store_true")
    a = ap.parse_args()

    from ddqn import DDQNAgent
    from stage3_adapter import run_logs
    recs = collect_records(a.tags, a.baseline, a.episodes, a.eval_episodes, a.arms, a.seeds)
    if not recs:
        print("no records with saved checkpoints found"); return
    seeds = list(range(a.seed_base, a.seed_base + a.n))
    ranges = [(a.seed_base, a.n)]
    ckdir = os.path.join(HERE, "outputs", "stage3c", "ckpt")
    logs = {}
    env0 = build_env_for_rec(dict(arm_name="oracle", seed=0, args=recs[0]["args"]), False)
    logs["twap"] = run_logs(env0, seeds, TwapAgent(env0.twap_action), "twap")
    for r in recs:
        name = f"{r['arm_name']}{r['_tag']}_s{r['seed']}_{a.which}"
        env = build_env_for_rec(r, a.use_cache, ranges)
        agent = DDQNAgent.load(os.path.join(ckdir, f"{r['ckpt']}_{a.which}.npz"))
        assert agent.obs_dim == env.obs_dim, "checkpoint/env mismatch"
        logs[name] = run_logs(env, seeds, agent, name)
        print("exported", name, flush=True)

    first = logs["twap"][0]
    print(f"\n{len(logs)} policies x {len(seeds)} worlds; log type: {type(first).__name__}")
    if dataclasses.is_dataclass(first):
        print("EpisodeLog fields:", [f.name for f in dataclasses.fields(first)])
    if hasattr(first, "IS"):
        for k, v in logs.items():
            print(f"  {k:45s} mean IS = {np.mean([x.IS for x in v]):9.1f}")
    od = os.path.join(HERE, "outputs", "stage3c", "logs")
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"logs_{a.which}.pkl")
    with open(p, "wb") as f:
        pickle.dump(logs, f)
    print("wrote", p)


if __name__ == "__main__":
    main()