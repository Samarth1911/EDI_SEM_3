import argparse, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np
from score_fresh import build_env_for_rec, collect_records, FRESH_BASE
from ddqn import DDQNAgent
from stage3_adapter import run_logs

class Ensemble:
    def __init__(self, ags): self.ags = ags
    def greedy(self, obs):
        return int(np.argmax(np.mean([g.q_values(obs)[0] for g in self.ags], axis=0)))

class TwapAgent:
    def __init__(self, a): self.a = a
    def greedy(self, obs): return self.a

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--episodes", type=int, default=30000)
ap.add_argument("--eval-episodes", type=int, default=200)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed-base", type=int, default=FRESH_BASE)
ap.add_argument("--arms", nargs="*", default=None)
ap.add_argument("--out", default=None, help="default: logs_ensemble_valsel.pkl (gaussian) / logs_ensemble_valsel_spread_hawkes.pkl")
ap.add_argument("--seeds", type=int, default=5, help="members per ensemble; every arm must have exactly this many records")
a = ap.parse_args()
recs = collect_records(a.tags, 1, a.episodes, a.eval_episodes, a.arms, None)
assert recs, "no records with checkpoints"
kinds = {r["args"]["obs_kind"] for r in recs}
assert len(kinds) == 1, f"records mix observation kinds {kinds}: export Gaussian and Hawkes tags separately"
obs_kind = kinds.pop()
out_name = a.out or ("logs_ensemble_valsel.pkl" if obs_kind == "gaussian" else f"logs_ensemble_valsel_{obs_kind}.pkl")
seeds = list(range(a.seed_base, a.seed_base + a.n))
ckdir = os.path.join(HERE, "outputs", "stage3c", "ckpt")
by_arm = {}
for r in recs:
    by_arm.setdefault(r["arm_name"], []).append(r)
env0 = build_env_for_rec(dict(arm_name="oracle", seed=0, args=recs[0]["args"]), False)
logs = {"twap": run_logs(env0, seeds, TwapAgent(env0.twap_action), "twap", obs_kind=recs[0]["args"]["obs_kind"])}
for arm, rs in by_arm.items():
    rs = sorted(rs, key=lambda r: r["seed"])
    assert len(rs) == a.seeds, f"{arm}: {len(rs)} members, expected {a.seeds} (train the missing seeds first)"
    env = build_env_for_rec(rs[0], False)          # real env: world_config needs env.sim.rp / env.filt.op
    ags = [DDQNAgent.load(os.path.join(ckdir, f"{r['ckpt']}_valsel.npz")) for r in rs]
    assert all(g.obs_dim == env.obs_dim for g in ags), f"{arm}: checkpoint/env mismatch"
    name = f"ens_{arm}"
    logs[name] = run_logs(env, seeds, Ensemble(ags), name, obs_kind=rs[0]["args"]["obs_kind"])
    print("exported", name, f"({len(ags)} members)", flush=True)
od = os.path.join(HERE, "outputs", "stage3c", "logs"); os.makedirs(od, exist_ok=True)
with open(os.path.join(od, out_name), "wb") as f:
    pickle.dump(logs, f)
print("wrote", os.path.join(od, out_name))