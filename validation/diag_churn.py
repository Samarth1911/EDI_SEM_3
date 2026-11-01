"""diag_churn.py -- why does LAST differ from VAL-SEL? Oracle arm. UNTESTED.
Run from EDI/validation:  python diag_churn.py --stem oracle_base1_abl --seeds 1,4"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from ddqn import DDQNAgent

ap = argparse.ArgumentParser()
ap.add_argument("--stem", default="oracle_base1_abl")
ap.add_argument("--seeds", default="1,4")
ap.add_argument("--worlds", type=int, default=100)
ap.add_argument("--world0", type=int, default=900000)
ap.add_argument("--reward-scale", type=float, default=0.1)
a = ap.parse_args()

ck = os.path.join(HERE, "outputs", "stage3c", "ckpt")
env = make_env(EnvConfig(obs_mode="oracle", reward_scale=a.reward_scale), seed=0, stationary_init=True)
N, mults = env.N, np.array(env.cfg.action_multipliers)
per_dollar = 1e4 * a.reward_scale / (env.p.S0 * env.p.q0)      # reward units per $
BUCKETS = [(0, 10), (10, 25), (25, 40), (40, N - 1)]
worlds = range(a.world0, a.world0 + a.worlds)


def rollout(agent):
    O, T, R = [], [], []
    for s in worlds:
        obs, _ = env.reset(seed=int(s)); done, t = False, 0
        while not done:
            O.append(obs.copy()); T.append(t)
            obs, _, done, _, _ = env.step(agent.greedy(obs)); t += 1
        R.extend(x["true_regime"] for x in env.get_episode()["steps"])
    return np.array(O), np.array(T), np.array(R)


def gap_dollars(Q):
    s = np.sort(Q, axis=1)
    return (s[:, -1] - s[:, -2]) / per_dollar


def q(pct, x):
    return " ".join(f"{np.percentile(x, p):7.2f}" for p in pct)


for sd in [int(x) for x in a.seeds.split(",")]:
    pa, pb = (os.path.join(ck, f"{a.stem}_s{sd}_{k}.npz") for k in ("valsel", "last"))
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"seed {sd}: checkpoints missing ({pa} / {pb}) -- skipped"); continue
    A, B = DDQNAgent.load(pa), DDQNAgent.load(pb)
    assert A.obs_dim == env.obs_dim == B.obs_dim, "obs_dim mismatch: checkpoint not from this env/arm"
    for who, pol in (("VAL-SEL", A), ("LAST", B)):
        O, T, R = rollout(pol)
        keep = T < N - 1                                   # last step is a forced sale, not a choice
        O, T, R = O[keep], T[keep], R[keep]
        QA, QB = A.q_values(O), B.q_values(O)
        aA, aB = QA.argmax(1), QB.argmax(1)
        dis = aA != aB
        regretB_under_A = (QA.max(1) - QA[np.arange(len(O)), aB]) / per_dollar   # B's choice judged by A's Q
        regretA_under_B = (QB.max(1) - QB[np.arange(len(O)), aA]) / per_dollar
        print(f"\n=== seed {sd}: states visited by the {who} policy ({len(O)} decision states) ===")
        print(f"action disagreement VAL-SEL vs LAST: {dis.mean():.3f}")
        print(f"{'t-bucket':>9s} {'disagree':>8s} {'mult VAL-SEL':>12s} {'mult LAST':>10s}")
        for lo, hi in BUCKETS:
            m = (T >= lo) & (T < hi)
            if m.any():
                print(f"{lo:3d}-{hi-1:<4d} {dis[m].mean():8.3f} {mults[aA[m]].mean():12.3f} {mults[aB[m]].mean():10.3f}")
        for r, nm in ((0, "calm"), (1, "stressed")):
            m = R == r
            if m.any():
                print(f"regime {nm:8s} disagreement {dis[m].mean():.3f}  (n={int(m.sum())})")
        print("top1-top2 Q gap in $ (p10 p50 p90):  VAL-SEL net", q([10, 50, 90], gap_dollars(QA)),
              "| LAST net", q([10, 50, 90], gap_dollars(QB)))
        if dis.any():
            print("on DISAGREEING states, regret in $ of the other net's action (p50 p90): "
                  f"B's pick under A's Q {q([50, 90], regretB_under_A[dis])} | "
                  f"A's pick under B's Q {q([50, 90], regretA_under_B[dis])}")

print("\nREADING: if disagreement is high AND the regrets on disagreeing states are small (a few $) AND the gaps are small,"
      "\n  the nets are splitting near-ties: LR decay / soft target / EMA address that. If disagreement is concentrated in one"
      "\n  t-bucket or regime with LARGE regrets, it is systematic (extrapolation or a bad region); averaging will not fix it.")