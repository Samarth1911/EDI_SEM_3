"""
score_fresh.py -- score saved checkpoints (valsel + last) on FRESH worlds (seeds FRESH_BASE..), never used for
training, validation, test or any configuration choice. UNTESTED. Depends on ddqn_train.run_episodes / paired
(signatures as used in run_stage3c_multiseed.py). Run from EDI/validation.
"""
import argparse, json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np

FRESH_BASE = 1_000_000
_CACHES = {}


def load_file(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def collect_records(tags, base, episodes, eval_n, arms=None, seeds=None):
    recs = []
    for tag in tags:
        p = os.path.join(HERE, "outputs", "stage3c", f"results_base{base}{tag}.jsonl")
        if not os.path.exists(p):
            print(f"WARNING: missing {p}"); continue
        for r in load_file(p):
            if r["episodes"] != episodes or r["eval_episodes"] != eval_n or not r.get("ckpt"):
                continue
            if (arms and r["arm_name"] not in arms) or (seeds is not None and r["seed"] >= seeds):
                continue
            r["_tag"] = tag
            recs.append(r)
    return recs


def _ensure_cache(filt, obs_kind, n_particles, start, count, root):
    import belief_cache as bc
    from belief_mdp_env import EnvConfig
    from env_factory import make_env
    from imm_particle_filter import IMMParams
    d = bc.cache_dir(root, filt, obs_kind, n_particles)
    os.makedirs(d, exist_ok=True)
    K = 2 if filt == "imm" else 1
    env = None
    for k in range(-(-count // bc.CHUNK)):
        s0 = start + k * bc.CHUNK
        path = bc.chunk_path(d, s0)
        if os.path.exists(path):
            continue
        if env is None:
            env = make_env(EnvConfig(obs_mode="filtered", n_regimes=K, reward_scale=0.1), obs_kind=obs_kind,
                           filter_regimes=filt, seed=0, imm_params=IMMParams(n_particles=n_particles),
                           stationary_init=True)
        print(f"  building cache chunk {s0} ({filt}, {obs_kind}) ...", flush=True)
        bc.build_chunk(env, s0, dict(filt=filt, obs_kind=obs_kind, n_particles=n_particles), path)


def build_env_for_rec(rec, use_cache, ranges=None):
    """Env identical to the one the run was trained with (spec + args come from the results record)."""
    from run_stage3c_multiseed import ARMS
    from belief_mdp_env import EnvConfig
    from env_factory import make_env
    from imm_particle_filter import IMMParams
    import belief_cache as bc
    a, spec = rec["args"], ARMS[rec["arm_name"]]
    single = spec.get("filt") == "single"
    cfg = EnvConfig(obs_mode=spec["arm"], use_uncertainty=not spec.get("no_unc", False),
                    use_regime=not spec.get("no_regime", False), n_regimes=1 if single else 2,
                    reward_scale=a["reward_scale"])
    if use_cache and spec["arm"] == "filtered":
        filt = "single" if single else "imm"
        root = os.path.join(HERE, "outputs", "belief_cache")
        for start, count in ranges:
            _ensure_cache(filt, a["obs_kind"], a["imm_particles"], start, count, root)
        key = (filt, a["obs_kind"], a["imm_particles"], tuple(ranges))
        if key not in _CACHES:
            _CACHES[key] = bc.BeliefCache(root, filt, a["obs_kind"], a["imm_particles"], list(ranges))
        return bc.make_cached_env(cfg, _CACHES[key])
    return make_env(cfg, obs_kind=a["obs_kind"], filter_regimes="single" if single else "imm", seed=rec["seed"],
                    imm_params=IMMParams(n_particles=a["imm_particles"]), stationary_init=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=200, help="the eval-episodes value of the training runs")
    ap.add_argument("--baseline", type=int, default=1)
    ap.add_argument("--n-fresh", type=int, default=500)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=None, help="only training seeds < this")
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--out-name", default="fresh_scores")
    a = ap.parse_args()

    from ddqn import DDQNAgent
    from ddqn_train import run_episodes, paired, EVAL_BASE
    assert FRESH_BASE >= EVAL_BASE + 100_000, "fresh worlds must not overlap the test worlds"
    fresh = list(range(FRESH_BASE, FRESH_BASE + a.n_fresh))
    recs = collect_records(a.tags, a.baseline, a.episodes, a.eval_episodes, a.arms, a.seeds)
    if not recs:
        print("no records with saved checkpoints (ckpt) found"); return
    ranges = [(FRESH_BASE, a.n_fresh)]
    twap = run_episodes(build_env_for_rec(dict(arm_name="oracle", seed=0, args=recs[0]["args"]), False), fresh, None)
    ckdir = os.path.join(HERE, "outputs", "stage3c", "ckpt")

    rows = []
    for r in recs:
        env = build_env_for_rec(r, a.use_cache, ranges)
        row = dict(arm=r["arm_name"], tag=r["_tag"], seed=r["seed"])
        for which in ("valsel", "last"):
            agent = DDQNAgent.load(os.path.join(ckdir, f"{r['ckpt']}_{which}.npz"))
            assert agent.obs_dim == env.obs_dim and agent.n_actions == env.n_actions, "checkpoint/env mismatch"
            res = run_episodes(env, fresh, agent)
            nf, full = paired(res, twap, "IS_decomp"), paired(res, twap, "IS")
            row[which] = dict(diff=nf["diff"], se=nf["se"], full=full["diff"],
                              cvar_diff=full["cvar95"] - full["base_cvar95"], max_q_left=float(res["q_left"].max()))
        rows.append(row)
        print(f"{row['arm']:16s} {row['tag']:14s} s{row['seed']}  valsel {row['valsel']['diff']:+8.1f} "
              f"(se {row['valsel']['se']:5.1f})   last {row['last']['diff']:+8.1f} (se {row['last']['se']:5.1f})", flush=True)

    print(f"\n=== FRESH worlds (n={a.n_fresh}, seeds {FRESH_BASE}..): diff vs TWAP in $, negative = better ===")
    print(f"{'arm':16s} {'tag':14s} {'n':>2s} | {'valsel mean':>11s} {'sd':>7s} | {'last mean':>9s} {'sd':>7s} | valsel CVaR95 diff")
    for arm, tag in sorted({(x["arm"], x["tag"]) for x in rows}):
        g = [x for x in rows if x["arm"] == arm and x["tag"] == tag]
        v, l = np.array([x["valsel"]["diff"] for x in g]), np.array([x["last"]["diff"] for x in g])
        sd = lambda z: z.std(ddof=1) if len(z) > 1 else float("nan")
        print(f"{arm:16s} {tag:14s} {len(g):2d} | {v.mean():+11.1f} {sd(v):7.1f} | {l.mean():+9.1f} {sd(l):7.1f} | "
              f"{np.mean([x['valsel']['cvar_diff'] for x in g]):+.0f}")
    p = os.path.join(HERE, "outputs", "stage3c", a.out_name + ".json")
    json.dump(rows, open(p, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()