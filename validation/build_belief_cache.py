"""Build belief-cache chunks with the REAL filter. Resumable (finished chunks are skipped). Run from EDI/validation."""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--filter", required=True, choices=["imm", "single"])
ap.add_argument("--what", nargs="+", required=True, choices=["val", "test", "train"])
ap.add_argument("--train-seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=30000)
ap.add_argument("--n-val", type=int, default=300)
ap.add_argument("--n-test", type=int, default=200)
ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
ap.add_argument("--imm-particles", type=int, default=2000)
ap.add_argument("--root", default=os.path.join(HERE, "outputs", "belief_cache"))
a = ap.parse_args()

from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
from ddqn_train import TRAIN_BASE, EVAL_BASE
from ddqn_train2 import VAL_BASE
import belief_cache as bc

K = 2 if a.filter == "imm" else 1
env = make_env(EnvConfig(obs_mode="filtered", n_regimes=K, reward_scale=0.1), obs_kind=a.obs_kind,
               filter_regimes=a.filter, seed=0, imm_params=IMMParams(n_particles=a.imm_particles), stationary_init=True)
ranges = []
if "val" in a.what:
    ranges.append((VAL_BASE, a.n_val))
if "test" in a.what:
    ranges.append((EVAL_BASE, a.n_test))
if "train" in a.what:
    ranges.append((TRAIN_BASE * (a.train_seed + 1), a.episodes))
d = bc.cache_dir(a.root, a.filter, a.obs_kind, a.imm_particles)
os.makedirs(d, exist_ok=True)
meta = dict(filt=a.filter, obs_kind=a.obs_kind, n_particles=a.imm_particles)
for start, count in ranges:
    for k in range(-(-count // bc.CHUNK)):
        s0 = start + k * bc.CHUNK
        path = bc.chunk_path(d, s0)
        if os.path.exists(path):
            print(f"skip chunk {s0} (exists)"); continue
        t0 = time.time()
        bc.build_chunk(env, s0, meta, path)
        dt = time.time() - t0
        print(f"built chunk {s0}: {dt:.0f}s ({dt / bc.CHUNK:.3f} s/world)", flush=True)
print("done ->", d)