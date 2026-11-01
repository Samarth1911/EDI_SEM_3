"""
check_grid_env.py -- verify a custom action grid works in BeliefMDPEnv BEFORE any training. UNTESTED until you run it.
Run from EDI/validation (venv active):
    python check_grid_env.py --grid 0.658,0.756,0.87,1.0,1.15,1.323,1.521

I have NOT seen belief_mdp_env.py. This relies on the Stage 3 handoff, which lists `action_multipliers` as an EnvConfig
field and says env.twap_action is the 1.0 action. If a check below fails, do not train: send me belief_mdp_env.py.
Checks (oracle observation, cheap):
  1. env.n_actions == len(grid) and cfg.action_multipliers == grid; the TWAP action's multiplier is exactly 1.0.
  2. TWAP on the custom-grid env reproduces the default-grid env's TWAP implementation shortfall bit for bit (5 worlds).
  3. every constant-multiplier policy fully liquidates the inventory.
  4. obs_dim is unchanged by the grid.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams

ap = argparse.ArgumentParser()
ap.add_argument("--grid", required=True)
ap.add_argument("--seed0", type=int, default=900_000)
a = ap.parse_args()
grid = tuple(float(x) for x in a.grid.split(","))
ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def build(**kw):
    return make_env(EnvConfig(obs_mode="oracle", use_uncertainty=True, use_regime=True, n_regimes=2,
                              reward_scale=0.1, **kw),
                    obs_kind="gaussian", filter_regimes="imm", seed=0,
                    imm_params=IMMParams(n_particles=200), stationary_init=True)


def run(env, seed, action):
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, done, _, _ = env.step(action)
    return env.get_episode()


env0, env1 = build(), build(action_multipliers=grid)
check("1a n_actions == len(grid)", env1.n_actions == len(grid), f"({env1.n_actions} vs {len(grid)})")
check("1b cfg.action_multipliers == grid", tuple(env1.cfg.action_multipliers) == grid)
check("1c twap_action multiplier is exactly 1.0", abs(env1.cfg.action_multipliers[env1.twap_action] - 1.0) < 1e-12)
check("4 obs_dim unchanged", env1.obs_dim == env0.obs_dim, f"({env1.obs_dim} vs {env0.obs_dim})")
d = max(abs(run(env0, s, env0.twap_action)["IS"] - run(env1, s, env1.twap_action)["IS"])
        for s in range(a.seed0, a.seed0 + 5))
check("2 TWAP IS identical across grids", d < 1e-9, f"(max |diff| = {d:.2e})")
worst = 0.0
for act in range(env1.n_actions):
    e = run(env1, a.seed0, act)
    worst = max(worst, abs(e["q0"] - sum(x["v"] for x in e["steps"])) / e["q0"])
check("3 every constant multiplier liquidates fully", worst < 1e-9, f"(max unsold fraction {worst:.2e})")
print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED: do not train; send belief_mdp_env.py")
sys.exit(0 if ok_all else 1)