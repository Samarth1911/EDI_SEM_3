"""With sigma=0 the price-noise term vanishes, so IS must equal sum(decomp) to float tolerance,
for ANY action sequence. Run from EDI/validation. UNTESTED until run."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from belief_mdp_env import EnvConfig
from env_factory import make_env
from simulator import SimParams

env = make_env(EnvConfig(obs_mode="oracle"), seed=1, sim_params=SimParams(sigma=0.0))
ok_all = True
for pattern in ("twap", "cycle", "front-loaded"):
    obs, _ = env.reset(seed=5); t, done = 0, False
    while not done:
        a = env.twap_action if pattern == "twap" else (t % env.n_actions if pattern == "cycle"
                                                       else (env.n_actions - 1 if t < 10 else 0))
        obs, r, done, _, info = env.step(a); t += 1
    ep = env.get_episode()
    rel = abs(ep["IS"] - ep["sum_decomp"]) / abs(ep["IS"])
    ok = rel < 1e-9
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {pattern:13s} IS={ep['IS']:.4f} sum_decomp={ep['sum_decomp']:.4f} rel.err={rel:.1e}")
sys.exit(0 if ok_all else 1)