"""Cached env must reproduce the REAL env exactly for arbitrary actions. Run from EDI/validation."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import belief_cache as bc
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams

bc.CHUNK = 4                    # tiny chunks for the test only
P, START, tmp = 300, 910000, tempfile.mkdtemp()


def roll(env, seed, acts):
    obs, _ = env.reset(seed=seed)
    O, R, done, i = [obs], [], False, 0
    while not done:
        obs, r, done, _, info = env.step(int(acts[i])); i += 1
        O.append(obs); R.append(r)
    return np.array(O), np.array(R), env.get_episode(), env.episode


def same_log(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys(), (ra.keys(), rb.keys())
        for k in ra:
            x, y = ra[k], rb[k]
            if x is None or y is None:
                assert x is y, k
            else:
                assert np.allclose(x, y, rtol=0, atol=1e-9), (k, x, y)


worst = {"obs": 0.0, "IS": 0.0, "S": 0.0}
for filt, K in (("imm", 2), ("single", 1)):
    mk = lambda cfg: make_env(cfg, obs_kind="gaussian", filter_regimes=filt, seed=0,
                              imm_params=IMMParams(n_particles=P), stationary_init=True)
    real0 = mk(EnvConfig(obs_mode="filtered", n_regimes=K, reward_scale=0.1))
    d = bc.cache_dir(tmp, filt, "gaussian", P); os.makedirs(d)
    bc.build_chunk(real0, START, dict(filt=filt, obs_kind="gaussian", n_particles=P), bc.chunk_path(d, START))
    cache = bc.BeliefCache(tmp, filt, "gaussian", P, [(START, 4)])
    assert cache.K == K
    variants = [dict(obs_mode="filtered")]
    if K == 2:
        variants += [dict(obs_mode="filtered", use_uncertainty=False), dict(obs_mode="filtered", use_regime=False),
                     dict(obs_mode="raw_proxy"), dict(obs_mode="oracle")]
    rng = np.random.default_rng(0)
    for var in variants:
        cfg = EnvConfig(n_regimes=K, reward_scale=0.1, **var)
        real, cached = mk(cfg), bc.make_cached_env(cfg, cache)
        for sd in (START, START + 2, START + 3):
            acts = rng.integers(0, real.n_actions, 60)
            Or, Rr, Er, epr = roll(real, sd, acts)
            Oc, Rc, Ec, epc = roll(cached, sd, acts)
            worst["obs"] = max(worst["obs"], float(np.abs(Or - Oc).max()))
            worst["IS"] = max(worst["IS"], abs(Er["IS"] - Ec["IS"]))
            worst["S"] = max(worst["S"], float(np.abs(epr.S - epc.S).max()))
            assert np.allclose(Rr, Rc, atol=1e-9), "rewards differ"
            assert abs(Er["sum_decomp"] - Ec["sum_decomp"]) < 1e-9
            assert np.allclose(epr.q, epc.q, atol=1e-9) and np.allclose(epr.S_exec, epc.S_exec, atol=1e-9)
            same_log(Er["steps"], Ec["steps"])
        print(f"  {filt:6s} {str(var):55s} OK")
try:
    bc.make_cached_env(cfg, cache).reset(seed=1); print("FAIL: unknown seed accepted")
except KeyError:
    print("  unknown seed correctly rejected")
print(f"max |obs diff|={worst['obs']:.2e}  max |IS diff|={worst['IS']:.2e}  max |S diff|={worst['S']:.2e}")
assert worst["obs"] < 1e-9 and worst["IS"] < 1e-6 and worst["S"] < 1e-9
print("cached-env self-test PASSED")