"""
check_independence_hawkes.py -- the Gaussian-arm checks repeated on the STATEFUL spread-Hawkes observation model.
1. reset(seed) is deterministic;  2. world, proxies and beliefs do not depend on our actions;
3. cached env == real env (obs, IS, S) for random actions.  UNTESTED. Run from EDI/validation.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import belief_cache as bc
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams

KIND, P, TOL = "spread_hawkes", 300, 1e-12
mk = lambda cfg: make_env(cfg, obs_kind=KIND, filter_regimes="imm", seed=0,
                          imm_params=IMMParams(n_particles=P), stationary_init=True)
cfg0 = EnvConfig(obs_mode="filtered", reward_scale=0.1)
env = mk(cfg0)
FIELDS = ["true_alpha", "true_kappa", "true_regime", "alpha_mean", "alpha_std", "kappa_mean", "kappa_std"]


def roll(e, seed, acts):
    e.reset(seed=seed)
    px, done, i = [dict(e._px)], False, 0
    while not done:
        _, _, done, _, _ = e.step(int(acts[i])); i += 1
        if not done:
            px.append(dict(e._px))
    return e.get_episode()["steps"], px


def maxdiff(A, B):
    (sa, pa), (sb, pb) = A, B
    d = {k: max(abs(x[k] - y[k]) for x, y in zip(sa, sb)) for k in FIELDS}
    d["regime_probs"] = max(abs(u - w) for x, y in zip(sa, sb) for u, w in zip(x["regime_probs"], y["regime_probs"]))
    for k in pa[0]:
        d["px:" + k] = max(abs(float(x[k]) - float(y[k])) for x, y in zip(pa, pb))
    return d


ok = True
twap, hi = [env.twap_action] * 60, [env.n_actions - 1] * 60
for sd in (900001, 900002, 900003):
    r1, r2, r3 = roll(env, sd, twap), roll(env, sd, twap), roll(env, sd, hi)
    det, ind = maxdiff(r1, r2), maxdiff(r1, r3)
    print(f"world {sd}: actions differ: {max(abs(x['v'] - y['v']) for x, y in zip(r1[0], r3[0])) > 0}")
    print(f"   determinism  max|diff| = {max(det.values()):.2e}")
    print(f"   independence max|diff| = {max(ind.values()):.2e}   worst field: {max(ind, key=ind.get)}")
    ok &= max(det.values()) < TOL and max(ind.values()) < TOL
print("independence / determinism:", "PASS" if ok else "FAIL -> do NOT cache this arm")

if ok:
    bc.CHUNK = 4
    START, tmp = 910000, tempfile.mkdtemp()
    d = bc.cache_dir(tmp, "imm", KIND, P); os.makedirs(d)
    bc.build_chunk(env, START, dict(filt="imm", obs_kind=KIND, n_particles=P), bc.chunk_path(d, START))
    cache = bc.BeliefCache(tmp, "imm", KIND, P, [(START, 4)])
    real, cached = mk(cfg0), bc.make_cached_env(cfg0, cache)
    rng, worst = np.random.default_rng(0), 0.0
    for sd in (START, START + 2, START + 3):
        acts = rng.integers(0, real.n_actions, 60)
        for e in (real, cached):
            obs, _ = e.reset(seed=sd); O, done, i = [obs], False, 0
            while not done:
                obs, r, done, _, _ = e.step(int(acts[i])); i += 1; O.append(obs)
            (real_res if e is real else cached_res).__setitem__(0, 0) if False else None
            if e is real:
                Or, ISr, Sr = np.array(O), e.get_episode()["IS"], e.episode.S.copy()
            else:
                Oc, ISc, Sc = np.array(O), e.get_episode()["IS"], e.episode.S.copy()
        worst = max(worst, float(np.abs(Or - Oc).max()), abs(ISr - ISc), float(np.abs(Sr - Sc).max()))
    print(f"cached vs real (spread_hawkes): max diff over obs/IS/S = {worst:.2e}")
    print("cached-env exactness:", "PASS" if worst < 1e-9 else "FAIL")