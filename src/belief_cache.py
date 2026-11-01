"""
belief_cache.py -- Stage 3D (Person 3). UNTESTED until selftest_cached_env.py passes.
Beliefs, proxies and the true (alpha, kappa, regime) path do not depend on our trades (checked on one
world, Gaussian arm); only the price S does. So per world seed we run the REAL env once with TWAP, store
everything, and replay it for any action sequence. S is rebuilt as
    S[t+1] = S[t] - kappa_t * v_t + noise_t,  noise_t = S_twap[t+1] - S_twap[t] + kappa_t * v_twap_t.
Assumptions: interfaces exports BeliefState and RegimeBelief; simulator exports Episode(t,S,S_exec,kappa,alpha,v,q,IS,shock_step,shock_info).
"""
from __future__ import annotations
import json, os
import numpy as np
from simulator import SimParams, Episode
from interfaces import BeliefState, RegimeBelief
from belief_mdp_env import BeliefMDPEnv

CHUNK = 1000
_INT = ("st_regime", "ep_regime")


def cache_dir(root, filt, obs_kind, n_particles):
    return os.path.join(root, f"{filt}_{obs_kind}_p{n_particles}")


def chunk_path(d, start):
    return os.path.join(d, f"chunk_{start}.npz")


def _pack(b):
    K = len(b.regime_probs)
    if len(b.per_regime) != K:
        raise ValueError("belief.per_regime must have K entries to be cached")
    v = [b.kappa_mean, b.kappa_std, b.alpha_mean, b.alpha_std, b.ess,
         np.nan if b.loglik_increment is None else b.loglik_increment,
         np.nan if b.n_particles is None else b.n_particles]
    for r in b.per_regime:
        v += [r.prob, r.kappa_mean, r.kappa_std, r.alpha_mean, r.alpha_std, r.ess]
    return v


def _unpack(vec, t):
    K = (len(vec) - 7) // 6
    per = []
    for j in range(K):
        x = vec[7 + 6 * j: 13 + 6 * j]
        per.append(RegimeBelief(regime=j, prob=float(x[0]), kappa_mean=float(x[1]), kappa_std=float(x[2]),
                                alpha_mean=float(x[3]), alpha_std=float(x[4]), ess=float(x[5])))
    return BeliefState(t=t, regime_probs=[r.prob for r in per], kappa_mean=float(vec[0]), kappa_std=float(vec[1]),
                       alpha_mean=float(vec[2]), alpha_std=float(vec[3]), ess=float(vec[4]), per_regime=per,
                       n_particles=None if np.isnan(vec[6]) else int(vec[6]),
                       loglik_increment=None if np.isnan(vec[5]) else float(vec[5]))


def build_chunk(env, start, meta, path):
    """Run the REAL filtered env with TWAP on seeds start..start+CHUNK-1 and save one chunk."""
    N = env.N
    out = {k: [] for k in ("st_alpha", "st_kappa", "st_regime", "px", "bel",
                           "S", "v", "ep_alpha", "ep_kappa", "ep_t", "ep_regime")}
    for sd in range(start, start + CHUNK):
        env.reset(seed=sd)
        sa, sk, sr, px, bel = [], [], [], [], []
        for t in range(N):
            s = env._s
            assert int(s["t"]) == t
            sa.append(s["alpha"]); sk.append(s["kappa"]); sr.append(int(s["regime"]) if "regime" in s else -1)
            px.append([env._px["spread"], env._px["volume"], env._px["realvol"]])
            bel.append(_pack(env.belief))
            _, _, done, _, _ = env.step(env.twap_action)
        assert done
        ep = env.episode
        for k, val in (("st_alpha", sa), ("st_kappa", sk), ("st_regime", sr), ("px", px), ("bel", bel),
                       ("S", ep.S), ("v", ep.v), ("ep_alpha", ep.alpha), ("ep_kappa", ep.kappa), ("ep_t", ep.t)):
            out[k].append(np.asarray(val, dtype=np.float64 if k not in _INT else np.int64))
        reg = getattr(ep, "regime", None)
        out["ep_regime"].append(np.asarray(reg, dtype=np.int64) if reg is not None else -np.ones(N + 1, dtype=np.int64))
    arrs = {k: np.stack(v) for k, v in out.items()}
    tmp = path + ".tmp.npz"
    np.savez(tmp, seeds=np.arange(start, start + CHUNK), meta=np.array(json.dumps(meta)), **arrs)
    os.replace(tmp, path)


class BeliefCache:
    def __init__(self, root, filt, obs_kind, n_particles, ranges):
        d = cache_dir(root, filt, obs_kind, n_particles)
        self.chunks, missing = {}, []
        for start, count in ranges:
            for k in range(-(-count // CHUNK)):
                s0 = start + k * CHUNK
                p = chunk_path(d, s0)
                if not os.path.exists(p):
                    missing.append(p); continue
                z = np.load(p)
                meta = json.loads(str(z["meta"]))
                assert meta["filt"] == filt and meta["obs_kind"] == obs_kind and meta["n_particles"] == n_particles, meta
                self.chunks[s0] = {k2: z[k2] for k2 in z.files if k2 != "meta"}
        if missing:
            raise FileNotFoundError("belief cache chunks missing (run build_belief_cache.py):\n  " + "\n  ".join(missing))
        self.starts = sorted(self.chunks)
        first = self.chunks[self.starts[0]]
        self.K = (first["bel"].shape[2] - 7) // 6

    def get(self, seed):
        s0 = max((s for s in self.starts if s <= seed), default=None)
        if s0 is None or seed - s0 >= CHUNK:
            raise KeyError(f"seed {seed} is not in the loaded cache ranges")
        j, c = seed - s0, self.chunks[s0]
        return {k: c[k][j] for k in c if k != "seeds"}


class _Shared:
    row = None
    i = 0
    last = 0


class _ReplayGen:
    """Mimics MarketSimulator.run_stepwise's pull protocol on cached data."""
    def __init__(self, p, row):
        self.p, self.row, self.t = p, row, 0
        self.S, self.q = float(row["S"][0]), float(p.q0)
        self.Ss, self.Sx, self.vs, self.qs = [self.S], [], [], [self.q]

    def _state(self):
        r, t = self.row, self.t
        s = {"t": t, "S": self.S, "kappa": float(r["st_kappa"][t]), "alpha": float(r["st_alpha"][t]),
             "q": self.q, "N": self.p.N, "shock_step": None}
        if int(r["st_regime"][t]) >= 0:
            s["regime"] = int(r["st_regime"][t])
        return s

    def __next__(self):
        return self._state()

    def send(self, v):
        p, r, t = self.p, self.row, self.t
        v = float(np.clip(0.0 if v is None else v, 0.0, self.q))
        kap, alp = float(r["st_kappa"][t]), float(r["st_alpha"][t])
        self.vs.append(v)
        self.Sx.append(self.S - alp * v)
        noise = r["S"][t + 1] - r["S"][t] + kap * r["v"][t]
        self.S = self.S - kap * v + float(noise)
        self.q = self.q - v
        self.Ss.append(self.S); self.qs.append(self.q)
        self.t += 1
        if self.t >= p.N:
            v_a, Sx_a = np.array(self.vs), np.array(self.Sx)
            IS = p.S0 * p.q0 - float(np.sum(Sx_a * v_a))
            ep = Episode(t=r["ep_t"].copy(), S=np.array(self.Ss), S_exec=Sx_a, kappa=r["ep_kappa"].copy(),
                         alpha=r["ep_alpha"].copy(), v=v_a, q=np.array(self.qs), IS=IS,
                         shock_step=None, shock_info=None)
            if int(r["ep_regime"][0]) >= 0:
                try:
                    ep.regime = r["ep_regime"].copy()
                except Exception:
                    pass
            raise StopIteration(ep)
        return self._state()


class _FakeSim:
    def __init__(self, cache, shared):
        self.p, self.cache, self.sh = SimParams(), cache, shared

    def run_stepwise(self, seed=None, **kw):
        if kw:
            raise ValueError(f"cached env does not support sim_kwargs {kw}")
        self.sh.row, self.sh.i = self.cache.get(int(seed)), 0
        return _ReplayGen(self.p, self.sh.row)


class _FakeOM:
    def __init__(self, shared):
        self.sh = shared

    def reset(self, seed=None):
        self.sh.i = 0

    def observe_step(self, kappa, alpha, regime=0, rng=None):
        i = self.sh.i
        x = self.sh.row["px"][i]
        self.sh.i, self.sh.last = i + 1, i
        return {"spread": float(x[0]), "volume": float(x[1]), "realvol": float(x[2])}


class _FakeFilt:
    def __init__(self, shared):
        self.sh = shared

    def reset(self, seed=None):
        pass

    def update(self, obs):
        return _unpack(self.sh.row["bel"][self.sh.last], self.sh.last)


def make_cached_env(cfg, cache):
    sh = _Shared()
    return BeliefMDPEnv(cfg, _FakeSim(cache, sh), _FakeOM(sh), _FakeFilt(sh))