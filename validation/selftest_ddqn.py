"""Stage 3B self-test. Run from EDI/validation:  python selftest_ddqn.py   (UNTESTED until run)"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from ddqn import MLP, DDQNAgent, DDQNConfig

results = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}"); results.append(bool(ok))

# 1. analytic gradients vs central finite differences (targets spread so BOTH Huber zones are hit)
rng = np.random.default_rng(0)
net = MLP([5, 7, 6, 4], rng)
X, act, y = rng.normal(size=(9, 5)), rng.integers(0, 4, 9), rng.normal(size=9) * 3
_, gW, gb = net.loss_and_grads(X, act, y, 1.0)
bad, h = 0, 1e-6
for arrs, gs in ((net.W, gW), (net.b, gb)):
    for arr, g in zip(arrs, gs):
        for _ in range(8):
            ix = tuple(int(rng.integers(0, s)) for s in arr.shape)
            old = arr[ix]
            arr[ix] = old + h; lp = net.loss_and_grads(X, act, y, 1.0)[0]
            arr[ix] = old - h; lm = net.loss_and_grads(X, act, y, 1.0)[0]
            arr[ix] = old
            num = (lp - lm) / (2 * h)
            if abs(num - g[ix]) > 1e-7 + 1e-5 * max(abs(num), abs(g[ix])):
                bad += 1
check("gradient check vs finite differences", bad == 0, f"({bad} mismatches of 8x{2*len(gW)} probes)")

# 2. Double-DQN target = online argmax evaluated by TARGET net (test must be discriminating)
for sd in range(100):
    r_ = np.random.default_rng(sd)
    onl, tgt = MLP([2, 3, 3], r_), MLP([2, 3, 3], r_)
    s2 = r_.normal(size=(6, 2))
    if (np.argmax(onl.forward(s2), 1) != np.argmax(tgt.forward(s2), 1)).any():
        break
rew, d = r_.normal(size=6), np.array([0, 0, 1, 0, 1, 0.0])
yt = DDQNAgent.ddqn_targets(onl, tgt, rew, s2, d, 0.9)
exp = rew + 0.9 * (1 - d) * tgt.forward(s2)[np.arange(6), np.argmax(onl.forward(s2), 1)]
vanilla = rew + 0.9 * (1 - d) * tgt.forward(s2).max(1)
check("double-DQN target uses online argmax / target value", np.allclose(yt, exp) and not np.allclose(yt, vanilla))

# 3. contextual bandit: obs one-hot x in {0,1}, best action = x+1 of 3
ag = DDQNAgent(2, 3, DDQNConfig(hidden=(16,), lr=5e-3, batch_size=64, warmup_steps=0), seed=0)
rb = np.random.default_rng(1)
for _ in range(2000):
    x = int(rb.integers(2)); a_ = int(rb.integers(3))
    ag.buf.add(np.eye(2)[x], a_, 1.0 if a_ == x + 1 else 0.0, np.zeros(2), 1.0)
for _ in range(1500):
    ag.update()
check("learns contextual bandit", ag.greedy(np.eye(2)[0]) == 1 and ag.greedy(np.eye(2)[1]) == 2)

# 4. two-step chain: needs bootstrapping + done masking. s0: a0 -> (r=0, s1) ; a1 -> (r=.5, terminal). s1: a0 -> r=1.
s0, s1, z = np.array([1., 0, 0]), np.array([0, 1., 0]), np.zeros(3)
cfg = DDQNConfig(hidden=(32,), lr=3e-3, batch_size=64, warmup_steps=0, target_sync=100)
ag = DDQNAgent(3, 2, cfg, seed=0)
rc = np.random.default_rng(2)
for _ in range(3000):
    a_ = int(rc.integers(2))
    if rc.random() < 0.5:
        ag.buf.add(s0, a_, 0.0, s1, 0.0) if a_ == 0 else ag.buf.add(s0, 1, 0.5, z, 1.0)
    else:
        ag.buf.add(s1, a_, 1.0 if a_ == 0 else 0.0, z, 1.0)
for _ in range(3000):
    ag.update()
q = ag.q_values(s0)[0]
check("bootstraps through a 2-step chain", ag.greedy(s0) == 0 and abs(q[0] - 1.0) < 0.2 and abs(q[1] - 0.5) < 0.2,
      f"Q(s0)={np.round(q,3)} (want ~[1.0, 0.5])")

# 5. determinism
def build():
    a = DDQNAgent(3, 2, DDQNConfig(hidden=(8,), batch_size=16, warmup_steps=0), seed=5)
    r = np.random.default_rng(9)
    for _ in range(200):
        a.buf.add(r.normal(size=3), int(r.integers(2)), float(r.normal()), r.normal(size=3), float(r.random() < .1))
    for _ in range(50):
        a.update()
    return a
A, B = build(), build()
check("same seed -> identical weights", all(np.array_equal(p, q) for p, q in zip(A.online.params(), B.online.params())))

# 6. save/load roundtrip
with tempfile.TemporaryDirectory() as td:
    pth = os.path.join(td, "a.npz"); A.save(pth); L = DDQNAgent.load(pth)
    xx = np.random.default_rng(3).normal(size=(4, 3))
    check("save/load roundtrip", np.allclose(A.q_values(xx), L.q_values(xx)))

# 7. env smoke (real stack, tiny): trains a few episodes, evaluates greedy, inventory fully sold
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
from ddqn_train import train, run_episodes
env = make_env(EnvConfig(obs_mode="filtered", reward_scale=0.1), seed=1,
               imm_params=IMMParams(n_particles=200), stationary_init=True)
agent, log = train(env, DDQNConfig(warmup_steps=100, buffer_size=5000, batch_size=32), 6, seed=0, verbose=False)
res = run_episodes(env, [900000, 900001], agent)
check("env smoke: trains, weights finite, greedy eval liquidates", len(log) == 6 and res["q_left"].max() < 1e-6 * 20000,
      f"(max unsold {res['q_left'].max():.2e})")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)