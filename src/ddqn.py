"""
ddqn.py -- Stage 3B (Person 3): Double DQN (van Hasselt et al., 2016) in plain numpy.
PROVISIONAL / UNTESTED until validation/selftest_ddqn.py has been run.
All hyper-parameters are my untuned choices; Macri & Lillo's network/hyper-parameters are unknown to me.
Target: y = r + gamma*(1-done)*Q_target(s', argmax_a Q_online(s',a)); Huber loss; hard target sync;
uniform replay; Adam with global-norm gradient clipping.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DDQNConfig:
    hidden: tuple = (64, 64)
    lr: float = 1e-3
    gamma: float = 1.0            # finite horizon; time-left is in the state
    batch_size: int = 128
    buffer_size: int = 100_000
    warmup_steps: int = 2_000
    update_every: int = 4         # env steps per gradient step
    target_sync: int = 500        # gradient steps between hard target copies
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4   # fraction of episodes over which epsilon decays linearly
    huber_delta: float = 1.0
    grad_clip: float = 10.0


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.W[-1] *= 0.1

    def params(self):
        return self.W + self.b

    def forward(self, X, cache=False):
        hs, h = [X], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            hs.append(h)
        return (h, hs) if cache else h

    def loss_and_grads(self, X, act, y, delta):
        out, hs = self.forward(X, cache=True)
        n, idx, L = len(X), np.arange(len(X)), len(self.W)
        err = out[idx, act] - y
        a = np.abs(err)
        loss = float(np.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta)).mean())
        d = np.zeros_like(out)
        d[idx, act] = np.clip(err, -delta, delta) / n
        gW, gb = [None] * L, [None] * L
        for i in reversed(range(L)):
            gW[i], gb[i] = hs[i].T @ d, d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (hs[i] > 0)
        return loss, gW, gb

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            np.copyto(p, q)


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.k = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads, clip=None):
        if clip is not None:
            n = float(np.sqrt(sum(float((g * g).sum()) for g in grads)))
            if n > clip:
                grads = [g * (clip / n) for g in grads]
        self.k += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.k)
            vh = self.v[i] / (1 - self.b2 ** self.k)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class Replay:
    def __init__(self, cap, obs_dim):
        self.cap, self.n, self.ptr = cap, 0, 0
        self.s, self.s2 = np.zeros((cap, obs_dim)), np.zeros((cap, obs_dim))
        self.a, self.r, self.d = np.zeros(cap, dtype=np.int64), np.zeros(cap), np.zeros(cap)

    def __len__(self):
        return self.n

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr, self.n = (i + 1) % self.cap, min(self.n + 1, self.cap)

    def sample(self, k, rng):
        j = rng.integers(0, self.n, k)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


class DDQNAgent:
    def __init__(self, obs_dim, n_actions, cfg: DDQNConfig, seed: int = 0):
        self.obs_dim, self.n_actions, self.cfg = obs_dim, n_actions, cfg
        self.rng = np.random.default_rng(seed)
        sizes = [obs_dim, *cfg.hidden, n_actions]
        self.online, self.target = MLP(sizes, self.rng), MLP(sizes, self.rng)
        self.target.copy_from(self.online)
        self.opt = Adam(self.online.params(), cfg.lr)
        self.buf = Replay(cfg.buffer_size, obs_dim)
        self.n_updates = 0

    def q_values(self, obs):
        return self.online.forward(np.atleast_2d(np.asarray(obs, dtype=np.float64)))

    def greedy(self, obs) -> int:
        return int(np.argmax(self.q_values(obs)[0]))

    def act(self, obs, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    @staticmethod
    def ddqn_targets(online, target, r, s2, done, gamma):
        a2 = np.argmax(online.forward(s2), axis=1)                       # online SELECTS
        q2 = target.forward(s2)[np.arange(len(r)), a2]                   # target EVALUATES
        return r + gamma * (1.0 - done) * q2

    def update(self) -> float:
        c = self.cfg
        s, a, r, s2, d = self.buf.sample(c.batch_size, self.rng)
        y = self.ddqn_targets(self.online, self.target, r, s2, d, c.gamma)
        loss, gW, gb = self.online.loss_and_grads(s, a, y, c.huber_delta)
        self.opt.step(gW + gb, clip=c.grad_clip)
        self.n_updates += 1
        if self.n_updates % c.target_sync == 0:
            self.target.copy_from(self.online)
        return loss

    def save(self, path):
        sizes = [self.obs_dim, *self.cfg.hidden, self.n_actions]
        np.savez(path, sizes=np.array(sizes),
                 **{f"W{i}": w for i, w in enumerate(self.online.W)},
                 **{f"b{i}": b for i, b in enumerate(self.online.b)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        sizes = [int(x) for x in z["sizes"]]
        ag = cls(sizes[0], sizes[-1], DDQNConfig(hidden=tuple(sizes[1:-1])), seed=0)
        for i in range(len(ag.online.W)):
            ag.online.W[i][...], ag.online.b[i][...] = z[f"W{i}"], z[f"b{i}"]
        ag.target.copy_from(ag.online)
        return ag