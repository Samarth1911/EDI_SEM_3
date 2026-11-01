"""
diag_agent_vs_ce.py -- Stage 3 revisit, step 1. NO TRAINING, NO ENV. UNTESTED until you run it.
Run from EDI/validation (venv active):
    python diag_agent_vs_ce.py --agent-logs outputs\\stage3c\\logs\\logs_valsel.pkl

Question: in the SAME states, how does each trained agent's trade size compare with the certainty-equivalent (CE) plan
that beat it by 12-86 $ (noise-free)?  Does it respond to alpha / regime at all, is it noisy, or does it ride the grid ends?

Definitions (no assumption about the env's action coding):
  m_agent(t) = v_t / (q_before_t / (N-t))            multiple of remaining-TWAP actually traded, from the logged step
  m_ce(t)    = w_0 (N-t) / sum_s w_s,  w_s = 1/E[alpha_s]   CE multiple given TRUE alpha_t and TRUE regime_t
                (q-independent, so it is valid in the agent's own states; same exact-forecast CE as run_stage4_benchmarks)
Only steps t < N-1 are used (the last step is a forced sale). Worlds come from the TWAP log's truth (paths do not depend on trading).
Reported per arm (mean +/- sd over training seeds where stated):
  slope, corr : OLS of log m_agent on log m_ce.  slope ~1 = tracks CE; ~0 = ignores alpha/regime information.
  mean|dlog|  : mean |log m_agent - log m_ce|   (0 = identical action)
  thirds      : mean multiple in early/mid/late thirds of the horizon, agent vs CE (front/back-loading)
  regime      : mean multiple by TRUE regime, agent vs CE
  ends        : share of steps at the smallest (0.25) / largest (2.0) grid multiplier
"""
import argparse, os, pickle, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
import interfaces as I                                   # needed so pickled EpisodeLogs can be loaded
from simulator import SimParams
from regime_simulator import RegimeParams

ap = argparse.ArgumentParser()
ap.add_argument("--agent-logs", default=os.path.join(HERE, "outputs", "stage3c", "logs", "logs_valsel.pkl"))
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

sp = SimParams(); N, tau = sp.N, sp.tau
rp = RegimeParams.default()
K = int(rp.n_regimes); Pm = np.asarray(rp.P, float).reshape(K, K)
_eff = [rp.effective_cir(sp, r) for r in range(K)]
LAM = np.array([e["lambda_alpha"] for e in _eff], float); TH = np.array([e["theta_alpha"] for e in _eff], float)
GMIN, GMAX = 0.25, 2.0


def forecast(alpha_t, reg_t, n_ahead):
    out = np.empty(n_ahead); pi = np.zeros(K); pi[reg_t] = 1.0; m = pi * alpha_t; out[0] = alpha_t
    for k in range(1, n_ahead):
        pi_next = pi @ Pm
        m = (1.0 - LAM * tau) * (m @ Pm) + LAM * TH * tau * pi_next
        pi = pi_next; out[k] = m.sum()
    return out


def ce_mult(alpha_t, reg_t, t):
    w = 1.0 / np.maximum(forecast(alpha_t, int(reg_t), N - t), 1e-12)
    return float(w[0] * (N - t) / w.sum())


def arm_of(name):
    r = re.sub(r"_s(?:eed)?_?\d+(?:_[A-Za-z]+)?$", "", name)
    return r if r else name


with open(a.agent_logs, "rb") as f:
    logs = pickle.load(f)
tk = next(k for k in logs if "twap" in k.lower())
seeds = [l.seed for l in logs[tk]][:a.n]
M, REG = [], []                                           # per world: CE multiple and true regime for t = 0..N-2
for l in logs[tk][:a.n]:
    st = l.steps
    assert all(s.true_alpha is not None and s.true_regime is not None for s in st), "logs lack truth fields"
    M.append([ce_mult(st[t].true_alpha, st[t].true_regime, t) for t in range(N - 1)])
    REG.append([int(st[t].true_regime) for t in range(N - 1)])
M, REG = np.array(M), np.array(REG)
thirds = [(0, 16), (16, 33), (33, N - 1)]
tt = np.arange(N - 1)
print(f"{len(M)} worlds. CE multiple: mean {M.mean():.3f}  by true regime: " +
      ", ".join(f"r{r}={M[REG == r].mean():.3f}" for r in range(K) if (REG == r).any()) +
      "  by thirds: " + ", ".join(f"{M[:, lo:hi].mean():.3f}" for lo, hi in thirds))


def per_policy(name):
    ls = logs[name][:a.n]
    assert [l.seed for l in ls] == seeds, f"{name}: world seeds differ from TWAP log"
    ma = np.array([[s.v / (s.q_before / (N - t)) if s.q_before > 1e-9 else np.nan for t, s in enumerate(l.steps[:N - 1])]
                   for l in ls])
    return ma


arms = {}
for k in logs:
    if "twap" not in k.lower():
        arms.setdefault(arm_of(k), []).append(k)
print(f"\n{'arm':26s} {'slope':>12s} {'corr':>12s} {'mean|dlog|':>10s} {'thirds agent (CE)':>28s} {'regime0 / regime1 agent':>24s} {'ends lo/hi':>11s}")
for arm, ks in sorted(arms.items()):
    sl, co, dl, pooled = [], [], [], []
    for k in ks:
        ma = per_policy(k); ok = np.isfinite(ma)
        x = np.log(M[ok]); y = np.log(np.maximum(ma[ok], 1e-6))
        sl.append(np.polyfit(x, y, 1)[0]); co.append(np.corrcoef(x, y)[0, 1]); dl.append(np.mean(np.abs(y - x)))
        pooled.append(ma)
    P = np.nanmean(np.array(pooled), axis=0)                 # seed-averaged multiple per world/step
    allm = np.array(pooled)
    th = " ".join(f"{np.nanmean(P[:, lo:hi]):.2f}" for lo, hi in thirds)
    thc = " ".join(f"{M[:, lo:hi].mean():.2f}" for lo, hi in thirds)
    rg = " / ".join(f"{np.nanmean(P[REG == r]):.2f}" for r in range(K))
    lo_s = np.nanmean(allm <= GMIN + 1e-6); hi_s = np.nanmean(allm >= GMAX - 1e-6)
    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else float("nan")
    print(f"{arm:26s} {np.mean(sl):+5.2f}+-{sd(sl):4.2f} {np.mean(co):+5.2f}+-{sd(co):4.2f} {np.mean(dl):10.3f} "
          f"{th + ' (' + thc + ')':>28s} {rg:>24s} {lo_s:5.2f}/{hi_s:4.2f}")

print("\nHow to read: slope near 0 and low corr -> the agent barely uses alpha/regime information (learning problem: "
      "needs a CE prior/feature). slope near 1 but large mean|dlog| -> right idea, noisy action (approximation/variance). "
      "Large 'ends' shares or thirds far from CE -> systematic front/back-loading.")