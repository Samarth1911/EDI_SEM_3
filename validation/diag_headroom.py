"""
diag_headroom.py -- how much can ANY policy gain over TWAP in this world?  NOT RUN by its author.
Run from EDI/validation:  python diag_headroom.py --seed-sd 17 --n-seeds 5 --agent-gain 68

Noise-free execution cost of a schedule v (the part of IS a policy can influence):
    C(v) = sum_t alpha_t v_t^2 + sum_{s<t} kappa_s v_s v_t = v' H v,
    H_tt = alpha_t,  H_ts = H_st = kappa_s / 2  (s < t).
The CLAIRVOYANT schedule minimises C(v) given the TRUE alpha/kappa paths (it knows future regime switches),
subject to sum v = q0, v >= 0. No causal policy can beat it on average, so (TWAP - clairvoyant) is an UPPER
BOUND on the achievable gain. It is loose: it is not limited to the multiplier grid and it sees the future.
Pure-diffusion price noise is excluded (zero-mean for any schedule that cannot see prices).
Worlds are drawn like the env's stationary-start worlds (seeds 900000+i); I cannot confirm they are the SAME
worlds your env produced for those seeds, but they are statistically the same process.
A built-in check runs the real simulator with sigma=0 and requires IS == C(TWAP) to 1e-6 relative.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from simulator import SimParams
from regime_simulator import RegimeParams, RegimeSwitchingSimulator


def stationary(rp):
    pi = np.clip(np.asarray(rp.stationary_distribution(), dtype=float), 0.0, None)
    return RegimeParams(specs=rp.specs, P=np.array(rp.P, dtype=float), init_probs=pi / pi.sum())


def world(sp, rp, seed):
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    ep = sim.simulate_episode(velocities=np.full(sp.N, sp.q0 / sp.N), shock_step=None,
                              shock_jump_alpha_mult=3.0, seed=seed)
    return ep


def build_H(alpha, kappa):
    N = len(alpha)
    H = np.diag(np.asarray(alpha, dtype=float))
    for s in range(N - 1):
        H[s + 1:, s] += 0.5 * kappa[s]
        H[s, s + 1:] += 0.5 * kappa[s]
    return H


def clairvoyant(H, q0):
    """min v'Hv s.t. sum v = q0, v >= 0 by a one-at-a-time active-set heuristic; KKT violation is returned."""
    N = len(H)
    free = np.ones(N, dtype=bool)
    for _ in range(N):
        idx = np.where(free)[0]
        m = len(idx)
        A = np.zeros((m + 1, m + 1))
        A[:m, :m] = 2 * H[np.ix_(idx, idx)]
        A[:m, m] = 1.0
        A[m, :m] = 1.0
        rhs = np.zeros(m + 1)
        rhs[m] = q0
        sol = np.linalg.solve(A, rhs)
        vf, lam = sol[:m], sol[m]
        if vf.min() >= -1e-9:
            break
        free[idx[np.argmin(vf)]] = False
    v = np.zeros(N)
    v[idx] = np.maximum(vf, 0.0)
    g = 2 * H @ v
    off = ~free
    viol = float(np.max(np.maximum(0.0, -(g[off] + lam)))) / max(abs(lam), 1e-30) if off.any() else 0.0
    return v, viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=300)
    ap.add_argument("--world0", type=int, default=900000)
    ap.add_argument("--agent-gain", type=float, default=None, help="e.g. 68 = oracle agent beat TWAP by $68")
    ap.add_argument("--seed-sd", type=float, default=17.0, help="across-training-seed sd of the agent's gain, $")
    ap.add_argument("--n-seeds", type=int, default=5)
    a = ap.parse_args()

    sp, rp = SimParams(), stationary(RegimeParams.default())
    N, q0 = sp.N, sp.q0

    # ---- built-in exactness check of the cost formula against the real simulator (sigma = 0) ----
    sp0 = SimParams(sigma=0.0)
    ep0 = world(sp0, rp, 5)
    Hc = build_H(ep0.alpha[:N], ep0.kappa[:N])
    vt = np.full(N, q0 / N)
    rel = abs(ep0.IS - vt @ Hc @ vt) / abs(ep0.IS)
    print(f"[{'PASS' if rel < 1e-6 else 'FAIL'}] cost formula vs real simulator (sigma=0, TWAP): "
          f"IS={ep0.IS:.4f}  v'Hv={vt @ Hc @ vt:.4f}  rel.err={rel:.1e}")
    if rel >= 1e-6:
        sys.exit("cost formula does not match the simulator: do not trust the numbers below")

    c_tw, c_cl, viol, frac_stress = [], [], [], []
    for i in range(a.worlds):
        ep = world(sp, rp, a.world0 + i)
        H = build_H(ep.alpha[:N], ep.kappa[:N])
        v, vi = clairvoyant(H, q0)
        c_tw.append(vt @ H @ vt)
        c_cl.append(v @ H @ v)
        viol.append(vi)
        frac_stress.append(float(np.mean(np.asarray(ep.regime[:N]) == 1)))
    c_tw, c_cl = np.array(c_tw), np.array(c_cl)
    gain = c_tw - c_cl
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    print(f"\n{a.worlds} worlds | share of steps in stress {np.mean(frac_stress):.2f}")
    print(f"TWAP noise-free cost        {c_tw.mean():9.1f}")
    print(f"clairvoyant optimal cost    {c_cl.mean():9.1f}")
    print(f"headroom (upper bound)      {gain.mean():9.1f}  (se {se:.1f})  = {100 * gain.mean() / c_tw.mean():.1f}% of TWAP cost")
    print(f"clairvoyant gain: p10/p50/p90 per world = {np.percentile(gain, 10):.0f} / {np.percentile(gain, 50):.0f} / {np.percentile(gain, 90):.0f}")
    print(f"active-set KKT violation (relative): max {max(viol):.2e}  ( > 1e-3 means the heuristic solver is unreliable)")
    if a.agent_gain is not None:
        print(f"oracle-agent gain ${a.agent_gain:.0f} = {100 * a.agent_gain / gain.mean():.0f}% of this (loose) ceiling")

    print(f"\n== detectability with training-seed sd ${a.seed_sd:.0f} ==")
    for n in (5, 10, 20, 40):
        mde = 2.8 * a.seed_sd * np.sqrt(2.0 / n)
        print(f"   n={n:3d} seeds/arm: minimum detectable difference between two arms ~ ${mde:.0f} (80% power, 5% two-sided)")
    for d in (30, 20, 10):
        print(f"   to detect a ${d} difference you need ~{int(np.ceil(2 * (2.8 * a.seed_sd / d) ** 2))} seeds per arm")
    print("   (rough normal-approximation; assumes unpaired arms with equal sd)")


if __name__ == "__main__":
    main()