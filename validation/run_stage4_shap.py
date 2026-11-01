"""
run_stage4_shap.py -- Stage 4B (Person 4): exact Shapley (SHAP) attribution of the IMM-PF's one-step belief update to its three
proxy channels (spread, volume, realvol). NOT RUN by its author. Run from EDI/validation (venv active):
    python run_stage4_shap.py                         (Gaussian observations, 200 worlds)
    python run_stage4_shap.py --obs-kind spread_hawkes

WHAT IS ATTRIBUTED (read before quoting)
  Game: the 3 channels are the players. At step t, starting from the filter's ACTUAL state (its history used all channels),
  v(S) = posterior functional after weighting the propagated particles with the likelihood of ONLY the channels in S.
  An absent channel is marginalised by dropping its likelihood factor: exact under the filter's own assumption that the three
  proxies are conditionally independent Gaussians given a particle. Because the filter reads exactly spread/volume/realvol, enumerating
  all 8 coalitions gives the exact Shapley values (the target Kernel SHAP estimates, with a conditional-expectation value function);
  the `shap` library is not needed for 3 features. v(empty) = the PREDICTED value, so  sum_c phi_c = v(all) - v(empty) = the step's update.
  Functionals:  A = posterior mean of alpha (regime mixture), reported in units of the base long-run mean theta_alpha;
                R = posterior probability of the stressed regime.
  Groups: all steps | true calm | true stressed | first 3 steps after a switch INTO stressed | first 3 steps after a switch INTO calm.
  Intervals are 95% bootstrap CIs over WORLDS.
LIMITS: this describes how THIS FILTER reacts (including its misspecification on the Hawkes arm), not which proxy is causally
  informative in the world, and not which input a trading policy relies on. One-step attribution along the true filter path; it is not a
  cumulative attribution of the whole history. Parameters are the simulator's (uncalibrated). Attribution worlds: seeds 2,000,000+i (unused elsewhere).
CHECKS run first (script stops if any fails): Shapley toy games; full coalition == the filter's real belief at every step; efficiency.
"""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from observation_model import ObsParams
from spread_hawkes import make_observation_model_v2
import imm_particle_filter as ipf
from imm_particle_filter import IMMParticleFilter, IMMParams
import interfaces as I

predicted_proxy_means = getattr(ipf, "predicted_proxy_means", None)
if predicted_proxy_means is None:
    from observation_model import predicted_proxy_means
_log_gauss = ipf._log_gauss
CH = ("spread", "volume", "realvol")
GROUPS = ("all", "calm", "stressed", "into_stressed", "into_calm")


# ------------------------------------------------------------------ Shapley
def shapley3(v):
    """Exact Shapley values for 3 players; v[mask] for mask in 0..7 (bit i = channel i present)."""
    phi = np.zeros(3)
    for i in range(3):
        for mask in range(8):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            phi[i] += math.factorial(s) * math.factorial(2 - s) / 6.0 * (v[mask | (1 << i)] - v[mask])
    return phi


def selftest_shapley():
    ok = True
    add = np.array([sum(c for i, c in enumerate((1.0, 2.0, 3.0)) if m >> i & 1) for m in range(8)])
    ok &= np.allclose(shapley3(add), [1, 2, 3])
    una = np.array([1.0 if m == 7 else 0.0 for m in range(8)])
    ok &= np.allclose(shapley3(una), [1 / 3] * 3)
    maj = np.array([1.0 if bin(m).count("1") >= 2 else 0.0 for m in range(8)])
    ok &= np.allclose(shapley3(maj), [1 / 3] * 3)
    return bool(ok)


# ------------------------------------------------------------------ coalition values from the filter's state
def coalition_values(f, y):
    """Replicates IMMParticleFilter.update's posterior math for every channel subset; does NOT change the filter."""
    op, infl = f.op, f.pp.obs_noise_inflation
    ps, pv, pr = predicted_proxy_means(f.kappa, f.alpha, op)
    comp = (_log_gauss(y["spread"], ps, op.spread_noise_std * infl),
            _log_gauss(y["volume"], pv, op.volume_noise_std * infl),
            _log_gauss(y["realvol"], pr, op.vol_noise_std * infl))
    vA, vR = np.empty(8), np.empty(8)
    for mask in range(8):
        ll = np.zeros_like(f.w)
        for i in range(3):
            if mask >> i & 1:
                ll = ll + comp[i]
        with np.errstate(divide="ignore"):
            a = np.log(f.w) + ll
            logL = logsumexp(a, axis=1)
            log_joint = np.log(f.mu_prior) + logL
        inc = float(logsumexp(log_joint))
        if np.isfinite(inc):
            mu = np.exp(log_joint - inc)
            w_post = np.exp(a - np.where(np.isfinite(logL), logL, 0.0)[:, None])
            bad = ~np.isfinite(logL)
            if np.any(bad):
                w_post[bad] = f.w[bad]
        else:
            mu, w_post = f.mu_prior.copy(), f.w.copy()
        mu = np.maximum(mu, f.pp.regime_prob_floor)
        mu = mu / mu.sum()
        am = (w_post * f.alpha).sum(axis=1)
        vA[mask] = float((mu * am).sum())
        vR[mask] = float(mu[1])
    return vA, vR


def windows(reg, to):
    m = np.zeros(len(reg), bool)
    for t in range(1, len(reg)):
        if reg[t] == to and reg[t - 1] != to:
            m[t:t + 3] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=2_000_000)
    ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--outdir", default=os.path.join(HERE, "outputs", "stage4"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    print("Shapley toy-game self-test:", "PASS" if selftest_shapley() else "FAIL")
    if not selftest_shapley():
        sys.exit("shapley3 is wrong; stopping")

    sp, op = SimParams(), ObsParams()
    rp = RegimeParams.default(); rp.init_probs = rp.stationary_distribution()
    assert int(rp.n_regimes) == 2, "this script assumes K = 2 (calm/stressed)"
    N, th_a = sp.N, sp.theta_alpha
    f = IMMParticleFilter(sp, op, rp, IMMParams(n_particles=a.particles), seed=0)

    SA = np.zeros((a.n, len(GROUPS), 3)); SAs = np.zeros_like(SA)          # sums of |phi| and signed phi, functional A
    SR = np.zeros_like(SA); SRs = np.zeros_like(SA)                        # functional R
    CNT = np.zeros((a.n, len(GROUPS)))
    DA = np.zeros((a.n, len(GROUPS))); IA = np.zeros_like(DA)              # sum |update|, sum |interaction|  (A)
    DR = np.zeros_like(DA); IR = np.zeros_like(DA)                         # (R)
    max_full_err = max_eff_err = 0.0
    t0 = time.time()
    for i in range(a.n):
        seed = a.seed0 + i
        st = I.seed_streams(seed)
        sim = RegimeSwitchingSimulator(sp, rp, seed=st["sim"])
        gen = sim.run_stepwise(seed=st["sim"]); next(gen)
        while True:
            try:
                gen.send(0.0)
            except StopIteration as stop:
                ep = stop.value
                break
        om = make_observation_model_v2(a.obs_kind, seed=st["obs"], tau=sp.tau); om.reset(st["obs"])
        f.reset(st["filter"])
        reg = np.asarray(ep.regime[:N], dtype=int)
        masks = {"all": np.ones(N, bool), "calm": reg == 0, "stressed": reg == 1,
                 "into_stressed": windows(reg, 1), "into_calm": windows(reg, 0)}
        for t in range(N):
            raw = om.observe_step(float(ep.kappa[t]), float(ep.alpha[t]), int(ep.regime[t]))
            obs = I.Observation.from_dict(t, raw)
            vA, vR = coalition_values(f, obs.core_dict())
            bel = f.update(obs)                                            # advance the REAL filter
            max_full_err = max(max_full_err, abs(vA[7] - bel.alpha_mean) / th_a, abs(vR[7] - bel.regime_probs[1]))
            pA, pR = shapley3(vA / th_a), shapley3(vR)
            dA, dR = (vA[7] - vA[0]) / th_a, vR[7] - vR[0]
            max_eff_err = max(max_eff_err, abs(pA.sum() - dA), abs(pR.sum() - dR))
            inA = dA - sum((vA[1 << c] - vA[0]) / th_a for c in range(3))
            inR = dR - sum((vR[1 << c] - vR[0]) for c in range(3))
            for g, name in enumerate(GROUPS):
                if masks[name][t]:
                    CNT[i, g] += 1
                    SA[i, g] += np.abs(pA); SAs[i, g] += pA; SR[i, g] += np.abs(pR); SRs[i, g] += pR
                    DA[i, g] += abs(dA); IA[i, g] += abs(inA); DR[i, g] += abs(dR); IR[i, g] += abs(inR)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{a.n} worlds ({time.time() - t0:.0f}s)", flush=True)

    print(f"full coalition vs the filter's real belief: max error {max_full_err:.2e} (want < 1e-9)")
    print(f"Shapley efficiency (sum phi = v(all) - v(empty)): max error {max_eff_err:.2e} (want < 1e-9)")
    if max_full_err > 1e-9 or max_eff_err > 1e-9:
        sys.exit("CHECK FAILED: the replicated update does not match the filter; do not use the numbers")

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, a.n, a.n) for _ in range(a.n_boot)]

    def pooled(S, idx):
        return S[idx].sum(axis=0) / np.maximum(CNT[idx].sum(axis=0), 1)[:, None]

    summary = {}
    for fname, Sabs, Ssgn, D, Iv, unit in (("A", SA, SAs, DA, IA, "units of theta_alpha"), ("R", SR, SRs, DR, IR, "probability")):
        title = "alpha posterior mean" if fname == "A" else "P(stressed)"
        print(f"\n=== functional {fname}: {title} ({unit}); obs={a.obs_kind}, {a.n} worlds, {a.particles} particles/regime ===")
        print(f"{'group (steps)':26s} {'channel':8s} {'mean|phi|  [95% CI]':>28s} {'share':>6s} {'signed mean':>12s}")
        summary[fname] = {}
        full = pooled(Sabs, np.arange(a.n))
        bt = np.array([pooled(Sabs, ix) for ix in boots])                   # (B, G, 3)
        sg = pooled(Ssgn, np.arange(a.n))
        for g, name in enumerate(GROUPS):
            ns = int(CNT[:, g].sum())
            if ns == 0:
                print(f"{name + ' (0)':26s} n/a"); continue
            tot = full[g].sum()
            summary[fname][name] = {"steps": ns, "mean_update": float(D[:, g].sum() / ns),
                                    "mean_interaction": float(Iv[:, g].sum() / ns), "channels": {}}
            for c, ch in enumerate(CH):
                lo, hi = np.percentile(bt[:, g, c], [2.5, 97.5])
                summary[fname][name]["channels"][ch] = dict(mean_abs=float(full[g, c]), ci=[float(lo), float(hi)],
                                                            share=float(full[g, c] / tot), signed=float(sg[g, c]))
                print(f"{(name + ' (' + str(ns) + ')') if c == 0 else '':26s} {ch:8s} {full[g, c]:10.4f} [{lo:8.4f},{hi:8.4f}] "
                      f"{full[g, c] / tot:6.2f} {sg[g, c]:+12.4f}")
            print(f"{'':26s} mean |update| {D[:, g].sum() / ns:.4f}   mean |interaction| {Iv[:, g].sum() / ns:.4f} "
                  f"(interaction = update minus sum of single-channel effects)")

    out = os.path.join(a.outdir, f"shap_{a.obs_kind}.json")
    with open(out, "w") as fh:
        json.dump(dict(obs_kind=a.obs_kind, n=a.n, seed0=a.seed0, particles=a.particles, max_full_err=max_full_err,
                       max_eff_err=max_eff_err, summary=summary), fh, indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    xs = [g for g in GROUPS if g in summary["A"]]
    for k, fname in enumerate(("A", "R")):
        for c, ch in enumerate(CH):
            ax[k].bar(np.arange(len(xs)) + 0.27 * (c - 1), [summary[fname][g]["channels"][ch]["share"] for g in xs],
                      width=0.27, label=ch)
        ax[k].set_xticks(np.arange(len(xs))); ax[k].set_xticklabels(xs, rotation=20, fontsize=8)
        ax[k].set_ylabel("share of mean |Shapley|"); ax[k].set_title("alpha posterior mean" if fname == "A" else "P(stressed)")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"IMM-PF belief-update attribution by proxy channel ({a.obs_kind})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, f"shap_{a.obs_kind}.png"), dpi=130)
    print("wrote", out, "and the figure")


if __name__ == "__main__":
    main()