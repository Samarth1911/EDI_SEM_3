"""
validate_imm_pf.py  (STAGE 2 -- Person 2)
-----------------------------------------
Run from project root:   python validation/validate_imm_pf.py
Quick smoke test:        python validation/validate_imm_pf.py --quick
Include the v1 Hawkes arm (it has a look-ahead, see Stage 0):  --kinds=gaussian,spread_hawkes,hawkes

Validates the IMM-PF the same way the bootstrap filter was validated (tracking RMSE,
uncertainty calibration) plus the new Stage 2 check (regime-detection accuracy report).
Two kinds of results are kept separate on purpose (same convention as validate_stage1.py):
  [HARD]  correctness / contract checks with an explicit pass/fail rule (mixing algebra,
          determinism, belief contract, ESS bounds, K=1 reduces to the bootstrap PF, a real
          K=2 StepResponse passes the Stage 0 validator).
  [INFO]  performance numbers with NO pass/fail (tracking, calibration, regime detection,
          sensitivity to jump_prob / noise inflation / a misspecified regime model, shock case).

No result in this file has been observed by the author of the code: nothing here is a claim
until you run it. Everything is written to validation/outputs/:
  09_imm_tracking_<kind>.png     tracking vs truth with regime shading, bootstrap PF overlaid
  10_imm_calibration_<kind>.png  posterior std, z-scores, ESS, pooled coverage curve
  11_regime_detection.png        p(stressed) timeline, reliability, detection delays, forced-window response
  stage2_validation_summary.json all numbers
  stage2_results.md              section (b) of the Stage 2 handoff, numbers auto-filled
"""
import sys, os, json, math, time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams, force_regime_window
from observation_model import ObsParams
from spread_hawkes import make_observation_model_v2
from particle_filter import ParticleFilter, PFParams, predicted_proxy_means
from imm_particle_filter import IMMParticleFilter, IMMParams, imm_mixing
from belief_metrics import (TrackingPool, beliefs_to_arrays, regime_detection_report, paired_diff)
import interfaces as I

OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

QUICK = "--quick" in sys.argv
KINDS = ["gaussian", "spread_hawkes"]
for _a in sys.argv:
    if _a.startswith("--kinds="):
        KINDS = [k for k in _a.split("=", 1)[1].split(",") if k]

N_EP_MAIN = 40 if QUICK else 200      # episodes for the main tracking / regime-detection statistics
N_EP_SWEEP = 15 if QUICK else 60      # episodes per sensitivity configuration
N_EP_K1 = 8 if QUICK else 20          # episodes for the K=1 regression check
N_EP_FORCED = 10 if QUICK else 40     # episodes for the controlled forced-stress test
BASE_SEED = 50_000                    # disjoint from the seeds used in Stages 0/1
SHOCK_STEP, SHOCK_MULT = 25, 3.0


# ------------------------------------------------------------------ small helpers
def fm(x, spec=".3g"):
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "n/a"
        return format(x, spec)
    except Exception:
        return str(x)


def _clean(o):
    """Make results JSON-safe (numpy -> python, nan/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def make_rp() -> RegimeParams:
    """Default 2-regime world, but starting from the STATIONARY distribution, so episodes do not
    all start calm (Stage 0 DECISION 5: ~13% of calm-start N=50 episodes never see stress)."""
    base = RegimeParams.default()
    pi = np.clip(base.stationary_distribution(), 0.0, None)
    return RegimeParams(specs=base.specs, P=base.P.copy(), init_probs=pi / pi.sum())


def rp_wrong_P(rp: RegimeParams) -> RegimeParams:
    """Filter believes regimes are LESS persistent than they are (0.90/0.90 vs 0.96/0.92)."""
    return RegimeParams(specs=rp.specs, P=np.array([[0.90, 0.10], [0.10, 0.90]]), init_probs=rp.init_probs)


def rp_wrong_levels(rp: RegimeParams) -> RegimeParams:
    """Filter believes stress is milder than it is (alpha x2.0 vs x2.5, kappa x1.3 vs x1.6)."""
    specs = (rp.specs[0], replace(rp.specs[1], theta_alpha_mult=2.0, theta_kappa_mult=1.3))
    return RegimeParams(specs=specs, P=rp.P.copy(), init_probs=rp.init_probs)


def gen_episode(seed, kind, rp, sp, shock_step=None, shock_mult=SHOCK_MULT, forced_path=None):
    """Simulate one regime-switching episode under TWAP (kappa/alpha/regime paths do not depend on
    the trade schedule), generate proxies with the chosen observation arm, return (episode, rows).
    `rows` are interfaces.Observation objects: NO regime, NO truth -- exactly what a filter may see."""
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    kw = dict(velocities=sim.twap_schedule(), seed=seed)
    if shock_step is not None:
        kw.update(shock_step=shock_step, shock_jump_alpha_mult=shock_mult)
    if forced_path is not None:
        kw["forced_regime_path"] = forced_path
    ep = sim.simulate_episode(**kw)
    st = I.seed_streams(seed)
    om = make_observation_model_v2(kind, seed=st["obs"], tau=sp.tau)
    df = om.observe_episode(ep, seed=st["obs"])
    rows = [I.Observation.from_dict(i, r) for i, r in enumerate(df.to_dict("records"))]
    return ep, rows


def make_eps(seeds, kind, rp, sp, **kw):
    return [(s,) + gen_episode(s, kind, rp, sp, **kw) for s in seeds]


def run_config(eps, sp, rp_f, ip, shock_step=None):
    """Run the IMM-PF over cached episodes; pool tracking errors; collect regime probabilities."""
    pool, p1s, regs, beliefs_all = TrackingPool(), [], [], []
    bel_errs, ess_viol, per_step = [], 0, []
    for seed, ep, rows in eps:
        f = IMMParticleFilter(sp, ObsParams(), rp_f, ip, seed=I.seed_streams(seed)["filter"])
        t0 = time.perf_counter()
        bel = [f.update(o) for o in rows]
        per_step.append((time.perf_counter() - t0) / len(rows))
        N, K, n = len(rows), rp_f.n_regimes, ip.n_particles
        for b in bel:
            bel_errs += I.validate_belief(b, K=K)
            if not (1.0 - 1e-6 <= b.ess <= K * n + 1e-6):
                ess_viol += 1
            for rb in b.per_regime:
                if not (1.0 - 1e-6 <= rb.ess <= n + 1e-6):
                    ess_viol += 1
        A = beliefs_to_arrays(bel)
        reg = np.asarray(ep.regime[:N])
        win = None
        if shock_step is not None:
            win = np.zeros(N, dtype=bool)
            win[shock_step:shock_step + 10] = True
        pool.add(A["alpha_mean"], A["alpha_std"], A["kappa_mean"], A["kappa_std"],
                 ep.alpha[:N], ep.kappa[:N], reg, ess=A["ess"], window=win)
        p1s.append(A["p"][:, 1] if A["p"].shape[1] > 1 else np.zeros(N))
        regs.append(reg)
        beliefs_all.append(bel)
    return dict(pool=pool, p1s=p1s, regs=regs, beliefs=beliefs_all, bel_errs=bel_errs,
                ess_viol=ess_viol, step_ms=1000.0 * float(np.mean(per_step)))


def run_bootstrap(eps, sp, shock_step=None):
    """The UNMODIFIED validated bootstrap PF (single regime, default PFParams) on the same observations."""
    pool, hs = TrackingPool(), []
    for seed, ep, rows in eps:
        pf = ParticleFilter(sp, ObsParams(), PFParams(), seed=I.seed_streams(seed)["filter"])
        for o in rows:
            pf.step(o.core_dict())
        h = {k: np.array(v[1:]) for k, v in pf.history.items()}        # history[t+1] <-> step t
        N = len(rows)
        win = None
        if shock_step is not None:
            win = np.zeros(N, dtype=bool)
            win[shock_step:shock_step + 10] = True
        pool.add(h["alpha_mean"], h["alpha_std"], h["kappa_mean"], h["kappa_std"],
                 ep.alpha[:N], ep.kappa[:N], np.asarray(ep.regime[:N]), ess=h["ess"], window=win)
        hs.append(h)
    return pool, hs


# ------------------------------------------------------------------ HARD checks
def check_mixing():
    rng, errs = np.random.default_rng(0), []
    for _ in range(50):
        K = int(rng.integers(2, 5))
        P = rng.random((K, K)); P /= P.sum(axis=1, keepdims=True)
        mu = rng.random(K); mu /= mu.sum()
        mp, M = imm_mixing(mu, P)
        if not np.allclose(mp, P.T @ mu):
            errs.append("mu_pred != P^T mu")
        if not np.allclose(M.sum(axis=0), 1.0):
            errs.append("mixing columns do not sum to 1")
        if not np.allclose(M * mp[None, :], mu[:, None] * P):
            errs.append("M * mu_pred != joint")
    mp, M = imm_mixing(np.array([0.3, 0.7]), np.eye(2))
    if not (np.allclose(mp, [0.3, 0.7]) and np.allclose(M, np.eye(2))):
        errs.append("identity transition should give identity mixing")
    mp, M = imm_mixing(np.array([0.5, 0.5]), np.array([[1.0, 0.0], [1.0, 0.0]]))
    if not (np.allclose(mp, [1.0, 0.0]) and np.allclose(M.sum(axis=0), 1.0)):
        errs.append("unreachable regime should still get a valid mixing column")
    return errs


def check_determinism(sp, rp, ip, rows):
    def go(f):
        return np.array([[b.alpha_mean, b.kappa_mean, *b.regime_probs] for b in (f.update(o) for o in rows)])
    f = IMMParticleFilter(sp, ObsParams(), rp, replace(ip, n_particles=300), seed=123)
    a = go(f)
    b = go(IMMParticleFilter(sp, ObsParams(), rp, replace(ip, n_particles=300), seed=123))
    f.reset(123)
    c = go(f)
    return [] if (np.array_equal(a, b) and np.array_equal(a, c)) else ["same seed gave different beliefs"]


def check_k1(sp, n_ep):
    """K=1 IMM-PF must reduce to the bootstrap PF (same algorithm, same draw order)."""
    rp1 = RegimeParams.single()
    eps = make_eps([BASE_SEED + 70_000 + i for i in range(n_ep)], "gaussian", rp1, sp)
    imm = run_config(eps, sp, rp1, IMMParams(n_particles=5000, jump_prob=0.08))
    boot, _ = run_bootstrap(eps, sp)
    a, b = imm["pool"].summary(), boot.summary()
    ra, rk = a["alpha_rmse"] / b["alpha_rmse"], a["kappa_rmse"] / b["kappa_rmse"]
    dcov = abs(a["alpha_cov90"] - b["alpha_cov90"])
    ok = (0.90 <= ra <= 1.10) and (0.90 <= rk <= 1.10) and dcov <= 0.05
    return ok, dict(alpha_rmse_ratio=ra, kappa_rmse_ratio=rk, alpha_cov90_abs_diff=dcov,
                    imm_alpha_rmse=a["alpha_rmse"], boot_alpha_rmse=b["alpha_rmse"], n_episodes=n_ep,
                    rule="alpha & kappa RMSE ratio in [0.90, 1.10] and |alpha cov90 diff| <= 0.05 "
                         "(tolerances are the author's judgement, not derived)")


def check_step_response(sp, rp, ip, eps, kind):
    """A real K=2 belief must pass the Stage 0 StepResponse validator (the bootstrap PF could not:
    its K=1 belief mismatched config.n_regimes=2 -- see validate_interfaces.py)."""
    seed, ep, rows = eps[0]
    f = IMMParticleFilter(sp, ObsParams(), rp, replace(ip, n_particles=300), seed=1)
    bel = [f.update(o) for o in rows]
    N = len(rows)
    tr = I.TruthStep(t=N - 1, S=float(ep.S[N - 1]), kappa=float(ep.kappa[N - 1]), alpha=float(ep.alpha[N - 1]),
                     regime=int(ep.regime[N - 1]), q=float(ep.q[N - 1]), N=N)
    resp = I.step_response(
        episode_id="ep-test", seed=int(seed), truth=tr, obs=rows[-1], belief=bel[-1],
        policies={"twap": {"q": 0.0, "v": 0.0, "S": tr.S, "S_exec": tr.S, "cum_cost": 0.0}},
        config={"obs_kind": kind, "n_regimes": rp.n_regimes, "regime_names": rp.names,
                "q0": sp.q0, "S0": sp.S0}, done=True)
    errs = I.validate_step_response(resp)
    json.dumps(resp)                                        # must be JSON-serialisable
    return errs


def check_identity_P(sp, n_ep=3):
    """Exercises the K=2 MIXING/RESAMPLING path (the K=1 check bypasses it). With P = identity and a
    one-hot prior the wrong regime must stay essentially impossible: no probability may leak between
    banks through the union resampling. Calm world -> p(stressed) ~ 0; stressed world -> p(stressed) ~ 1."""
    base = RegimeParams.default()
    worlds = {"calm": (np.array([1.0, 0.0]), lambda p: float(np.max(p)) < 1e-3),
              "stressed": (np.array([0.0, 1.0]), lambda p: float(np.min(p)) > 1.0 - 1e-3)}
    detail, ok = [], True
    for name, (init, good) in worlds.items():
        rp = RegimeParams(specs=base.specs, P=np.eye(2), init_probs=init)
        worst = []
        for i in range(n_ep):
            seed = BASE_SEED + 110_000 + i
            ep, rows = gen_episode(seed, "gaussian", rp, sp)
            f = IMMParticleFilter(sp, ObsParams(), rp, IMMParams(n_particles=500), seed=seed)
            p1 = np.array([f.update(o).regime_probs[1] for o in rows])
            worst.append(float(p1.max()) if name == "calm" else float(p1.min()))
        w = max(worst) if name == "calm" else min(worst)
        ok &= good(np.array(worst))
        detail.append(f"{name}: worst p(stressed) = {w:.2e}")
    return ok, "; ".join(detail) + "  (rule: calm max < 1e-3, stressed min > 0.999; thresholds are the author's judgement)"


def proxy_residuals(eps):
    """INFO diagnostic: is the Gaussian likelihood right for this observation arm? Uses the TRUE state
    to compute z = (y - predicted proxy mean) / assumed noise std per channel and regime.
    A mean z far from 0 is a BIAS (widening the noise cannot fix it); sd z far from 1 is a scale error."""
    op = ObsParams()
    std = dict(spread=op.spread_noise_std, volume=op.volume_noise_std, realvol=op.vol_noise_std)
    Z = {c: [] for c in std}
    R = []
    for _seed, ep, rows in eps:
        N = len(rows)
        pm = dict(zip(("spread", "volume", "realvol"), predicted_proxy_means(np.asarray(ep.kappa[:N]), np.asarray(ep.alpha[:N]), op)))
        for c in std:
            y = np.array([getattr(o, c) for o in rows], dtype=float)
            Z[c].append((y - pm[c]) / std[c])
        R.append(np.asarray(ep.regime[:N]))
    reg = np.concatenate(R)
    out = {}
    for c in std:
        z = np.concatenate(Z[c])
        out[c] = {f"regime{r}": dict(mean_z=float(z[reg == r].mean()), sd_z=float(z[reg == r].std()), n=int((reg == r).sum()))
                  for r in (0, 1) if np.any(reg == r)}
    return out


# ------------------------------------------------------------------ controlled forced-stress test
def forced_window_test(kind, sp, ip, n_seeds, start=20, dur=20):
    """Calm for `start` steps, forced stress for `dur` steps, calm again. Isolates detection delay from
    the randomness of natural spells. The filter still expects switching at rate P."""
    rp = RegimeParams.default()                             # calm start; filter knows it
    path = force_regime_window(np.zeros(sp.N + 1, dtype=int), start, dur, 1)
    end, P1 = start + dur, []
    for i in range(n_seeds):
        seed = BASE_SEED + 90_000 + i
        ep, rows = gen_episode(seed, kind, rp, sp, forced_path=path)
        f = IMMParticleFilter(sp, ObsParams(), rp, ip, seed=I.seed_streams(seed)["filter"])
        P1.append([f.update(o).regime_probs[1] for o in rows])
    P1 = np.asarray(P1)
    on = [np.flatnonzero(r[start:end] >= 0.5) for r in P1]
    rec = [np.flatnonzero(r[end:] < 0.5) for r in P1]
    med = lambda L: float(np.median([h[0] for h in L if h.size])) if any(h.size for h in L) else float("nan")
    return dict(n=int(n_seeds), start=start, end=end,
                onset_detect_rate=float(np.mean([h.size > 0 for h in on])), onset_median_delay=med(on),
                recovery_detect_rate=float(np.mean([h.size > 0 for h in rec])), recovery_median_delay=med(rec),
                false_alarm_before_window=float(np.mean([np.any(r[:start] >= 0.5) for r in P1])),
                mean_p1=P1.mean(axis=0).tolist(), sd_p1=P1.std(axis=0).tolist())


# ------------------------------------------------------------------ figures
def _shade(ax, reg):
    reg, t, N = np.asarray(reg, dtype=int), 0, len(reg)
    while t < N:
        if reg[t] == 1:
            e = t
            while e < N and reg[e] == 1:
                e += 1
            ax.axvspan(t - 0.5, e - 0.5, color="tab:red", alpha=0.13, lw=0)
            t = e
        else:
            t += 1


def fig_tracking(ep, bel, boot_h, kind, path):
    A, N = beliefs_to_arrays(bel), len(bel)
    t, reg = np.arange(N), np.asarray(ep.regime[:N])
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, true, name in ((axes[0], "alpha", ep.alpha[:N], r"$\alpha_t$ (temporary impact)"),
                                (axes[1], "kappa", ep.kappa[:N], r"$\kappa_t$ (permanent impact)")):
        mu, sd = A[f"{key}_mean"], A[f"{key}_std"]
        _shade(ax, reg)
        ax.plot(t, true, color="black", lw=1.8, label=f"true {name}")
        ax.plot(t, mu, color="tab:blue", lw=1.5, label="IMM-PF mean")
        ax.fill_between(t, mu - 1.645 * sd, mu + 1.645 * sd, color="tab:blue", alpha=0.25, label="IMM-PF 90% band")
        ax.plot(t, boot_h[f"{key}_mean"], color="tab:green", lw=1.2, ls="--", label="bootstrap PF mean (regime-blind)")
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    _shade(ax, reg)
    ax.plot(t, A["p"][:, 1], color="tab:red", lw=1.6, label="filtered P(stressed)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("step"); ax.set_ylabel("P(stressed)")
    ax.set_title("Regime probability (red shading = truly stressed)")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"IMM-PF validation (1/3): tracking, one episode, obs arm = {kind}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_calibration(ep, bel, boot_h, summ, kind, path):
    A, N = beliefs_to_arrays(bel), len(bel)
    t, reg = np.arange(N), np.asarray(ep.regime[:N])
    za = (A["alpha_mean"] - ep.alpha[:N]) / np.maximum(A["alpha_std"], 1e-30)
    zk = (A["kappa_mean"] - ep.kappa[:N]) / np.maximum(A["kappa_std"], 1e-30)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]; _shade(ax, reg)
    ax.plot(t, A["alpha_std"], color="tab:blue", label="IMM-PF")
    ax.plot(t, boot_h["alpha_std"], color="tab:green", ls="--", label="bootstrap PF")
    ax.set_title(r"(a) posterior std of $\alpha_t$"); ax.legend(fontsize=8)
    ax = axes[0, 1]; _shade(ax, reg)
    ax.plot(t, za, color="tab:orange", label=r"$\alpha$ z-score (IMM-PF)")
    ax.plot(t, zk, color="tab:blue", alpha=0.7, label=r"$\kappa$ z-score (IMM-PF)")
    ax.axhspan(-1.645, 1.645, color="green", alpha=0.12, label="90% band")
    ax.set_ylim(-6, 6); ax.set_title("(b) z = (estimate - truth) / std, should stay O(1)"); ax.legend(fontsize=8)
    ax = axes[1, 0]; _shade(ax, reg)
    ax.plot(t, A["ess"], color="tab:green", label="combined ESS (Decision 3)")
    if "ess_regime" in A:
        for j in range(A["ess_regime"].shape[1]):
            ax.plot(t, A["ess_regime"][:, j], lw=1, alpha=0.8, label=f"bank {j} ESS")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title("(c) effective sample size"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    lv = [0.5, 0.8, 0.9, 0.95]
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="perfect")
    ax.plot(lv, [summ["imm"]["alpha_cov_curve"][str(l)] for l in lv], "o-", color="tab:blue", label="IMM-PF")
    ax.plot(lv, [summ["boot"]["alpha_cov_curve"][str(l)] for l in lv], "s--", color="tab:green", label="bootstrap PF")
    ax.set_xlim(0.45, 1); ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("nominal central-interval level"); ax.set_ylabel("empirical coverage of true alpha")
    ax.set_title("(d) pooled coverage over all episodes"); ax.legend(fontsize=8)
    fig.suptitle(f"IMM-PF validation (2/3): uncertainty calibration, obs arm = {kind}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_regime(main, forced, showcase_idx, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cols = {"gaussian": "tab:blue", "spread_hawkes": "tab:red", "hawkes": "tab:purple"}
    k0 = next(iter(main))
    reg = main[k0]["regs"][showcase_idx]
    ax = axes[0, 0]; _shade(ax, reg)
    for kind, m in main.items():
        ax.plot(np.arange(len(reg)), m["p1s"][showcase_idx], color=cols.get(kind), lw=1.5, label=f"P(stressed), {kind}")
    ax.axhline(0.5, color="gray", ls=":"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("step")
    ax.set_title("(a) filtered regime probability, one episode (red = truly stressed)"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="perfect")
    for kind, m in main.items():
        rel = m["regime"]["reliability"]
        ok = [i for i, c in enumerate(rel["count"]) if c > 0]
        ax.plot([rel["pred"][i] for i in ok], [rel["emp"][i] for i in ok], "o-", color=cols.get(kind), label=kind)
    ax.set_xlabel("predicted P(stressed)"); ax.set_ylabel("empirical frequency of stressed")
    ax.set_title("(b) reliability of the regime probability (pooled)"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    dmax = max([max(m["regime"]["by_threshold"]["0.5"]["onset"]["delays"] or [0]) for m in main.values()] + [1])
    bins = np.arange(-0.5, dmax + 1.5, 1.0)
    for i, (kind, m) in enumerate(main.items()):
        d = m["regime"]["by_threshold"]["0.5"]["onset"]["delays"]
        if d:
            ax.hist(d, bins=bins, alpha=0.55, color=cols.get(kind), label=f"{kind} (n={len(d)})")
    ax.set_xlabel("steps from true onset to P(stressed) >= 0.5 (detected spells only)")
    ax.set_title("(c) stress-onset detection delay, natural spells"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for kind, fw in forced.items():
        mu, sd = np.array(fw["mean_p1"]), np.array(fw["sd_p1"])
        x = np.arange(len(mu))
        ax.plot(x, mu, color=cols.get(kind), label=kind)
        ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=cols.get(kind), alpha=0.15)
    fw0 = next(iter(forced.values()))
    ax.axvspan(fw0["start"] - 0.5, fw0["end"] - 0.5, color="tab:red", alpha=0.13, lw=0)
    ax.axhline(0.5, color="gray", ls=":"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("step")
    ax.set_title("(d) forced stress window: mean P(stressed) +/- 1 sd across seeds"); ax.legend(fontsize=8)
    fig.suptitle("IMM-PF validation (3/3): regime detection", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ results section (b), numbers auto-filled
def write_results(R, path):
    L = []
    w = L.append
    meta = R["meta"]
    w("# Stage 2 -- validation results (auto-generated by `validation/validate_imm_pf.py`)\n")
    w(f"Mode: {'QUICK (smoke test, do not quote these numbers)' if meta['quick'] else 'full'}. "
      f"Observation arms: {', '.join(meta['kinds'])}. Main episodes per arm: {meta['n_ep_main']} "
      f"(N = {meta['N']} steps, stationary regime start). Runtime {fm(meta['seconds'], '.0f')} s. "
      f"IMM defaults: {meta['imm_params']}.\n")

    w("## HARD checks\n")
    w("| check | result | detail |\n|---|---|---|")
    for h in R["hard"]:
        w(f"| {h['name']} | {'PASS' if h['ok'] else '**FAIL**'} | {h['detail']} |")
    w("")

    for kind, m in R["main"].items():
        a, b, det = m["imm"], m["boot"], m["regime"]
        w(f"## Tracking and calibration -- obs arm `{kind}` (pooled over {a['n_episodes']} episodes, {a['n_steps']} steps)\n")
        w("| metric | IMM-PF | bootstrap PF (regime-blind, validated) |\n|---|---|---|")
        rows = [("alpha RMSE", "alpha_rmse"), ("alpha RMSE, truly calm steps", "alpha_rmse_regime0"),
                ("alpha RMSE, truly stressed steps", "alpha_rmse_regime1"),
                ("alpha bias (est - truth), calm", "alpha_bias_regime0"),
                ("alpha bias (est - truth), stressed", "alpha_bias_regime1"),
                ("alpha 90% coverage (nominal 0.90)", "alpha_cov90"),
                ("alpha 90% coverage, calm", "alpha_cov90_regime0"),
                ("alpha 90% coverage, stressed", "alpha_cov90_regime1"),
                ("kappa RMSE", "kappa_rmse"), ("kappa 90% coverage", "kappa_cov90"),
                ("naive 'always calm long-run mean' alpha RMSE", "alpha_rmse_naive_theta"),
                ("min ESS (IMM: combined set)", "min_ess"), ("median ESS", "median_ess")]
        for label, key in rows:
            spec = ".3f" if "cov" in key else ".3g"
            w(f"| {label} | {fm(a.get(key), spec)} | {fm(b.get(key), spec)} |")
        p = m["paired_alpha_rmse"]
        w(f"\nPer-episode alpha RMSE, IMM minus bootstrap (paired): mean {fm(p['mean'])} "
          f"(s.e. {fm(p['se'])}, n = {p['n']}); IMM lower in {fm(100 * p['frac_a_lower'], '.0f')}% of episodes. "
          f"Negative = IMM better. Update time: {fm(m['step_ms'], '.2f')} ms/step "
          f"({meta['imm_params_short']}).\n")

        w(f"### Regime detection -- `{kind}` (IMM-PF, {det['n_steps']} steps, {fm(100 * det['frac_stressed'], '.1f')}% truly stressed)\n")
        w(f"- MAP accuracy {fm(det['acc'], '.3f')} vs always-predict-majority {fm(det['acc_always_majority'], '.3f')}; "
          f"recall calm {fm(det['recall_calm'], '.3f')}, recall stressed {fm(det['recall_stressed'], '.3f')}, "
          f"balanced accuracy {fm(det['bal_acc'], '.3f')}.")
        w(f"- Brier {fm(det['brier'], '.4f')} vs best constant {fm(det['brier_const'], '.4f')} "
          f"(skill {fm(det['brier_skill'], '.3f')}); log-loss {fm(det['logloss'], '.3f')} vs best constant "
          f"{fm(det['logloss_const'], '.3f')}; AUC {fm(det['auc'], '.3f')}; ECE {fm(det['ece'], '.3f')}.\n")
        w("| threshold | onset: spells | detected | detected (len>=5) | median delay | mean delay | p90 delay | pre-declared | "
          "recovery: detected | median delay | steady-calm false alarms | steady-stress recall |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for th, d in det["by_threshold"].items():
            o, r = d["onset"], d["recovery"]
            w(f"| {th} | {o['n_spells']} | {fm(o['detect_rate'], '.2f')} | {fm(o['detect_rate_len_ge5'], '.2f')} "
              f"(n={o['n_len_ge5']}) | {fm(o['median_delay'])} | {fm(o['mean_delay'])} | {fm(o['p90_delay'])} | "
              f"{fm(o['pre_declared_frac'], '.2f')} | {fm(r['detect_rate'], '.2f')} | {fm(r['median_delay'])} | "
              f"{fm(d['false_alarm_rate'], '.3f')} | {fm(d['stressed_recall'], '.3f')} |")
        w("\nDelays are in steps and exist only for detected spells (optimistic alone; read with the detection rate). "
          "`pre-declared` = fraction of spells where P(target) was already >= threshold the step BEFORE the switch.\n")
        fw = R["forced"].get(kind)
        if fw:
            w(f"Controlled test (calm, then forced stress on steps {fw['start']}-{fw['end'] - 1}, then calm; {fw['n']} seeds): "
              f"onset detected in {fm(100 * fw['onset_detect_rate'], '.0f')}% (median delay {fm(fw['onset_median_delay'])} steps), "
              f"recovery declared in {fm(100 * fw['recovery_detect_rate'], '.0f')}% (median delay {fm(fw['recovery_median_delay'])} steps), "
              f"false alarm before the window in {fm(100 * fw['false_alarm_before_window'], '.0f')}% of episodes.\n")

    if R.get("shock"):
        w("## Injected alpha shock inside the regime world (x%g at step %d; window = 10 steps from the shock)\n" % (SHOCK_MULT, SHOCK_STEP))
        w("| arm | IMM alpha RMSE (window) | bootstrap alpha RMSE (window) | IMM cov90 (window) | bootstrap cov90 (window) | IMM overall alpha RMSE | bootstrap overall |\n|---|---|---|---|---|---|---|")
        for kind, s in R["shock"].items():
            a, b = s["imm"], s["boot"]
            w(f"| {kind} | {fm(a.get('alpha_rmse_window'))} | {fm(b.get('alpha_rmse_window'))} | "
              f"{fm(a.get('alpha_cov90_window'), '.3f')} | {fm(b.get('alpha_cov90_window'), '.3f')} | "
              f"{fm(a.get('alpha_rmse'))} | {fm(b.get('alpha_rmse'))} |")
        w("")

    w("## Sensitivity sweeps (INFO; same episodes for every configuration within an arm)\n")
    for kind, rows in R["sweeps"].items():
        w(f"### `{kind}` ({R['meta']['n_ep_sweep']} episodes)\n")
        w("| configuration | alpha RMSE | alpha cov90 | balanced acc | Brier skill | onset detected | onset median delay | steady-calm false alarms |\n|---|---|---|---|---|---|---|---|")
        for r in rows:
            w(f"| {r['name']} | {fm(r['alpha_rmse'])} | {fm(r['alpha_cov90'], '.3f')} | {fm(r['bal_acc'], '.3f')} | "
              f"{fm(r['brier_skill'], '.3f')} | {fm(r['onset_detect_rate'], '.2f')} | {fm(r['onset_median_delay'])} | "
              f"{fm(r['false_alarm_rate'], '.3f')} |")
        w("")
    if R.get("diag"):
        w("## Proxy-likelihood diagnostic (INFO): z = (observed proxy - mean at the TRUE state) / assumed noise std\n")
        w("Mean z far from 0 = bias in the Gaussian likelihood (noise inflation cannot fix it); sd z far from 1 = wrong noise scale.\n")
        w("| arm | channel | mean z calm | mean z stressed | sd z calm | sd z stressed |\n|---|---|---|---|---|---|")
        for kind, d in R["diag"].items():
            for ch, v in d.items():
                g = lambda r, k: fm(v.get(f"regime{r}", {}).get(k), ".2f")
                w(f"| {kind} | {ch} | {g(0, 'mean_z')} | {g(1, 'mean_z')} | {g(0, 'sd_z')} | {g(1, 'sd_z')} |")
        w("")
    w("## Timing\n")
    for kind, m in R["main"].items():
        w(f"- `{kind}`: {fm(m['step_ms'], '.2f')} ms per filtering step (single thread, includes belief construction).")
    w("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ------------------------------------------------------------------ main
def main():
    t_start = time.time()
    sp, ip, rp = SimParams(), IMMParams(), make_rp()
    R = dict(meta=dict(quick=QUICK, kinds=KINDS, n_ep_main=N_EP_MAIN, n_ep_sweep=N_EP_SWEEP, N=sp.N,
                       imm_params=str(ip), imm_params_short=f"{ip.n_particles} particles x {rp.n_regimes} banks"),
             hard=[], main={}, shock={}, forced={}, sweeps={}, diag={}, k1=None)
    hard = R["hard"]

    def add_hard(name, ok, detail=""):
        hard.append(dict(name=name, ok=bool(ok), detail=detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))

    print(f"validate_imm_pf  (mode: {'QUICK' if QUICK else 'full'}, arms: {KINDS})")

    # ---- HARD: algebra, protocol
    errs = check_mixing()
    add_hard("IMM mixing algebra (columns sum to 1, mu_pred = P^T mu, identity and unreachable-regime cases)",
             not errs, "; ".join(sorted(set(errs)))[:200])
    est_ok = isinstance(IMMParticleFilter(sp, ObsParams(), rp, IMMParams(n_particles=50), seed=0), I.Estimator)
    add_hard("IMMParticleFilter satisfies interfaces.Estimator (reset, update)", est_ok)
    try:
        ok_id, det_id = check_identity_P(sp)
    except Exception as e:                                   # a crash in the new test must not hide the rest
        ok_id, det_id = False, f"check crashed: {type(e).__name__}: {e}"
    add_hard("K=2 mixing path: identity P + one-hot prior keeps the wrong regime impossible (no bank leakage)", ok_id, det_id)

    # ---- main runs per observation arm
    main_res, eps_by_kind = {}, {}
    for kind in KINDS:
        print(f"\n[main] arm = {kind}: simulating {N_EP_MAIN} episodes ...")
        eps = make_eps([BASE_SEED + i for i in range(N_EP_MAIN)], kind, rp, sp)
        eps_by_kind[kind] = eps
        print(f"[main] arm = {kind}: IMM-PF ...")
        imm = run_config(eps, sp, rp, ip)
        print(f"[main] arm = {kind}: bootstrap PF baseline ...")
        boot, boot_hs = run_bootstrap(eps, sp)
        tr, tb = imm["pool"].summary(sp.theta_alpha, sp.theta_kappa), boot.summary(sp.theta_alpha, sp.theta_kappa)
        det = regime_detection_report(imm["p1s"], imm["regs"])
        main_res[kind] = dict(imm=tr, boot=tb, regime=det, step_ms=imm["step_ms"],
                              paired_alpha_rmse=paired_diff(imm["pool"].ep_alpha_rmse, boot.ep_alpha_rmse),
                              p1s=imm["p1s"], regs=imm["regs"], beliefs=imm["beliefs"], boot_hs=boot_hs)
        add_hard(f"C3 belief contract, K=2, every step of every episode [{kind}]", not imm["bel_errs"],
                 "; ".join(sorted(set(imm["bel_errs"])))[:200])
        add_hard(f"ESS within bounds (combined in [1, K*n], per bank in [1, n]) [{kind}]", imm["ess_viol"] == 0,
                 f"{imm['ess_viol']} violations")
        print(f"  alpha RMSE  IMM {tr['alpha_rmse']:.3e}  bootstrap {tb['alpha_rmse']:.3e}   "
              f"cov90 IMM {tr['alpha_cov90']:.3f}  bootstrap {tb['alpha_cov90']:.3f}")
        print(f"  regime: acc {det['acc']:.3f} (majority {det['acc_always_majority']:.3f}), "
              f"bal-acc {det['bal_acc']:.3f}, Brier skill {det['brier_skill']:.3f}, AUC {det['auc']:.3f}")

    # ---- HARD: determinism, K=1 regression, C5
    k0 = KINDS[0]
    errs = check_determinism(sp, rp, ip, eps_by_kind[k0][0][2])
    add_hard("Determinism: same seed -> identical beliefs (also after reset(seed))", not errs, "; ".join(errs))
    print("\n[k1] K=1 regression against the bootstrap PF ...")
    ok, k1 = check_k1(sp, N_EP_K1)
    R["k1"] = k1
    add_hard("K=1 IMM-PF reproduces the bootstrap PF statistically", ok,
             f"alpha RMSE ratio {k1['alpha_rmse_ratio']:.6f}, kappa RMSE ratio {k1['kappa_rmse_ratio']:.6f}, "
             f"|cov90 diff| {k1['alpha_cov90_abs_diff']:.6f}")
    errs = check_step_response(sp, rp, ip, eps_by_kind[k0], k0)
    add_hard("C5 StepResponse with a real K=2 belief passes validate_step_response and json.dumps", not errs,
             "; ".join(errs)[:200])

    # ---- controlled forced-stress test
    for kind in KINDS:
        print(f"[forced] arm = {kind} ...")
        R["forced"][kind] = forced_window_test(kind, sp, ip, N_EP_FORCED)

    # ---- injected shock inside the regime world
    for kind in KINDS:
        print(f"[shock] arm = {kind} ...")
        eps = make_eps([BASE_SEED + 30_000 + i for i in range(N_EP_SWEEP)], kind, rp, sp,
                       shock_step=SHOCK_STEP, shock_mult=SHOCK_MULT)
        a = run_config(eps, sp, rp, ip, shock_step=SHOCK_STEP)["pool"].summary(sp.theta_alpha)
        b = run_bootstrap(eps, sp, shock_step=SHOCK_STEP)[0].summary(sp.theta_alpha)
        R["shock"][kind] = dict(imm=a, boot=b)

    # ---- sensitivity sweeps
    cfgs = [("default", rp, ip),
            ("jump_prob = 0", rp, replace(ip, jump_prob=0.0)),
            ("jump_prob = 0.08 (bootstrap PF value)", rp, replace(ip, jump_prob=0.08)),
            ("obs_noise_inflation = 1.4", rp, replace(ip, obs_noise_inflation=1.4)),
            ("obs_noise_inflation = 2.0", rp, replace(ip, obs_noise_inflation=2.0)),
            ("obs_noise_inflation = 3.0", rp, replace(ip, obs_noise_inflation=3.0)),
            ("roughening_frac = 0.05 (default 0.15)", rp, replace(ip, roughening_frac=0.05)),
            ("MISSPECIFIED: filter P diag 0.90/0.90 (true 0.96/0.92)", rp_wrong_P(rp), ip),
            ("MISSPECIFIED: filter stress levels alpha x2.0, kappa x1.3 (true x2.5, x1.6)", rp_wrong_levels(rp), ip)]
    for kind in KINDS:
        print(f"[sweep] arm = {kind} ...")
        sub, rows = eps_by_kind[kind][:N_EP_SWEEP], []
        for name, rp_f, ip_f in cfgs:
            res = run_config(sub, sp, rp_f, ip_f)
            s, d = res["pool"].summary(), regime_detection_report(res["p1s"], res["regs"])
            th = d["by_threshold"]["0.5"]
            rows.append(dict(name=name, alpha_rmse=s["alpha_rmse"], alpha_cov90=s["alpha_cov90"], bal_acc=d["bal_acc"],
                             brier_skill=d["brier_skill"], onset_detect_rate=th["onset"]["detect_rate"],
                             onset_median_delay=th["onset"]["median_delay"], false_alarm_rate=th["false_alarm_rate"]))
        R["sweeps"][kind] = rows
        R["diag"][kind] = proxy_residuals(eps_by_kind[kind])

    # ---- figures
    N = sp.N
    idx = max(range(len(eps_by_kind[k0])),
              key=lambda i: int(np.sum(np.diff(np.asarray(eps_by_kind[k0][i][1].regime[:N])) != 0)))
    for kind in KINDS:
        m, ep = main_res[kind], eps_by_kind[kind][idx][1]
        fig_tracking(ep, m["beliefs"][idx], m["boot_hs"][idx], kind, os.path.join(OUT_DIR, f"09_imm_tracking_{kind}.png"))
        fig_calibration(ep, m["beliefs"][idx], m["boot_hs"][idx], dict(imm=m["imm"], boot=m["boot"]), kind,
                        os.path.join(OUT_DIR, f"10_imm_calibration_{kind}.png"))
    fig_regime(main_res, R["forced"], idx, os.path.join(OUT_DIR, "11_regime_detection.png"))

    # ---- write results (JSON first so nothing is lost if the markdown writer has a bug)
    R["main"] = {k: dict(imm=v["imm"], boot=v["boot"], regime=v["regime"], step_ms=v["step_ms"],
                         paired_alpha_rmse=v["paired_alpha_rmse"]) for k, v in main_res.items()}
    R["meta"]["seconds"] = time.time() - t_start
    with open(os.path.join(OUT_DIR, "stage2_validation_summary.json"), "w") as f:
        json.dump(_clean(R), f, indent=2)
    try:
        write_results(R, os.path.join(OUT_DIR, "stage2_results.md"))
        print(f"\nWrote figures, stage2_validation_summary.json and stage2_results.md to {OUT_DIR}")
    except Exception as e:                                                        # keep the run's numbers regardless
        print(f"\nWARNING: stage2_results.md could not be written ({type(e).__name__}: {e}); "
              f"the numbers are in stage2_validation_summary.json")

    n_ok = sum(h["ok"] for h in hard)
    print(f"\nHARD checks passed: {n_ok}/{len(hard)}   (INFO numbers are in stage2_results.md; they have no pass/fail)")
    sys.exit(0 if n_ok == len(hard) else 1)


if __name__ == "__main__":
    main()
