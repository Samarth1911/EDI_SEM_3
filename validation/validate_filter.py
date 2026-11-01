"""
validate_filter.py
--------------------
Validates the particle filter BEFORE anything is built on top of it.

Deliverables (written to validation/outputs/):
  01_tracking_plot.png     posterior mean +/- 1 std vs. true kappa_t, alpha_t, RMSE reported
  02_calibration_plot.png  (a) posterior std, (b) z-score = error/std against the 90% band,
                           (c) effective sample size -- around an injected shock
  validation_summary.json  multi-seed statistics (also printed)

WHY z-scores and not just "std goes up": a posterior std can spike and then
collapse toward zero while the estimate is still far from truth (weight
degeneracy) -- that looks like "uncertainty widening" but is the filter being
confidently wrong. An earlier version of this filter did exactly that. The
honest calibration test is whether |error| / std stays O(1), i.e. whether
~90% of steps fall inside the +/-1.645 std band.

Run from project root:  python validation/validate_filter.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import MarketSimulator
from observation_model import ObservationModel
from particle_filter import ParticleFilter

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SHOCK_STEP = 25
SHOCK_MULT = 3.0
SEED = 11
Z90 = 1.645


def run_episode(seed, shock_step=SHOCK_STEP, mult=SHOCK_MULT):
    sim = MarketSimulator(seed=seed)
    ep = sim.simulate_episode(velocities=sim.twap_schedule(), shock_step=shock_step,
                              shock_jump_kappa_mult=1.0, shock_jump_alpha_mult=mult)
    om = ObservationModel(seed=seed)
    obs_df = om.observe_episode(ep, seed=seed)
    pf = ParticleFilter(sim.p, om.p, seed=seed)
    hist = pf.run(obs_df)
    h = {k: np.array(v[1:]) for k, v in hist.items()}   # history[t+1] <-> true time t
    return sim.p, ep, h


def metrics(p, ep, h, shock_step):
    N = len(ep.v)
    ae = h["alpha_mean"] - ep.alpha[:N]
    ke = h["kappa_mean"] - ep.kappa[:N]
    za, zk = ae / np.maximum(h["alpha_std"], 1e-30), ke / np.maximum(h["kappa_std"], 1e-30)
    w = slice(shock_step, shock_step + 10) if shock_step is not None else slice(0, 0)
    out = dict(
        alpha_rmse=float(np.sqrt(np.mean(ae ** 2))), kappa_rmse=float(np.sqrt(np.mean(ke ** 2))),
        alpha_rmse_naive_theta=float(np.sqrt(np.mean((p.theta_alpha - ep.alpha[:N]) ** 2))),
        kappa_rmse_naive_theta=float(np.sqrt(np.mean((p.theta_kappa - ep.kappa[:N]) ** 2))),
        alpha_cov90=float(np.mean(np.abs(za) < Z90)), kappa_cov90=float(np.mean(np.abs(zk) < Z90)),
        min_ess=float(h["ess"].min()),
    )
    if shock_step is not None:
        out["alpha_cov90_shock_window"] = float(np.mean(np.abs(za[w]) < Z90))
        out["alpha_max_abs_z_shock_window"] = float(np.abs(za[w]).max())
        out["alpha_rmse_shock_window"] = float(np.sqrt(np.mean(ae[w] ** 2)))
    return out


def tracking_plot(p, ep, h):
    N = len(ep.v); t = ep.t[:N]
    m = metrics(p, ep, h, SHOCK_STEP)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, key, true, col, name, rm, naive in [
        (axes[0], "kappa", ep.kappa[:N], "tab:blue", r"$\kappa_t$ (permanent impact)", m["kappa_rmse"], m["kappa_rmse_naive_theta"]),
        (axes[1], "alpha", ep.alpha[:N], "tab:orange", r"$\alpha_t$ (temporary impact)", m["alpha_rmse"], m["alpha_rmse_naive_theta"]),
    ]:
        mu, sd = h[f"{key}_mean"], h[f"{key}_std"]
        ax.plot(t, true, color="black", lw=1.8, label=f"true {name}")
        ax.plot(t, mu, color=col, lw=1.5, label="filter posterior mean")
        ax.fill_between(t, mu - 1.645 * sd, mu + 1.645 * sd, color=col, alpha=0.25, label="90% credible band")
        ax.axvline(ep.t[SHOCK_STEP], color="red", ls="--", lw=1, alpha=0.7, label="shock injected")
        ax.set_title(f"{name}: RMSE = {rm:.2e}   (naive 'always guess long-run mean' = {naive:.2e})")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time")
    fig.suptitle(f"Particle filter validation (1/2): tracking -- single episode, seed {SEED}, "
                 f"alpha x{SHOCK_MULT:g} shock at step {SHOCK_STEP}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "01_tracking_plot.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return m, path


def calibration_plot(p, ep, h):
    N = len(ep.v); t = ep.t[:N]
    za = (h["alpha_mean"] - ep.alpha[:N]) / np.maximum(h["alpha_std"], 1e-30)
    zk = (h["kappa_mean"] - ep.kappa[:N]) / np.maximum(h["kappa_std"], 1e-30)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ts = ep.t[SHOCK_STEP]

    axes[0].plot(t, h["alpha_std"], color="tab:orange", label=r"posterior std, $\alpha_t$")
    axes[0].set_ylabel(r"$\alpha_t$ posterior std", color="tab:orange")
    ax0b = axes[0].twinx()
    ax0b.plot(t, h["kappa_std"], color="tab:blue", label=r"posterior std, $\kappa_t$")
    ax0b.set_ylabel(r"$\kappa_t$ posterior std", color="tab:blue")
    axes[0].axvline(ts, color="red", ls="--", lw=1, alpha=0.7)
    axes[0].set_title("(a) Posterior std: rises at the shock, settles as the filter re-converges")

    axes[1].plot(t, za, color="tab:orange", label=r"$\alpha$ z-score = (est - true)/std")
    axes[1].plot(t, zk, color="tab:blue", alpha=0.7, label=r"$\kappa$ z-score")
    axes[1].axhspan(-Z90, Z90, color="green", alpha=0.12, label="90% band (+/-1.645)")
    axes[1].axvline(ts, color="red", ls="--", lw=1, alpha=0.7)
    axes[1].set_ylim(-6, 6); axes[1].set_ylabel("z-score")
    axes[1].set_title("(b) Calibration: is the reported uncertainty the right size? (should stay O(1))")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, h["ess"], color="tab:green", label="effective sample size (ESS)")
    axes[2].axvline(ts, color="red", ls="--", lw=1, alpha=0.7, label="shock injected")
    axes[2].set_ylabel("ESS"); axes[2].set_xlabel("time")
    axes[2].set_title("(c) ESS dips at the shock (few particles near the new truth) then recovers")
    axes[2].legend(loc="lower right", fontsize=8)

    fig.suptitle("Particle filter validation (2/2): uncertainty and calibration around an injected shock", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "02_calibration_plot.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def multi_seed(n=30, shock_step=SHOCK_STEP, mult=SHOCK_MULT, base=1000):
    rows = []
    for s in range(n):
        p, ep, h = run_episode(base + s, shock_step, mult)
        rows.append(metrics(p, ep, h, shock_step))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows


def multi_seed_noshock(n=30, base=2000):
    rows = []
    for s in range(n):
        sim = MarketSimulator(seed=base + s)
        ep = sim.simulate_episode(velocities=sim.twap_schedule())
        om = ObservationModel(seed=base + s); df = om.observe_episode(ep, seed=base + s)
        pf = ParticleFilter(sim.p, om.p, seed=base + s)
        h = {k: np.array(v[1:]) for k, v in pf.run(df).items()}
        rows.append(metrics(sim.p, ep, h, None))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def shock_sweep(n=15):
    out = {}
    for mult in [1.5, 2.0, 3.0, 4.0, 5.0]:
        rows = [metrics(*run_episode(3000 + s, 25, mult), 25) for s in range(n)]
        out[f"x{mult:g}"] = dict(
            alpha_rmse=float(np.mean([r["alpha_rmse"] for r in rows])),
            cov90_window=float(np.mean([r["alpha_cov90_shock_window"] for r in rows])),
        )
    return out


if __name__ == "__main__":
    p, ep, h = run_episode(SEED)
    m, tp = tracking_plot(p, ep, h)
    cp = calibration_plot(p, ep, h)

    print("=== Particle filter validation ===")
    print(f"[single episode, seed {SEED}, alpha x{SHOCK_MULT:g} shock at step {SHOCK_STEP}]")
    print(f"  alpha RMSE {m['alpha_rmse']:.2e}  (naive guess-the-mean: {m['alpha_rmse_naive_theta']:.2e})")
    print(f"  kappa RMSE {m['kappa_rmse']:.2e}  (naive guess-the-mean: {m['kappa_rmse_naive_theta']:.2e})")
    print(f"  alpha std: pre-shock {h['alpha_std'][SHOCK_STEP-1]:.2e} -> peak {h['alpha_std'][SHOCK_STEP:SHOCK_STEP+10].max():.2e} "
          f"-> step +20 {h['alpha_std'][SHOCK_STEP+20]:.2e}")

    print("\n[30 seeds, shock x3 at step 25]")
    ms, _ = multi_seed()
    for k in ["alpha_rmse", "alpha_rmse_naive_theta", "kappa_rmse", "kappa_rmse_naive_theta",
              "alpha_cov90", "kappa_cov90", "alpha_cov90_shock_window", "alpha_max_abs_z_shock_window", "min_ess"]:
        print(f"  {k:32s} {ms[k]:.3g}")
    print("\n[30 seeds, NO shock]")
    mn = multi_seed_noshock()
    for k in ["alpha_rmse", "alpha_rmse_naive_theta", "kappa_rmse", "kappa_rmse_naive_theta", "alpha_cov90", "kappa_cov90"]:
        print(f"  {k:32s} {mn[k]:.3g}")
    print("\n[shock-size sweep, 15 seeds each, step 25]")
    sw = shock_sweep()
    for k, v in sw.items():
        print(f"  alpha {k:5s} RMSE {v['alpha_rmse']:.2e}  90%-coverage in shock window {v['cov90_window']:.2f}")

    with open(os.path.join(OUT_DIR, "validation_summary.json"), "w") as f:
        json.dump(dict(single_episode=m, multi_seed_shock=ms, multi_seed_noshock=mn, shock_sweep=sw), f, indent=2)

    print("\nKNOWN LIMITS (deliberately not hidden):")
    print("  - Jump prior covers alpha multipliers 0.5x-6x. Shocks outside that range")
    print("    (e.g. 0.3x, 8x+) collapse the filter (ESS->1, overconfident). Tested, not fixed.")
    print("  - At the shock step ESS is only ~4-8: few particles land near the new truth,")
    print("    so std is noisiest exactly there. Estimates are accurate; treat that std as approximate.")
    print("  - Jumps in kappa are not modelled. Filter model params (theta, lambda, sigma, rho) are KNOWN.")
    print(f"\nPlots written to:\n  {tp}\n  {cp}\n  {os.path.join(OUT_DIR, 'validation_summary.json')}")
