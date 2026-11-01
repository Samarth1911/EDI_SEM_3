"""
run_policy_comparison.py
-------------------------
PRELIMINARY HEURISTIC showing the belief signal carries useful information --
the actual contribution is the Double DQN agent trained on this belief, which
is next phase, not this one.

Compares Implementation Shortfall (IS, $; lower = better) of:
  TWAP | belief heuristic | belief heuristic w/o uncertainty term | oracle (same rule, true alpha)

Protocol (so the numbers can be trusted):
  * Knobs tuned on TUNE seeds; all reported numbers come from disjoint EVAL seeds.
  * Common random numbers: every policy sees identical market noise per seed,
    so the PAIRED difference vs TWAP is reported alongside raw mean/std.
  * Shock episodes: alpha jump at a random step in [8,40], size uniform in [2,5]x
    (inside the filter's validated jump range). Calm episodes: no shock.
  * Shock and calm are reported SEPARATELY -- never pooled.

Run from project root:  python validation/experiments/run_policy_comparison.py
"""
import sys, os, json, itertools, time
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heuristic_policy import run_policy_episode, HeuristicParams

OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

TUNE_SEED0, EVAL_SEED0 = 5_000, 50_000
N_TUNE_SHOCK, N_TUNE_CALM = 60, 30
N_EVAL = 300
KINDS = ["twap", "belief_mean_only", "belief", "oracle"]
LABEL = {"twap": "TWAP",
         "belief_mean_only": "Belief heuristic (mean only)",
         "belief": "Belief + uncertainty penalty z=1",
         "oracle": "Oracle (same rule, true alpha)"}


def scenario(i, base, shocked):
    """Deterministic scenario draw for episode i (same across all policies)."""
    if not shocked:
        return base + i, None, 1.0
    r = np.random.default_rng(base + 777_000 + i)
    return base + i, int(r.integers(8, 41)), float(r.uniform(2.0, 5.0))


def run_set(kinds, seeds_spec, hp):
    out = {k: [] for k in kinds}
    for seed, step, mult in seeds_spec:
        for k in kinds:
            out[k].append(run_policy_episode(k, seed, step, mult, hp).IS)
    return {k: np.array(v) for k, v in out.items()}


def tune():
    """Tune (gamma, taper_power) for the MEAN-ONLY belief heuristic. The uncertainty
    weight z_unc is NOT tuned to look good: a prior sweep showed z>0 was worse in every
    (gamma, taper) cell, so it is evaluated below as an explicit ablation instead."""
    spec = [scenario(i, TUNE_SEED0, True) for i in range(N_TUNE_SHOCK)] + \
           [scenario(i, TUNE_SEED0 + 5000, False) for i in range(N_TUNE_CALM)]
    best, best_score, table = None, np.inf, []
    for gamma, taper in itertools.product([0.5, 1.0, 1.5, 2.0], [0, 1, 2]):
        hp = HeuristicParams(gamma=gamma, taper_power=taper, z_unc=1.0)
        r = run_set(["twap", "belief_mean_only"], spec, hp)
        score = float(np.mean(r["belief_mean_only"] - r["twap"]))
        table.append((gamma, taper, score))
        if score < best_score:
            best, best_score = hp, score
    return best, best_score, table


def z_ablation(hp):
    """Uncertainty-penalty sweep on the held-out SHOCK eval episodes (other knobs fixed)."""
    spec = [scenario(i, EVAL_SEED0, True) for i in range(N_EVAL)]
    base = run_set(["twap", "belief_mean_only"], spec, hp)
    out = {"z=0 (mean only)": paired(base["belief_mean_only"], base["twap"])[:2]}
    for z in [0.5, 1.0, 2.0, 3.0]:
        r = run_set(["belief"], spec, replace(hp, z_unc=z))
        out[f"z={z:g}"] = paired(r["belief"], base["twap"])[:2]
    return {k: dict(paired_diff_vs_twap=v[0], paired_se=v[1]) for k, v in out.items()}


def paired(a, b):
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), float((d < 0).mean())


def evaluate(hp):
    res = {}
    for name, shocked in [("shock", True), ("calm", False)]:
        spec = [scenario(i, EVAL_SEED0 + (0 if shocked else 9000), shocked) for i in range(N_EVAL)]
        r = run_set(KINDS, spec, hp)
        res[name] = dict(spec=spec, IS=r)
    return res


def summarize(res):
    summ = {}
    for name, d in res.items():
        IS = d["IS"]
        summ[name] = {}
        for k in KINDS:
            m, se, win = paired(IS[k], IS["twap"])
            summ[name][k] = dict(mean=float(IS[k].mean()), std=float(IS[k].std(ddof=1)),
                                 paired_diff_vs_twap=m, paired_se=se, frac_better_than_twap=win)
        # timing breakdown (shock only)
        if name == "shock":
            steps = np.array([s[1] for s in d["spec"]])
            bins = {"early (8-19)": steps < 20, "mid (20-31)": (steps >= 20) & (steps < 32), "late (32-40)": steps >= 32}
            summ[name]["by_timing"] = {}
            for b, mask in bins.items():
                summ[name]["by_timing"][b] = {"n": int(mask.sum())}
                for k in ["belief", "belief_mean_only", "oracle"]:
                    m, se, _ = paired(IS[k][mask], IS["twap"][mask])
                    summ[name]["by_timing"][b][k] = dict(paired_diff_vs_twap=m, paired_se=se)
    return summ


def plot(res, summ, hp):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cols = {"twap": "gray", "belief_mean_only": "tab:blue", "belief": "tab:cyan", "oracle": "tab:green"}
    for ax, name in zip(axes[:2], ["shock", "calm"]):
        ms = [summ[name][k]["mean"] for k in KINDS]; ss = [summ[name][k]["std"] for k in KINDS]
        ax.bar(range(4), ms, yerr=ss, color=[cols[k] for k in KINDS], capsize=4)
        ax.set_xticks(range(4)); ax.set_xticklabels(["TWAP", "Belief\n(mean only)", "Belief\n+unc. z=1", "Oracle"])
        lo = min(m - s for m, s in zip(ms, ss)); ax.set_ylim(max(0, lo * 0.9), max(m + s for m, s in zip(ms, ss)) * 1.05)
        ax.set_ylabel("Implementation Shortfall ($)")
        ax.set_title(f"{name} episodes: mean IS +/- 1 std (n={N_EVAL})\n(std is dominated by price noise -- see paired panel)")
    ax = axes[2]
    for k in ["belief_mean_only", "oracle"]:
        d = res["shock"]["IS"][k] - res["shock"]["IS"]["twap"]
        ax.hist(d, bins=30, alpha=0.55, color=cols[k], label=f"{LABEL[k]}  (mean {d.mean():+.0f}, {100*(d<0).mean():.0f}% beat TWAP)")
    ax.axvline(0, color="k", lw=1); ax.legend(fontsize=8)
    ax.set_xlabel("IS(policy) - IS(TWAP), same market path ($, negative = better)")
    ax.set_title("Paired difference vs TWAP, shock episodes")
    fig.suptitle("PRELIMINARY heuristic -- shows the belief signal is informative; the contribution is the Double DQN (next phase)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, "03_policy_comparison.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


if __name__ == "__main__":
    t0 = time.time()
    print("Tuning (gamma, taper) on separate seeds...")
    hp, score, table = tune()
    print(f"  chosen: gamma={hp.gamma}, taper_power={hp.taper_power}, s in [{hp.s_min},{hp.s_max}]  (tune-set mean IS gain vs TWAP {score:+.1f})")
    res = evaluate(hp)
    summ = summarize(res)
    for name in ["shock", "calm"]:
        print(f"\n=== {name.upper()} episodes (n={N_EVAL}, disjoint eval seeds) ===")
        print(f"{'policy':34s} {'mean IS':>9s} {'std IS':>8s} | {'paired vs TWAP':>15s} {'(+/-SE)':>8s} {'beats TWAP':>10s}")
        for k in KINDS:
            s_ = summ[name][k]
            print(f"{LABEL[k]:34s} {s_['mean']:9.1f} {s_['std']:8.1f} | {s_['paired_diff_vs_twap']:+15.1f} {s_['paired_se']:8.1f} {100*s_['frac_better_than_twap']:9.0f}%")
    print("\nShock episodes by timing (paired diff vs TWAP, $, negative = better):")
    for b_, d in summ["shock"]["by_timing"].items():
        print(f"  {b_:14s} n={d['n']:3d}  belief(mean) {d['belief_mean_only']['paired_diff_vs_twap']:+7.1f}  belief+unc {d['belief']['paired_diff_vs_twap']:+7.1f}  oracle {d['oracle']['paired_diff_vs_twap']:+7.1f}")
    print("\nUncertainty-penalty ablation, shock episodes (paired diff vs TWAP, $; more negative = better):")
    abl = z_ablation(hp)
    for k, v in abl.items():
        print(f"  {k:18s} {v['paired_diff_vs_twap']:+8.1f}  (+/-{v['paired_se']:.1f})")
    gap = summ["shock"]["belief_mean_only"]["paired_diff_vs_twap"] - summ["shock"]["oracle"]["paired_diff_vs_twap"]
    print(f"\nBelief-vs-oracle gap on shock episodes: {gap:+.2f} $ (SE ~{summ['shock']['belief_mean_only']['paired_se']:.1f} on each)")
    path = plot(res, summ, hp)
    with open(os.path.join(OUT, "policy_comparison_summary.json"), "w") as f:
        json.dump(dict(knobs=hp.__dict__, tune_gain=score, summary=summ, uncertainty_ablation=abl,
                       tune_table=[dict(gamma=a_, taper_power=b_, mean_is_diff=c_) for a_, b_, c_ in table]), f, indent=2)
    print(f"\nPlot: {path}\n({time.time()-t0:.0f}s)")