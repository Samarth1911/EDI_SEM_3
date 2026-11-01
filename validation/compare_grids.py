"""
compare_grids.py -- compare one arm across two Stage 3 results files. UNTESTED until you run it. Run from EDI/validation:
    python compare_grids.py --a outputs\\stage3c\\results_base1_abl.jsonl --b outputs\\stage3c\\results_base1_abl_g12.jsonl --arm oracle

Reads the records written by run_stage3c_multiseed.py. Metrics (all on the 200 test worlds, vs TWAP, $; negative = better):
  VAL-SEL nf : record['best_val']['nf']['diff']  (checkpoint chosen on validation worlds, scored on test worlds)  PRIMARY
  LAST nf    : record['last']['nf']['diff']      (final checkpoint)                                              SECONDARY
  CVaR95 diff: full-IS CVaR95 minus TWAP's, for both checkpoints
Welch t-test over training seeds (n = number of finished seeds per file). Small n: treat p-values as rough.
Difference reported as B - A; negative = B better.
"""
import argparse, json, os
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--arm", default="oracle")
ap.add_argument("--episodes", type=int, default=30000)
ap.add_argument("--eval-episodes", type=int, default=200)
a = ap.parse_args()


def load(path):
    recs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["arm_name"] == a.arm and r["episodes"] == a.episodes and r["eval_episodes"] == a.eval_episodes:
                recs[r["seed"]] = r                       # last record wins if a seed was rerun
    return [recs[s] for s in sorted(recs)]


A, B = load(a.a), load(a.b)
assert A and B, f"no matching records (arm={a.arm}, episodes={a.episodes}, eval={a.eval_episodes}); A={len(A)} B={len(B)}"
metrics = {
    "VAL-SEL nf (PRIMARY)": lambda r: r["best_val"]["nf"]["diff"],
    "LAST nf": lambda r: r["last"]["nf"]["diff"],
    "VAL-SEL CVaR95 diff": lambda r: r["best_val"]["full"]["cvar95"] - r["best_val"]["full"]["base_cvar95"],
    "LAST CVaR95 diff": lambda r: r["last"]["full"]["cvar95"] - r["last"]["full"]["base_cvar95"],
    "val-sel episode": lambda r: r.get("best_val_ep", float("nan")),
    "collapse frac": lambda r: r.get("val_collapse_frac", float("nan")),
}
print(f"arm={a.arm}  A={os.path.basename(a.a)} (n={len(A)} seeds {[r['seed'] for r in A]})  "
      f"B={os.path.basename(a.b)} (n={len(B)} seeds {[r['seed'] for r in B]})")
print(f"\n{'metric':24s} {'A mean (sd)':>16s} {'B mean (sd)':>16s} {'B - A':>9s} {'Welch t':>8s} {'p':>7s}")
for name, f in metrics.items():
    x = np.array([f(r) for r in A], float); y = np.array([f(r) for r in B], float)
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
    if len(x) > 1 and len(y) > 1 and np.isfinite(x).all() and np.isfinite(y).all():
        t, p = stats.ttest_ind(y, x, equal_var=False)
    else:
        t = p = float("nan")
    print(f"{name:24s} {x.mean():8.1f} ({sd(x):5.1f}) {y.mean():8.1f} ({sd(y):5.1f}) {y.mean() - x.mean():+9.1f} {t:+8.2f} {p:7.3f}")
print("\nPer-seed VAL-SEL nf  A:", np.round([r["best_val"]["nf"]["diff"] for r in A], 1),
      " B:", np.round([r["best_val"]["nf"]["diff"] for r in B], 1))
print("Per-seed LAST nf     A:", np.round([r["last"]["nf"]["diff"] for r in A], 1),
      " B:", np.round([r["last"]["nf"]["diff"] for r in B], 1))
print("\nPre-registered criterion: B counts as an improvement on the PRIMARY metric only if B - A is clearly negative "
      "relative to seed spread.\nNote: B uses a different action grid, so its CE ceiling differs; also compare each to its own grid's CE reference.")