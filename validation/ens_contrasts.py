"""
ens_contrasts.py -- ablation contrasts from the arrays saved by score_ensemble.py. NO TRAINING. UNTESTED until you run it.
Run from EDI/validation (venv active) after score_ensemble.py has been run for each arm you want:
    foreach ($a in "raw_proxy","filt_full","filt_mean_only","filt_no_regime","filt_single","filt_mean_bonus") { python score_ensemble.py --arm $a }
    python ens_contrasts.py

Two kinds of inference, kept separate (read the difference carefully):
  WORLD-level : paired over worlds, ensemble vs ensemble. Conditions on the ONE ensemble of 5 trained agents per arm, so the
                se only reflects world sampling. It says nothing about how a retrained arm would behave.
  SEED-level  : each member is scored by its mean over the worlds; Welch t-test over the 5 members per arm. Includes training
                variability (this is the same kind of inference as the earlier arm contrasts, n = 5 vs 5).
Seven contrasts, listed in advance (not searched). Holm adjustment over the 7 within each (level, block).
All costs are noise-free differences vs TWAP in $ (negative = better); a contrast X - Y < 0 means X cheaper.
"""
import argparse, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "outputs", "stage3c"))
a = ap.parse_args()
ARMS = ["oracle", "raw_proxy", "filt_full", "filt_mean_only", "filt_no_regime", "filt_single", "filt_full_bonus", "filt_mean_bonus"]
CONTRASTS = [("filt_full", "raw_proxy", "filtered belief vs raw proxy"),
             ("filt_full", "filt_mean_only", "uncertainty features (no bonus)"),
             ("filt_full_bonus", "filt_full", "exploration bonus (with std features)"),
             ("filt_mean_bonus", "filt_mean_only", "exploration bonus (mean only)"),
             ("filt_full", "filt_no_regime", "regime layer"),
             ("filt_full", "filt_single", "IMM filter vs single-regime filter"),
             ("oracle", "filt_full", "oracle input vs filtered input")]
Z = {}
for arm in ARMS:
    p = os.path.join(a.dir, f"ens_nf_{arm}.npz")
    if os.path.exists(p):
        Z[arm] = np.load(p)
    else:
        print(f"(missing {os.path.basename(p)}: run score_ensemble.py --arm {arm})")
assert Z, "no ens_nf_*.npz files found"
ref = next(iter(Z.values()))
for arm, z in Z.items():
    for blk in ("test", "fresh"):
        assert np.array_equal(z[f"{blk}_seeds"], ref[f"{blk}_seeds"]), f"{arm}/{blk}: different worlds"
        assert np.allclose(z[f"{blk}_twap"], ref[f"{blk}_twap"]), f"{arm}/{blk}: TWAP cost differs"


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def world_test(d):
    se = d.std(ddof=1) / math.sqrt(len(d)); t = d.mean() / se
    return d.mean(), se, 2 * stats.t.sf(abs(t), len(d) - 1)


for blk in ("test", "fresh"):
    n = len(ref[f"{blk}_seeds"])
    print(f"\n================ {blk.upper()} worlds ({n}) ================")
    print(f"{'arm':16s} {'ENSEMBLE vs TWAP (se)':>24s} {'members: mean':>14s} {'sd':>6s} {'min..max':>16s}")
    for arm, z in Z.items():
        tw, en, mem = z[f"{blk}_twap"], z[f"{blk}_ens"], z[f"{blk}_mem"]
        e = en - tw; s = (mem - tw).mean(axis=1)
        print(f"{arm:16s} {e.mean():+14.1f} ({e.std(ddof=1) / math.sqrt(n):5.1f}) {s.mean():+14.1f} {s.std(ddof=1):6.1f} "
              f"{s.min():+8.1f}..{s.max():+6.1f}")
    rows = []
    for x, y, label in CONTRASTS:
        if x in Z and y in Z:
            ex = Z[x][f"{blk}_ens"] - Z[y][f"{blk}_ens"]
            sx = (Z[x][f"{blk}_mem"] - Z[x][f"{blk}_twap"]).mean(axis=1)
            sy = (Z[y][f"{blk}_mem"] - Z[y][f"{blk}_twap"]).mean(axis=1)
            tt, ps = stats.ttest_ind(sx, sy, equal_var=False)
            wm, wse, wp = world_test(ex)
            rows.append((x, y, label, wm, wse, wp, sx.mean() - sy.mean(), tt, ps))
    if not rows:
        continue
    wadj = holm(np.array([r[5] for r in rows])); sadj = holm(np.array([r[8] for r in rows]))
    print(f"\n  {'contrast (X - Y)':52s} {'WORLD-level diff (se) p_holm':>30s} {'SEED-level diff (Welch t) p_holm':>34s}")
    for r, wa, sa in zip(rows, wadj, sadj):
        print(f"  {r[0] + ' - ' + r[1] + ' [' + r[2] + ']':52s} {r[3]:+9.1f} ({r[4]:4.1f}) {wa:8.3f} {r[6]:+14.1f} ({r[7]:+5.2f}) {sa:8.3f}")
print("\nWORLD-level conditions on one ensemble per arm (world sampling only). SEED-level includes training variability but has "
      "n = 5 per arm.\nAn effect that is significant only at WORLD level is a statement about these trained agents, not about the arm.")