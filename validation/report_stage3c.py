"""
report_stage3c.py -- read-only summary of outputs/stage3c/results_base*.jsonl (written by run_stage3c_multiseed.py).
NOT RUN by its author (syntax-checked only).  Run from EDI/validation:

    python report_stage3c.py                    # every results file
    python report_stage3c.py _ts2000 base1      # only files whose tag or name matches ('base1' = untagged, baseline ON)

For every finished (arm, seed) it prints two TEST-world numbers (paired difference vs TWAP in $, negative = better):
  LAST     the final checkpoint (the driver's primary metric)
  VAL-SEL  the checkpoint chosen by validation score (the driver calls it 'best_val'), scored on the same test worlds
Per arm it then gives the mean across training seeds, the sd across seeds and the s.e. of that mean.
The per-seed 'se' is world noise only (paired s.e. over the test worlds); the across-seed sd is training noise.

CAVEAT: VAL-SEL is a fair held-out number only if the validation worlds used inside train_v3 are DISJOINT from the test
worlds (EVAL_BASE ...). That was not checked here (the training modules were not available to the author of this file).
With disjoint sets, picking the checkpoint on validation is early stopping: it is optimistic on the VALIDATION score
(winner's curse) but not on the test score.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OD = os.path.join(HERE, "outputs", "stage3c")


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def _mult(r, key):
    m = r["last"].get("mult", {}).get(key)
    return float(m[0]) if m else float("nan")


def report(name, recs):
    print(f"\n=== {name}  ({len(recs)} finished runs) ===")
    print(f"{'arm':16s} {'seed':>4s} {'eps':>6s} {'nTest':>5s} | {'LAST':>8s} {'se':>6s} | "
          f"{'VAL-SEL':>8s} {'se':>6s} {'@ep':>6s} | {'collapse':>8s} {'secs':>5s}")
    for r in sorted(recs, key=lambda r: (r["arm_name"], r["episodes"], r["eval_episodes"], r["seed"])):
        L, B = r["last"]["nf"], r["best_val"]["nf"]
        print(f"{r['arm_name']:16s} {r['seed']:4d} {r['episodes']:6d} {r['eval_episodes']:5d} | "
              f"{L['diff']:+8.1f} {L['se']:6.1f} | {B['diff']:+8.1f} {B['se']:6.1f} {int(r['best_val_ep']):6d} | "
              f"{r['val_collapse_frac']:8.2f} {r['secs']:5.0f}")

    groups = {}
    for r in recs:
        groups.setdefault((r["arm_name"], r["episodes"], r["eval_episodes"]), []).append(r)
    print(f"\n{'arm':16s} {'n':>2s} | {'LAST mean':>9s} {'sd':>6s} {'se':>6s} {'<0':>5s} | "
          f"{'VAL-SEL mean':>12s} {'sd':>6s} {'se':>6s} {'<0':>5s} | mult calm/str (LAST)")
    for (arm, _ep, _ne), rs in sorted(groups.items()):
        n = len(rs)
        last = np.array([r["last"]["nf"]["diff"] for r in rs], dtype=float)
        sel = np.array([r["best_val"]["nf"]["diff"] for r in rs], dtype=float)
        mc = float(np.mean([_mult(r, "0") for r in rs]))
        ms = float(np.mean([_mult(r, "1") for r in rs]))
        print(f"{arm:16s} {n:2d} | {last.mean():+9.1f} {_sd(last):6.1f} {_sd(last) / np.sqrt(n):6.1f} "
              f"{str(int((last < 0).sum())) + '/' + str(n):>5s} | {sel.mean():+12.1f} {_sd(sel):6.1f} "
              f"{_sd(sel) / np.sqrt(n):6.1f} {str(int((sel < 0).sum())) + '/' + str(n):>5s} | {mc:6.3f}/{ms:6.3f}")


def main():
    want = set(sys.argv[1:])
    files = sorted(glob.glob(os.path.join(OD, "results_base*.jsonl")))
    if not files:
        print(f"no results_base*.jsonl in {OD}")
        return
    shown = 0
    for path in files:
        name = os.path.basename(path)[len("results_"):-len(".jsonl")]     # e.g. base1_ts2000
        tag = name[len("base0"):]                                          # e.g. _ts2000 ('' if untagged)
        if want and name not in want and tag not in want:
            continue
        recs = load(path)
        if recs:
            report(name, recs)
            shown += 1
    if not shown:
        print("no file matched", sorted(want))
    print("\nnegative = better than TWAP.  VAL-SEL is a fair test estimate only if validation and test worlds are "
          "disjoint (see the docstring).")


if __name__ == "__main__":
    main()