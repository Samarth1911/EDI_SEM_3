"""
compare_arms.py -- Stage 3 ablation report (reads results_base{0|1}<tag>.jsonl only; no training).
UNTESTED. Self-check: see the commands (reproduces the EMA vs no-EMA paired numbers from the oracle runs).
Metrics: valsel = validation-selected checkpoint scored on the test worlds (PRIMARY), last = final checkpoint.
Pairing is by training seed (same training worlds, same test worlds). n is tiny: CIs use Student-t.
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
ARM_ORDER = ["oracle", "raw_proxy", "filt_full", "filt_mean_only", "filt_no_regime", "filt_single",
             "filt_full_bonus", "filt_mean_bonus"]
DEFAULT_PAIRS = [   # (label, A, B) -> reports mean(A - B)
    ("(a) filter vs raw proxy         : filt_full - raw_proxy", "filt_full", "raw_proxy"),
    ("(a) value of perfect info       : oracle - filt_full", "oracle", "filt_full"),
    ("(b) uncertainty features        : filt_full - filt_mean_only", "filt_full", "filt_mean_only"),
    ("(b) bonus, mean-only            : filt_mean_bonus - filt_mean_only", "filt_mean_bonus", "filt_mean_only"),
    ("(b) bonus, with std features    : filt_full_bonus - filt_full", "filt_full_bonus", "filt_full"),
    ("(c) regime probs masked         : filt_full - filt_no_regime", "filt_full", "filt_no_regime"),
    ("(c) K=1 filter instead of IMM   : filt_full - filt_single", "filt_full", "filt_single"),
]


def tcrit(df):
    return TCRIT.get(df, 2.0) if df >= 1 else float("nan")


def load_file(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get(r, m):
    return r["best_val"]["nf"]["diff"] if m == "valsel" else r["last"]["nf"]["diff"]


def get_se(r, m):
    return r["best_val"]["nf"]["se"] if m == "valsel" else r["last"]["nf"]["se"]


def collect(tags, base, episodes, eval_n, by_tag):
    recs = {}
    for tag in tags:
        path = os.path.join(HERE, "outputs", "stage3c", f"results_base{base}{tag}.jsonl")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        for r in load_file(path):
            if r["episodes"] != episodes or r["eval_episodes"] != eval_n:
                continue
            key = f"{r['arm_name']}@{tag}" if by_tag else r["arm_name"]
            recs.setdefault(key, {})[r["seed"]] = r          # a later duplicate (arm, seed) overwrites
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["_abl", "_abl_b1"])
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--baseline", type=int, default=1)
    ap.add_argument("--label-by-tag", action="store_true", help="arm key = arm@tag (to compare runs of the SAME arm)")
    ap.add_argument("--pairs", default=None, help='custom contrasts "A:B,C:D" (keys as printed); reports mean(A-B)')
    ap.add_argument("--out-name", default="ablation_report")
    a = ap.parse_args()

    recs = collect(a.tags, a.baseline, a.episodes, a.eval_episodes, a.label_by_tag)
    if not recs:
        print("no matching records"); return
    keys = [k for k in ARM_ORDER if k in recs] + sorted(k for k in recs if k not in ARM_ORDER)
    out = []
    P = lambda s="": (print(s), out.append(s))

    P(f"episodes={a.episodes}  eval_episodes={a.eval_episodes}  tags={a.tags}  (diff vs TWAP in $, negative = better)")
    for m, title in (("valsel", "PRIMARY: validation-selected checkpoint, scored on test worlds"),
                     ("last", "SECONDARY: final checkpoint")):
        P(f"\n=== per arm vs TWAP -- {title} ===")
        P(f"{'arm':30s} {'n':>2s} {'mean':>8s} {'sd(seeds)':>9s} {'se':>6s} {'95% CI half':>11s} {'world se':>8s}  verdict")
        for k in keys:
            rs = recs[k]; seeds = sorted(rs)
            x = np.array([get(rs[s], m) for s in seeds]); n = len(x)
            sd = x.std(ddof=1) if n > 1 else float("nan")
            se = sd / np.sqrt(n) if n > 1 else float("nan")
            ci = tcrit(n - 1) * se if n > 1 else float("nan")
            wse = float(np.mean([get_se(rs[s], m) for s in seeds]))
            v = "n<3" if n < 3 else ("BETTER than TWAP" if x.mean() + ci < 0 else ("WORSE" if x.mean() - ci > 0 else "n.s."))
            P(f"{k:30s} {n:2d} {x.mean():+8.1f} {sd:9.1f} {se:6.1f} {ci:11.1f} {wse:8.1f}  {v}")
    P("\nworld se = paired se over the shared test worlds (common to all seeds; not reduced by more seeds).")

    if a.pairs:
        pairs = [(f"{p.split(':')[0]} - {p.split(':')[1]}", *p.split(":")) for p in a.pairs.split(",")]
    else:
        pairs = DEFAULT_PAIRS
    for m in ("valsel", "last"):
        P(f"\n=== paired contrasts A - B (negative = A cheaper), by training seed -- {m} ===")
        P(f"{'contrast':66s} {'n':>2s} {'mean':>8s} {'se':>6s} {'t':>6s} {'95% CI half':>11s}  verdict")
        for label, A, B in pairs:
            if A not in recs or B not in recs:
                P(f"{label:66s}  (missing arm)"); continue
            common = sorted(set(recs[A]) & set(recs[B]))
            d = np.array([get(recs[A][s], m) - get(recs[B][s], m) for s in common]); n = len(d)
            if n < 3:
                P(f"{label:66s} {n:2d}  n<3: no test"); continue
            se = d.std(ddof=1) / np.sqrt(n); ci = tcrit(n - 1) * se
            v = "A better" if d.mean() + ci < 0 else ("B better" if d.mean() - ci > 0 else "n.s.")
            P(f"{label:66s} {n:2d} {d.mean():+8.1f} {se:6.1f} {d.mean() / se:+6.2f} {ci:11.1f}  {v}")
    P("\nNOTES: 7 contrasts x 2 metrics -> expect ~1 false 'better' at the 95% level by chance; treat single hits cautiously.")
    P("Test worlds are shared by all arms/seeds (paired comparisons are fair; absolute claims carry the 'world se').")
    P("Configs were chosen on these test worlds earlier: confirm headline numbers with score_fresh.py.")
    p = os.path.join(HERE, "outputs", "stage3c", a.out_name + ".md")
    with open(p, "w") as f:
        f.write("```\n" + "\n".join(out) + "\n```\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()