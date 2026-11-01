"""
report_ablation.py -- Stage 4: the single (a)/(b)/(c) ablation report. NO TRAINING, NO SIMULATION. NOT RUN by its author.
Run from EDI/validation (venv active) after score_ensemble.py has produced ens_nf_<arm>.npz for all 8 arms:
    python report_ablation.py                           (Gaussian, 8 arms)
    python report_ablation.py --obs-kind spread_hawkes  (Hawkes, 4 arms: 3 contrasts; needs ens_nf_<arm>_hawkes.npz and results_base1_hawkes.jsonl)
Writes outputs/stage4/ablation_report[_hawkes].md and .json.

Combines (nothing new is estimated beyond what those scripts already do):
  * SINGLE AGENTS  (results_base1<tag>.jsonl, as compare_arms.py): validation-selected checkpoint on the TEST worlds,
    contrast = paired by training seed (n = 5 pairs), Student-t. Holm over the 7 contrasts is added here (compare_arms has none).
  * ENSEMBLES      (ens_nf_<arm>.npz, as ens_contrasts.py): WORLD-level paired t (one ensemble per arm: world sampling only) and
    SEED-level Welch t over the 5 members (includes training variability), Holm over the 7 within each (level, block).
All costs: noise-free difference vs TWAP in $, negative = cheaper; contrast X - Y < 0 means X cheaper.
The 7 contrasts are the pre-listed ones from ens_contrasts.py / compare_arms.py, only regrouped:
  (a) information: filtered vs raw proxy; oracle vs filtered   (b) features: uncertainty; bonus x2   (c) regime: layer; IMM vs K=1.

SELF-CHECK: the ensemble numbers are compared with the values printed in STAGE3_REVISIT_HANDOFF.md (b)3 and (b)5 (tolerance =
rounding of the printed value). If any differ, the report still gets written but is headed 'HANDOFF CHECK FAILED' and the
exit code is 1. --skip-handoff-check disables it (use only for a different checkpoint set / world set).

VERDICT RULE (my convention, not from the handoffs): based on the ensemble SEED-level Holm p in BOTH world blocks:
  SUPPORTED = p < 0.05 in test and fresh, same sign;  BORDERLINE = p < 0.05 in exactly one;  NOT ESTABLISHED = neither.
"""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy import stats
import compare_arms as CA

ARMS = ["oracle", "raw_proxy", "filt_full", "filt_mean_only", "filt_no_regime", "filt_single", "filt_full_bonus", "filt_mean_bonus"]
BLKS = ("test", "fresh")
# (X, Y, group, label); order = ens_contrasts.CONTRASTS
CONTRASTS = [("filt_full", "raw_proxy", "a", "filtered belief vs raw proxy"),
             ("filt_full", "filt_mean_only", "b", "uncertainty features (no bonus)"),
             ("filt_full_bonus", "filt_full", "b", "exploration bonus (with std features)"),
             ("filt_mean_bonus", "filt_mean_only", "b", "exploration bonus (mean only)"),
             ("filt_full", "filt_no_regime", "c", "regime layer"),
             ("filt_full", "filt_single", "c", "IMM filter vs single-regime filter"),
             ("oracle", "filt_full", "a", "oracle input vs filtered input")]
GROUP_TITLE = {"a": "(a) Does the information help? (filter vs raw proxy; perfect info vs filter)",
               "b": "(b) Do uncertainty features and the exploration bonus help?",
               "c": "(c) Does the regime structure help? (regime probabilities; IMM vs K=1 filter)"}

# ---- values printed in STAGE3_REVISIT_HANDOFF.md: (b)3 table and (b)5 table (strings so the rounding tolerance is known)
H_ENS = {"test": dict(oracle=-82.2, raw_proxy=-45.9, filt_full=-76.9, filt_mean_only=-73.2, filt_no_regime=-57.9,
                      filt_single=-69.3, filt_full_bonus=-87.9, filt_mean_bonus=-67.7),
         "fresh": dict(oracle=-89.2, raw_proxy=-50.6, filt_full=-85.3, filt_mean_only=-82.4, filt_no_regime=-54.6,
                       filt_single=-73.0, filt_full_bonus=-93.7, filt_mean_bonus=-79.0)}
H_MEM = {"test": dict(oracle=-67.7, raw_proxy=-9.7, filt_full=-61.0, filt_mean_only=-63.4, filt_no_regime=-49.0,
                      filt_single=-50.1, filt_full_bonus=-73.3, filt_mean_bonus=-56.2),
         "fresh": dict(oracle=-76.9, raw_proxy=-11.3, filt_full=-70.0, filt_mean_only=-73.0, filt_no_regime=-48.5,
                       filt_single=-54.9, filt_full_bonus=-82.0, filt_mean_bonus=-68.1)}
# (world diff, seed diff, seed Holm p as printed)
H_CON = {("filt_full", "raw_proxy"): dict(test=(-31.0, -51.3, "0.066"), fresh=(-34.7, -58.7, "0.032")),
         ("filt_full", "filt_mean_only"): dict(test=(-3.7, 2.5, "0.95"), fresh=(-2.9, 3.0, "1.0")),
         ("filt_full_bonus", "filt_full"): dict(test=(-11.0, -12.3, "0.26"), fresh=(-8.5, -12.0, "0.22")),
         ("filt_mean_bonus", "filt_mean_only"): dict(test=(5.4, 7.2, "0.95"), fresh=(3.4, 4.9, "1.0")),
         ("filt_full", "filt_no_regime"): dict(test=(-19.0, -12.0, "0.89"), fresh=(-30.6, -21.4, "0.53")),
         ("filt_full", "filt_single"): dict(test=(-7.6, -10.9, "0.54"), fresh=(-12.3, -15.0, "0.22")),
         ("oracle", "filt_full"): dict(test=(-5.3, -6.7, "0.89"), fresh=(-4.0, -6.9, "0.77"))}

ap = argparse.ArgumentParser()
ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
ap.add_argument("--dir", default=os.path.join(HERE, "outputs", "stage3c"))
ap.add_argument("--out", default=os.path.join(HERE, "outputs", "stage4"))
ap.add_argument("--tags", nargs="+", default=None)
ap.add_argument("--episodes", type=int, default=30000)
ap.add_argument("--eval-episodes", type=int, default=200)
ap.add_argument("--baseline", type=int, default=1)
ap.add_argument("--skip-handoff-check", action="store_true")
a = ap.parse_args()
hawkes = a.obs_kind == "spread_hawkes"
SUF = "_hawkes" if hawkes else ""
a.tags = a.tags or (["_hawkes"] if hawkes else ["_abl", "_abl_b1"])
if hawkes:   # only these 4 arms were trained on Hawkes observations; contrasts needing other arms are dropped automatically
    ARMS = ["oracle", "raw_proxy", "filt_full", "filt_mean_only"]
# values printed in STAGE3_REVISIT_HANDOFF.md item 8 (Hawkes ensembles, 5 seeds); no earlier contrast values exist to check against
H_ENS_HK = {"test": dict(oracle=-72.4, raw_proxy=-70.8, filt_full=-77.8, filt_mean_only=-61.8),
            "fresh": dict(oracle=-81.0, raw_proxy=-74.0, filt_full=-83.4, filt_mean_only=-70.2)}
H_MEM_HK = {"test": dict(oracle=-60.9, raw_proxy=-31.5, filt_full=-56.1, filt_mean_only=-39.1),
            "fresh": dict(oracle=-66.9, raw_proxy=-36.0, filt_full=-64.5, filt_mean_only=-48.0)}
HE, HM = (H_ENS_HK, H_MEM_HK) if hawkes else (H_ENS, H_MEM)
HREF = "STAGE3_REVISIT_HANDOFF.md item 8" if hawkes else "STAGE3_REVISIT_HANDOFF.md (b)3 and (b)5"
os.makedirs(a.out, exist_ok=True)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def t_test_1samp(d):
    se = d.std(ddof=1) / math.sqrt(len(d)); t = d.mean() / se
    return float(d.mean()), float(se), float(t), float(2 * stats.t.sf(abs(t), len(d) - 1))


# ---------------------------------------------------------------- load
Z = {}
for arm in ARMS:
    p = os.path.join(a.dir, f"ens_nf_{arm}{SUF}.npz")
    if os.path.exists(p):
        Z[arm] = np.load(p)
    else:
        print(f"(missing {os.path.basename(p)}: run score_ensemble.py --arm {arm})")
if not Z:
    sys.exit("no ens_nf_*.npz files found")
ref = next(iter(Z.values()))
for arm, z in Z.items():
    for blk in BLKS:
        if not np.array_equal(z[f"{blk}_seeds"], ref[f"{blk}_seeds"]) or not np.allclose(z[f"{blk}_twap"], ref[f"{blk}_twap"]):
            sys.exit(f"{arm}/{blk}: different worlds or TWAP cost than the other arms; the contrasts would not be paired")
N_W = {blk: len(ref[f"{blk}_seeds"]) for blk in BLKS}

recs = {}
for tag in a.tags:
    p = os.path.join(a.dir, f"results_base{a.baseline}{tag}.jsonl")
    if not os.path.exists(p):
        print(f"(missing {p})"); continue
    for r in CA.load_file(p):
        if r["episodes"] == a.episodes and r["eval_episodes"] == a.eval_episodes:
            recs.setdefault(r["arm_name"], {})[r["seed"]] = r

# ---------------------------------------------------------------- per-arm numbers
arm_stats = {}
for arm, z in Z.items():
    d = {}
    for blk in BLKS:
        tw, en, mem = z[f"{blk}_twap"], z[f"{blk}_ens"], z[f"{blk}_mem"]
        e = en - tw; s = (mem - tw).mean(axis=1)
        d[blk] = dict(ens=float(e.mean()), ens_se=float(e.std(ddof=1) / math.sqrt(len(e))),
                      mem_mean=float(s.mean()), mem_sd=float(s.std(ddof=1)), n_members=int(len(s)))
    if arm in recs:
        x = np.array([CA.get(recs[arm][s], "valsel") for s in sorted(recs[arm])])
        d["single_test"] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else float("nan"), n=int(len(x)))
    arm_stats[arm] = d

# ---------------------------------------------------------------- contrasts
res = []                               # one dict per available contrast
for x, y, g, label in CONTRASTS:
    if x in Z and y in Z:
        res.append(dict(x=x, y=y, group=g, label=label))
for blk in BLKS:
    for r in res:
        ex = Z[r["x"]][f"{blk}_ens"] - Z[r["y"]][f"{blk}_ens"]
        sx = (Z[r["x"]][f"{blk}_mem"] - Z[r["x"]][f"{blk}_twap"]).mean(axis=1)
        sy = (Z[r["y"]][f"{blk}_mem"] - Z[r["y"]][f"{blk}_twap"]).mean(axis=1)
        wm, wse, _, wp = t_test_1samp(ex)
        tt, sp = stats.ttest_ind(sx, sy, equal_var=False)
        r[blk] = dict(world_diff=wm, world_se=wse, world_p=wp, seed_diff=float(sx.mean() - sy.mean()),
                      seed_t=float(tt), seed_p=float(sp))
    wadj = holm([r[blk]["world_p"] for r in res]); sadj = holm([r[blk]["seed_p"] for r in res])
    for r, wa, sa in zip(res, wadj, sadj):
        r[blk]["world_holm"], r[blk]["seed_holm"] = float(wa), float(sa)

sing = [r for r in res if r["x"] in recs and r["y"] in recs and len(set(recs[r["x"]]) & set(recs[r["y"]])) >= 3]
for r in sing:
    common = sorted(set(recs[r["x"]]) & set(recs[r["y"]]))
    d = np.array([CA.get(recs[r["x"]][s], "valsel") - CA.get(recs[r["y"]][s], "valsel") for s in common])
    m, se, t, p = t_test_1samp(d)
    r["single"] = dict(n=len(d), diff=m, se=se, t=t, p=p)
if sing:
    for r, pa in zip(sing, holm([r["single"]["p"] for r in sing])):
        r["single"]["holm"] = float(pa)


def verdict(r):
    pt, pf = r["test"]["seed_holm"], r["fresh"]["seed_holm"]
    k = int(pt < 0.05) + int(pf < 0.05)
    if k == 2 and np.sign(r["test"]["seed_diff"]) != np.sign(r["fresh"]["seed_diff"]):
        return "INCONSISTENT SIGN"
    return {2: "SUPPORTED", 1: "BORDERLINE", 0: "NOT ESTABLISHED"}[k]


for r in res:
    r["verdict"] = verdict(r)

# ---------------------------------------------------------------- handoff check
checks = []


def chk(name, got, want, tol):
    checks.append((name, abs(got - want) <= tol, f"got {got:+.3f} vs handoff {want:+.3f}"))


if not a.skip_handoff_check:
    for blk in BLKS:
        for arm in Z:
            chk(f"ensemble vs TWAP [{blk}/{arm}]", arm_stats[arm][blk]["ens"], HE[blk][arm], 0.06)
            chk(f"mean of members vs TWAP [{blk}/{arm}]", arm_stats[arm][blk]["mem_mean"], HM[blk][arm], 0.06)
        for r in ([] if hawkes else res):   # contrast values were only printed for the Gaussian arms
            w, s, ptxt = H_CON[(r["x"], r["y"])][blk]
            tag = f"{r['x']} - {r['y']} [{blk}]"
            chk(f"WORLD diff {tag}", r[blk]["world_diff"], w, 0.06)
            chk(f"SEED diff {tag}", r[blk]["seed_diff"], s, 0.06)
            dec = len(ptxt.split(".")[1])
            chk(f"SEED Holm p {tag}", r[blk]["seed_holm"], float(ptxt), 0.5 * 10 ** (-dec) + 1e-9)
failed = [c for c in checks if not c[1]]

# ---------------------------------------------------------------- report
out = []
P = lambda s="": (print(s), out.append(s))
if a.skip_handoff_check:
    P("HANDOFF CHECK SKIPPED (--skip-handoff-check)")
elif failed:
    P(f"HANDOFF CHECK FAILED: {len(failed)} of {len(checks)} values differ from {HREF} -- do not quote this report")
    for n, _, d in failed:
        P(f"   FAIL {n}: {d}")
else:
    P(f"handoff check: all {len(checks)} values match {HREF}")
P(f"\nSTAGE 4 ABLATION REPORT ({a.obs_kind}) -- noise-free cost difference vs TWAP in $, negative = cheaper; contrast X - Y < 0 means X cheaper.")
P(f"worlds: test {N_W['test']}, fresh {N_W['fresh']}; single agents: validation-selected checkpoints on the test worlds "
  f"(tags {a.tags}, {a.episodes} episodes).")

P("\n=== arms ===")
P(f"{'arm':16s} {'single agents (test): mean sd n':>34s} {'ENSEMBLE test (se)':>20s} {'ENSEMBLE fresh (se)':>21s} {'members mean test/fresh':>25s}")
for arm in ARMS:
    if arm not in arm_stats:
        continue
    s = arm_stats[arm]; st = s.get("single_test")
    sc = f"{st['mean']:+8.1f} {st['sd']:6.1f} {st['n']:3d}" if st else "n/a"
    P(f"{arm:16s} {sc:>34s} {s['test']['ens']:+12.1f} ({s['test']['ens_se']:4.1f}) {s['fresh']['ens']:+12.1f} ({s['fresh']['ens_se']:4.1f})"
      f" {s['test']['mem_mean']:+11.1f} /{s['fresh']['mem_mean']:+8.1f}")

for g in ("a", "b", "c"):
    rows = [r for r in res if r["group"] == g]
    if not rows:
        continue
    P(f"\n=== {GROUP_TITLE[g]} ===")
    for r in rows:
        P(f"\n  {r['x']} - {r['y']}  [{r['label']}]   VERDICT: {r['verdict']}")
        if "single" in r:
            s = r["single"]
            P(f"    single agents, paired by training seed (test worlds, n={s['n']}): {s['diff']:+7.1f} (se {s['se']:.1f}, t {s['t']:+.2f}, Holm p {s['holm']:.3f})")
        else:
            P("    single agents: n/a (missing arm or fewer than 3 common seeds)")
        for blk in BLKS:
            b = r[blk]
            P(f"    ensemble {blk:5s}  WORLD-level {b['world_diff']:+7.1f} (se {b['world_se']:.1f}, Holm p {b['world_holm']:.3f})"
              f"   SEED-level Welch {b['seed_diff']:+7.1f} (t {b['seed_t']:+.2f}, Holm p {b['seed_holm']:.3f})")

P("\n=== summary (ensemble SEED-level Holm p, test / fresh) ===")
for r in res:
    P(f"  {r['x'] + ' - ' + r['y']:36s} {r['test']['seed_diff']:+7.1f} / {r['fresh']['seed_diff']:+7.1f}   "
      f"p {r['test']['seed_holm']:.3f} / {r['fresh']['seed_holm']:.3f}   {r['verdict']}")

P("\nHOW TO READ")
P("- WORLD-level conditions on ONE ensemble of 5 agents per arm: its se is world sampling only. A world-level effect is a statement "
  "about these trained agents, not about the arm. SEED-level includes training variability but has n = 5 per arm.")
P("- Holm is over the contrasts available (7 Gaussian, 3 Hawkes) within each (level, block). Test and fresh worlds are not independent evidence: they re-score the "
  "same agents. The single-agent column uses the test worlds only and its Holm p is new here (no earlier number to compare).")
P("- The ensemble idea arose after test and fresh results were seen and is not pre-registered: ensemble contrasts are exploratory. "
  "The 7 contrasts themselves were listed in advance.")
P("- The VERDICT is a rule of this script (see its docstring), not a result of the earlier handoffs.")
P("- Oracle vs filtered compares arms whose inputs differ; the oracle arm has no posterior std, so read it next to filt_full only.")
P("- Test worlds were also used earlier to choose configurations; fresh worlds are the cleaner check for absolute numbers.")
if hawkes:
    P("- HAWKES: 4 arms only (oracle, raw_proxy, filt_full, filt_mean_only), so 3 of the 7 contrasts exist and Holm is over 3 (less strict). "
      "Seeds 3-4 were added after the 3-seed results were seen. The filter is overconfident on this arm. Handoff check covers the arm values only.")
with open(os.path.join(a.out, f"ablation_report{SUF}.md"), "w") as f:
    f.write("```\n" + "\n".join(out) + "\n```\n")
with open(os.path.join(a.out, f"ablation_report{SUF}.json"), "w") as f:
    json.dump(dict(arms=arm_stats, contrasts=res, n_worlds=N_W, handoff_check=dict(
        skipped=a.skip_handoff_check, n=len(checks), failed=[dict(name=n, detail=d) for n, _, d in failed])), f, indent=1)
print("wrote", a.out)
sys.exit(1 if failed else 0)