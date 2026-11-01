import json, numpy as np, os
rows = json.load(open(os.path.join("outputs", "stage3c", "fresh_scores.json")))
def get(arm, m):
    return {r["seed"]: r[m]["diff"] for r in rows if r["arm"] == arm}
P = [("filt_full","raw_proxy"), ("oracle","filt_full"), ("filt_full","filt_mean_only"),
     ("filt_mean_bonus","filt_mean_only"), ("filt_full_bonus","filt_full"),
     ("filt_full","filt_no_regime"), ("filt_full","filt_single")]
for m in ("valsel", "last"):
    print(m)
    for a, b in P:
        A, B = get(a, m), get(b, m); s = sorted(set(A) & set(B))
        d = np.array([A[k] - B[k] for k in s]); se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {a:16s}-{b:16s} n={len(d)} mean={d.mean():+7.1f} se={se:6.1f} t={d.mean()/se:+5.2f}")
print("|t| > 2.776 needed (df=4)")