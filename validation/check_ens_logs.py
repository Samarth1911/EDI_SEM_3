import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import argparse, pickle, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
a = ap.parse_args()
hk = a.obs_kind == "spread_hawkes"
logs = pickle.load(open(os.path.join(HERE, "outputs", "stage3c", "logs",
                                     "logs_ensemble_valsel_spread_hawkes.pkl" if hk else "logs_ensemble_valsel.pkl"), "rb"))

def nf(ep):
    return sum(s.true_alpha * s.v ** 2 + s.true_kappa * s.v * (s.q_before - s.v) for s in ep.steps)

tw = np.array([nf(e) for e in logs["twap"]])
print(f"{'policy':22s} {'max|log-npz| ens':>17s} {'max|log-npz| twap':>18s} {'d_nf vs TWAP':>13s}")
for name, eps in logs.items():
    if not name.startswith("ens_"):
        continue
    z = np.load(os.path.join(HERE, "outputs", "stage3c", f"ens_nf_{name[4:]}{'_hawkes' if hk else ''}.npz"))
    mine = np.array([nf(e) for e in eps])
    print(f"{name:22s} {np.abs(mine - z['fresh_ens'][:200]).max():17.2e} "
          f"{np.abs(tw - z['fresh_twap'][:200]).max():18.2e} {(mine - tw).mean():+13.1f}")