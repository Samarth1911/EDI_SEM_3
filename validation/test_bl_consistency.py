"""
test_bl_consistency.py -- Stage 4A: checks Barger-Lorig's first-order multiplier against INDEPENDENT derivations. NO simulation, no
training. NOT RUN by its author. Run from EDI/validation (venv active):   python test_bl_consistency.py

BL multiplier (barger_lorig.bl_multiplier):  m = 1 + T*drift_a/(2 alpha) + T^2*drift_k/(6 tau alpha),  T = n*tau (n steps remaining).

REFERENCES (derived by me, stated here so they can be checked):
 (1) alpha-term vs CONTINUOUS-TIME certainty-equivalent (CE) plan: nu_s ~ 1/E[a_s] with E[a_s] = theta + (a-theta) e^{-lam s}.
     Closed form, eps = (alpha-theta)/theta, x = lam*T:   m_CE = (1/(1+eps)) / ( ln((e^x + eps)/(1+eps)) / x ).
     Series in x at infinitesimal eps: coefficient c_CE = x/2 - x^2/6 + ..., BL's is x/2, so BL/CE = 1 + x/3 + x^2/36 + ...
     => BL equals CE only when x = lam*T_remaining is small; at the horizon start (x = lam*T about 3) BL overshoots about 2.2x EVEN for
     infinitesimal deviations. (Nonlinear example, x = 3, alpha = 2*theta: BL says 0.25, CE says about 0.64.)
 (2) kappa-term vs a NUMERICAL plan: minimise sum_s alpha_s v_s^2 + sum_s kappa_s v_s R_s (R_s = shares sold after s) with the mean
     forecast paths of alpha and kappa, exact discrete QP solved by linear algebra, on grids refined by factors 1, 4, 16 with
     alpha' = alpha*ref, tau' = tau/ref (so a = tau*alpha, hence the continuous model, is unchanged). Hand derivation: the first-order
     kappa effect equals BL's kappa-term to leading order in lam_k*T (with a factor 1/(1 - kappa/(2 alpha)) that vanishes under refinement).
VERIFIED here: BL's two terms against these references in the small-horizon limit, and BL's invariance to grid refinement.
NOT verified: the paper's printed formula itself (not re-read here); optimality under stochastic dynamics (the references are
certainty-equivalent: Jensen/volatility effects ignored); anything about the Taylor remainder at large lam*T beyond the numbers printed.
"""
import math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from simulator import SimParams
from regime_simulator import RegimeParams
from barger_lorig import params_from_regimes, drifts, bl_multiplier

sp = SimParams(); N, tau = sp.N, sp.tau
SETS = {"single": params_from_regimes(sp, RegimeParams.single())[0]}
_d = params_from_regimes(sp, RegimeParams.default())
SETS["calm"], SETS["stressed"] = _d[0], _d[1]
EPS = 1e-3
ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def m_bl(par, n, ea, ek, ref=1):
    la, ta, lk, tk = par
    t = N * ref - n * ref
    a_, k_ = ta * ref * (1 + ea), tk * (1 + ek)
    da, dk = drifts([(la, ta * ref, lk, tk)], [1.0], a_, k_)
    return bl_multiplier(t, N * ref, tau / ref, a_, k_, da, dk)


def m_ce_cont(par, n, ea):
    x = par[0] * n * tau
    return float((1.0 / (1 + ea)) / (np.log((np.exp(x) + ea) / (1 + ea)) / x))


def m_ce_disc(par, n, ea, ref=1):
    la, ta = par[0], par[1]
    nn, r = n * ref, 1 - la * tau / ref
    w = 1.0 / (1 + ea * r ** np.arange(nn))
    return float(w[0] * nn / w.sum())


def m_qp(par, n, ek, ref=1):
    la, ta, lk, tk = par
    nn, rk = n * ref, 1 - lk * tau / ref
    k_path = tk * (1 + ek * rk ** np.arange(nn))
    idx = np.arange(nn)
    H = 0.5 * k_path[np.minimum.outer(idx, idx)]
    H[idx, idx] = ta * ref
    np.linalg.cholesky(H)                                  # raises if not positive definite
    x = np.linalg.solve(H, np.ones(nn))
    v = x / x.sum()
    assert (v > 0).all(), "QP solution not interior"
    return float(v[0] * nn)


def coef(f, *args, **kw):                                  # -d m / d eps at 0 (central difference)
    return -(f(*args, +EPS, **kw) - f(*args, -EPS, **kw)) / (2 * EPS)


print(f"N={N}, tau={tau}, horizon T={N * tau}")
for name, par in SETS.items():
    la, ta, lk, tk = par
    print(f"\n================ parameter set '{name}': lam_a={la:.4g} theta_a={ta:.4g} lam_k={lk:.4g} theta_k={tk:.4g} "
          f"(lam_a*T={la * N * tau:.2f}, lam_k*T={lk * N * tau:.2f}) ================")

    # refinement invariance of BL itself
    inv = max(abs(m_bl(par, n, ea, ek, 1) - m_bl(par, n, ea, ek, 4)) for n in (50, 20, 5) for ea in (-0.3, 0.4) for ek in (-0.3, 0.5))
    check("BL multiplier is invariant to grid refinement", inv < 1e-10, f"(max diff {inv:.1e})")

    # ---- Test A: alpha-term, infinitesimal deviation (kappa at its mean) ----
    print("\n  A. alpha-term: response coefficient c = -dm/d(eps_alpha) at eps -> 0  (kappa at its long-run mean)")
    print(f"  {'n':>3s} {'x=lam*T':>8s} {'c_BL':>8s} {'c_CE cont':>10s} {'c_CE disc':>10s} {'c_CE disc/16':>13s} "
          f"{'BL/CEcont':>10s} {'series':>7s} {'BL/CEdisc':>10s}")
    rows = []
    for n in (50, 40, 30, 20, 10, 5, 3, 2):
        x = la * n * tau
        cb = -(m_bl(par, n, +EPS, 0.0) - m_bl(par, n, -EPS, 0.0)) / (2 * EPS)
        cc = coef(m_ce_cont, par, n)
        cd1, cd16 = coef(m_ce_disc, par, n, ref=1), coef(m_ce_disc, par, n, ref=16)
        rows.append((n, x, cb, cc, cd1, cd16))
        print(f"  {n:3d} {x:8.3f} {cb:8.4f} {cc:10.4f} {cd1:10.4f} {cd16:13.4f} {cb / cc:10.3f} {1 + x / 3 + x * x / 36:7.3f} {cb / cd1:10.3f}")
    disc_err = max(abs(r[5] / r[3] - 1) for r in rows)
    check("A1 refined discrete CE converges to the continuous closed form", disc_err < 0.06, f"(max rel. diff {disc_err:.3f})")
    ratios = [r[2] / r[3] for r in rows]                   # ordered by decreasing n
    check("A2 BL -> CE as lam*T_remaining -> 0", abs(ratios[-1] - 1) < 0.10 and all(a >= b - 1e-9 for a, b in zip(ratios, ratios[1:])),
          f"(ratio at n={rows[-1][0]}: {ratios[-1]:.3f}; at n=50: {ratios[0]:.3f}; monotone in n)")
    check("A3 series 1 + x/3 + x^2/36 matches the numerical ratio for x <= 1",
          all(abs(r[2] / r[3] - (1 + r[1] / 3 + r[1] ** 2 / 36)) < 0.03 for r in rows if r[1] <= 1.0))
    print(f"  => at the horizon start BL's linear response to an alpha deviation is {ratios[0]:.2f}x the continuous CE's "
          f"({rows[0][2] / rows[0][4]:.2f}x the discrete CE used in the benchmarks), for INFINITESIMAL deviations.")

    # ---- Test B: nonlinear deviations (single set only) ----
        # ---- Test B: nonlinear deviations (single set only) ----
    if name == "single":
        print("\n  B. multipliers at larger alpha deviations (kappa at mean); eps = (alpha-theta)/theta")
        print(f"  {'n':>3s} {'eps':>6s} {'BL':>8s} {'CE cont':>8s} {'CE disc':>8s}")
        for n in (50, 25):
            for ea in (-0.5, -0.25, 0.25, 0.5, 1.0, 1.5):
                print(f"  {n:3d} {ea:6.2f} {m_bl(par, n, ea, 0.0):8.3f} {m_ce_cont(par, n, ea):8.3f} {m_ce_disc(par, n, ea):8.3f}")

        # B1/B2: docstring example at the horizon start (n = N), eps_a = 1, kappa at its mean.
        # BL alpha-term: T*drift_a/(2 alpha) = -x*eps/(2(1+eps)) = -x/4 at eps = 1, so BL = 1 - x/4 (0.25 at x = 3).
        # CE cont = 0.5 / (ln((e^x + 1)/2) / x) (about 0.64 at x = 3).
        x = la * N * tau
        b, c = m_bl(par, N, 1.0, 0.0), m_ce_cont(par, N, 1.0)
        check("B1 BL = 1 - x/4 at eps_a=1 (kappa at mean)", abs(b - (1 - x / 4)) < 1e-6, f"(BL {b:.4f}, x={x:.3f})")
        check("B2 BL undershoots CE at large deviation (overshoot of response)", b < c - 0.1, f"(BL {b:.3f} vs CE {c:.3f})")
       # ---- Test C: kappa-term vs numerical QP plan on refined grids (Richardson-extrapolated) ----
    print("\n  C. kappa-term: c_k = -dm/d(eps_kappa) at eps -> 0  (alpha at its long-run mean); QP plan on grids refined by ref")
    print("     QPext = (4*QP16 - QP4)/3 removes the O(1/ref) discretisation error (assumed first order)")
    print(f"  {'n':>3s} {'x_k=lam_k*T':>11s} {'c_BL':>10s} {'QP ref4':>10s} {'QP ref16':>10s} {'QPext':>10s} {'QP16/BL':>8s} {'QPext/BL':>9s}")
    rc = []
    for n in (50, 25, 10, 5, 2):
        xk = lk * n * tau
        cb = -(m_bl(par, n, 0.0, +EPS) - m_bl(par, n, 0.0, -EPS)) / (2 * EPS)
        q4, q16 = (coef(m_qp, par, n, ref=r) for r in (4, 16))
        qx = (4 * q16 - q4) / 3
        rc.append((n, xk, cb, q4, q16, qx))
        print(f"  {n:3d} {xk:11.3f} {cb:10.5f} {q4:10.5f} {q16:10.5f} {qx:10.5f} {q16 / cb:8.3f} {qx / cb:9.3f}")
    r2, r5 = rc[-1], rc[-2]
    tol = lambda r: 0.3 * r[1] + 0.03          # expansion error is first order in x_k (observed about 0.25*x_k)
    check("C0 refinement moves the QP toward BL (n=2, ref4 -> ref16)", abs(r2[4] / r2[2] - 1) < abs(r2[3] / r2[2] - 1),
          f"(|ratio-1|: {abs(r2[3] / r2[2] - 1):.3f} -> {abs(r2[4] / r2[2] - 1):.3f})")
    check("C1 kappa-term matches the extrapolated plan for small lam_k*T (n=2)", abs(r2[5] / r2[2] - 1) < tol(r2),
          f"(ratio {r2[5] / r2[2]:.3f}, tol {tol(r2):.3f})")
    check("C2 ... and for n=5", abs(r5[5] / r5[2] - 1) < tol(r5), f"(ratio {r5[5] / r5[2]:.3f}, tol {tol(r5):.3f})")
    dev = [abs(r[5] / r[2] - 1) for r in rc]
    check("C3 deviation grows with horizon (expansion is first order in lam_k*T)", all(a >= b - 0.02 for a, b in zip(dev, dev[1:])),
          f"(|ratio-1| by n=50,25,10,5,2: {np.round(dev, 3)})")

print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
print("Reading: A2/C1 passing means BL's formula is the correct small-horizon limit of the independent references, so its "
      "underperformance in the simulations is the expansion's range (overshoot factor printed in A and the QPext/BL ratios in C), "
      "not a coding error. C-checks use Richardson-extrapolated QP; a persistent deviation larger than about 0.3*x_k would "
      "indicate a kappa-term error. A2 or C1 failing would indicate a formula/unit-mapping error in barger_lorig.py.")
sys.exit(0 if ok_all else 1)