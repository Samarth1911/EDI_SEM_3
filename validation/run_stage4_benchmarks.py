"""
run_stage4_benchmarks.py -- Stage 4A (Person 4). NOT RUN by its author. Run from EDI/validation (venv active):
    python run_stage4_benchmarks.py --n 200 --agent-logs outputs/stage3c/logs/logs_valsel.pkl

WHAT IT DOES
  1. For fresh worlds (seeds 1,000,000+i, the same worlds as the Stage 3 export) it runs the REAL simulator, observation
     model and IMM-PF once (a no-trade pass), then plays each benchmark through the real RegimeSwitchingSimulator:
       twap | heuristic_belief | heuristic_oracle | bl1_oracle_regime | bl1_oracle_calm | bl1_filtered_mix | bl1_oracle_regime_grid
     (Barger-Lorig first-order closed form: see barger_lorig.py for the source, the unit mapping and its limits).
  2. Every run becomes an interfaces.EpisodeLog and passes interfaces.validate_episode_log.
  3. Loads the Stage 3 logs (logs_valsel.pkl: policy name -> list[EpisodeLog]) and scores everything the same way:
       full IS (paired_diff vs TWAP, CVaR95) and NOISE-FREE cost  sum(alpha v^2 + kappa v (q - v))  (the Stage 3 metric).
  4. HARD CHECKS (it stops/warns loudly): (a) trading must not change kappa/alpha/regime (the cached-belief assumption),
     (b) my TWAP must reproduce the Stage 3 TWAP logs world by world -> proves the worlds are identical.

INFERENCE NOTE: benchmarks are deterministic, so their uncertainty is world-sampling only (paired se over worlds).
  For DDQN arms the seed-to-seed spread is reported separately; the arm-vs-benchmark t uses seed-averaged cost per world.
"""
import argparse, json, math, os, pickle, re, sys, time
from dataclasses import asdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np

from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from observation_model import ObsParams
from spread_hawkes import make_observation_model_v2
from imm_particle_filter import IMMParticleFilter, IMMParams
from heuristic_policy import heuristic_velocity, HeuristicParams
import interfaces as I
from barger_lorig import bl_volume, params_from_regimes

FRESH0 = 1_000_000
GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)      # Stage 3 action multipliers
H2H = ("ce_filtered_grid", "ce_oracle_grid", "ce_oracle", "heuristic_belief")   # head-to-head columns (BL dropped: it loses to TWAP)


def jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def nf_cost(log):
    """Noise-free (drift-only) cost of an EpisodeLog: sum alpha v^2 + kappa v (q_before - v)."""
    return float(sum(s.true_alpha * s.v ** 2 + s.true_kappa * s.v * (s.q_before - s.v) for s in log.steps))


def cvar95(x):
    k = max(1, int(math.ceil(0.05 * len(x))))
    return float(np.sort(x)[-k:].mean())


def prepass(seed, sp, rp, op, kind, imm):
    st = I.seed_streams(seed)
    sim = RegimeSwitchingSimulator(sp, rp, seed=st["sim"])
    gen = sim.run_stepwise(seed=st["sim"]); next(gen)
    while True:
        try:
            gen.send(0.0)
        except StopIteration as stop:
            ep = stop.value
            break
    om = make_observation_model_v2(kind, seed=st["obs"], tau=sp.tau); om.reset(st["obs"])
    imm.reset(st["filter"])
    bel = []
    for t in range(sp.N):
        raw = om.observe_step(float(ep.kappa[t]), float(ep.alpha[t]), int(ep.regime[t]))
        bel.append(imm.update(I.Observation.from_dict(t, raw)))
    return ep, bel


def make_policies(sp, rp, hp, counters):
    N, tau, K = sp.N, sp.tau, rp.n_regimes
    P = params_from_regimes(sp, rp)

    def bl(name, probs_fn, state_fn, grid=None):
        def f(t, s, b):
            a, k = state_fn(s, b)
            v, c = bl_volume(t, N, tau, s["q"], a, k, P, probs_fn(s, b), grid=grid)
            cc = counters.setdefault(name, [0, 0]); cc[0] += int(c); cc[1] += 1
            return v
        return f
    true_ak = lambda s, b: (s["alpha"], s["kappa"])
    est_ak = lambda s, b: (b.alpha_mean, b.kappa_mean)

    # ---- certainty-equivalent plan v_s ~ 1/E[alpha_s], EXACT mean forecast (CIR decay + Markov regime), receding horizon.
    # NOT Barger-Lorig: BL's alpha-term is the first-order Taylor expansion of this plan (diag_bl.py showed it overshoots).
    # Ignores permanent impact and Jensen effects: a reference, not an optimum.
    Pm = np.asarray(rp.P, dtype=float).reshape(K, K)
    _eff = [rp.effective_cir(sp, r) for r in range(K)]
    LAM = np.array([e["lambda_alpha"] for e in _eff], dtype=float)
    TH = np.array([e["theta_alpha"] for e in _eff], dtype=float)

    def ce_forecast(alpha_t, probs, n_ahead):
        out = np.empty(n_ahead)
        pi = np.asarray(probs, dtype=float).copy()
        m = pi * float(alpha_t)              # E[alpha; regime=j]. For beliefs this ASSUMES alpha_hat independent of regime.
        out[0] = alpha_t
        for k in range(1, n_ahead):
            pi_next = pi @ Pm
            m = (1.0 - LAM * tau) * (m @ Pm) + LAM * TH * tau * pi_next
            pi = pi_next
            out[k] = m.sum()
        return out

    def ce(probs_fn, alpha_fn, grid=None):
        def f(t, s, b):
            q = s["q"]
            if q <= 0:
                return 0.0
            if t >= N - 1:
                return float(q)
            w = 1.0 / np.maximum(ce_forecast(alpha_fn(s, b), probs_fn(s, b), N - t), 1e-12)
            v = q * w[0] / w.sum()
            if grid is not None:
                g = np.asarray(grid, dtype=float); base = q / (N - t)
                v = float(g[np.argmin(np.abs(np.log(g) - np.log(max(v / base, 1e-9))))]) * base
            return float(min(v, q))
        return f
    onehot = lambda s, b: np.eye(K)[int(s["regime"])]
    return {
        "twap": lambda t, s, b: 0.0 if s["q"] <= 0 else s["q"] / (N - t),
        "heuristic_belief": lambda t, s, b: heuristic_velocity(t, N, s["q"], b.alpha_mean, b.alpha_std, sp.theta_alpha, hp),
        "heuristic_oracle": lambda t, s, b: heuristic_velocity(t, N, s["q"], s["alpha"], 0.0, sp.theta_alpha, hp),
        "bl1_oracle_regime": bl("bl1_oracle_regime", lambda s, b: np.eye(K)[int(s["regime"])], true_ak),
        "bl1_oracle_calm": bl("bl1_oracle_calm", lambda s, b: np.eye(K)[0], true_ak),
        "bl1_filtered_mix": bl("bl1_filtered_mix", lambda s, b: b.regime_probs, est_ak),
        "bl1_oracle_regime_grid": bl("bl1_oracle_regime_grid", lambda s, b: np.eye(K)[int(s["regime"])], true_ak, grid=GRID),
        "ce_oracle": ce(onehot, lambda s, b: s["alpha"]),
        "ce_oracle_grid": ce(onehot, lambda s, b: s["alpha"], grid=GRID),
        "ce_filtered": ce(lambda s, b: b.regime_probs, lambda s, b: b.alpha_mean),
        "ce_filtered_grid": ce(lambda s, b: b.regime_probs, lambda s, b: b.alpha_mean, grid=GRID),
    }


def play(seed, fn, name, sp, rp, ep0, bel, wcfg):
    sim = RegimeSwitchingSimulator(sp, rp, seed=seed)
    gen = sim.run_stepwise(seed=seed)
    state = next(gen)
    while True:
        t = int(state["t"])
        v = float(fn(t, state, bel[t]))
        if t >= sp.N - 1:
            v = float(state["q"])                    # forced liquidation, identical for every policy
        try:
            state = gen.send(v)
        except StopIteration as stop:
            ep = stop.value
            break
    for a, b, what in ((ep.alpha, ep0.alpha, "alpha"), (ep.kappa, ep0.kappa, "kappa"), (ep.regime, ep0.regime, "regime")):
        if not np.array_equal(a, b):
            raise RuntimeError(f"seed {seed}: trading changed {what}; the cached-belief assumption is violated")
    return I.episode_log_from_episode(name, seed, wcfg, ep)


def arm_of(name):
    """'<arm>_s<seed>[_<suffix>]' -> '<arm>', e.g. 'filt_full_bonus_abl_b1_s3_valsel' -> 'filt_full_bonus_abl_b1'.
    (Old version required the name to END in digits; Stage 3 names end in '_valsel', so nothing was ever grouped.)"""
    a = re.sub(r"_s(?:eed)?_?\d+(?:_[A-Za-z]+)?$", "", name)
    return a if a else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
    ap.add_argument("--imm-particles", type=int, default=2000)
    ap.add_argument("--agent-logs", default=os.path.join(HERE, "outputs", "stage3c", "logs", "logs_valsel.pkl"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "outputs", "stage4"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    sp, op = SimParams(), ObsParams()
    rp = RegimeParams.default(); rp.init_probs = rp.stationary_distribution()   # = Stage 3 'stationary_init'
    imm = IMMParticleFilter(sp, op, rp, IMMParams(n_particles=a.imm_particles), seed=0)
    hp = HeuristicParams()
    wcfg = I.WorldConfig(sim=asdict(sp), regime={"P": rp.P, "names": rp.names, "specs": [asdict(s) for s in rp.specs]},
                         obs_kind=a.obs_kind)
    counters = {}
    pols = make_policies(sp, rp, hp, counters)
    seeds = [FRESH0 + i for i in range(a.n)]
    logs = {k: [] for k in pols}
    t0 = time.time()
    for i, s in enumerate(seeds):
        ep0, bel = prepass(s, sp, rp, op, a.obs_kind, imm)
        for name, fn in pols.items():
            logs[name].append(play(s, fn, name, sp, rp, ep0, bel, wcfg))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{a.n} worlds ({time.time() - t0:.0f}s)", flush=True)
    bad = sum(len(I.validate_episode_log(l)) for v in logs.values() for l in v)
    print(f"episode-log contract errors: {bad}")

    # ---- Stage 3 agents ------------------------------------------------------
    A = {}
    if os.path.exists(a.agent_logs):
        with open(a.agent_logs, "rb") as f:
            raw = pickle.load(f)
        for k, v in raw.items():
            if [l.seed for l in v][:a.n] == seeds:
                A[k] = v[:a.n]
            else:
                print(f"WARNING: skipping {k}: world seeds differ from {FRESH0}+i")
        tk = next((k for k in A if "twap" in k.lower()), None)
        if tk is not None:
            d = np.array([l.IS for l in A[tk]]) - np.array([l.IS for l in logs["twap"]])
            print(f"TWAP parity vs Stage 3 logs ({tk}): max |IS diff| = {np.abs(d).max():.3e}"
                  f"  -> {'WORLDS IDENTICAL' if np.abs(d).max() < 1e-6 else 'MISMATCH: do NOT compare (send env_factory.py)'}")
        print("Stage 3 policy names:", sorted(A)[:6], "..." if len(A) > 6 else "")
    else:
        print(f"(no Stage 3 logs at {a.agent_logs}; benchmarks only)")

    # ---- scoring ---------------------------------------------------------------
    full = {k: np.array([l.IS for l in v]) for k, v in logs.items()}
    nf = {k: np.array([nf_cost(l) for l in v]) for k, v in logs.items()}
    tw_full, tw_nf = full["twap"], nf["twap"]

    def row(name, f, n):
        d_f, d_n = f - tw_full, n - tw_nf
        se = lambda d: d.std(ddof=1) / math.sqrt(len(d))
        return dict(name=name, IS_mean=float(f.mean()), d_full=float(d_f.mean()), se_full=float(se(d_f)),
                    d_nf=float(d_n.mean()), se_nf=float(se(d_n)), cvar_diff=cvar95(f) - cvar95(tw_full))

    rows = [row(k, full[k], nf[k]) for k in logs]
    out = []
    P = lambda s="": (print(s), out.append(s))
    P(f"worlds: {a.n} (seeds {FRESH0}..{FRESH0 + a.n - 1}), obs={a.obs_kind}.  diffs vs TWAP in $, negative = cheaper.")
    P(f"\n{'policy':26s} {'meanIS':>8s} {'d_full(se)':>16s} {'d_noisefree(se)':>18s} {'CVaR95-TWAP':>11s}")
    for r in rows:
        P(f"{r['name']:26s} {r['IS_mean']:8.1f} {r['d_full']:+8.1f} ({r['se_full']:5.1f}) {r['d_nf']:+9.1f} ({r['se_nf']:5.1f}) {r['cvar_diff']:+11.1f}")
    P("\nBarger-Lorig truncated at zero (share of steps): " +
      ", ".join(f"{k}: {c[0] / max(c[1], 1):.3f}" for k, c in counters.items() if k.startswith("bl1")))

    arms = {}
    for k, v in A.items():
        if "twap" in k.lower():
            continue
        arms.setdefault(arm_of(k), []).append(k)
    arm_rows, arm_nf = [], {}
    if arms:
        P(f"\n--- Stage 3 agents (validation-selected checkpoints), aggregated over training seeds: {sorted((a_, len(v)) for a_, v in arms.items())}")
        P(f"{'arm':26s} {'n_seeds':>7s} {'d_full mean':>11s} {'d_nf mean':>10s} {'sd(seeds)':>9s} {'world se':>8s} {'CVaR95-TWAP':>11s}")
        for arm, ks in sorted(arms.items()):
            F = np.array([[l.IS for l in A[k]] for k in ks]); Nn = np.array([[nf_cost(l) for l in A[k]] for k in ks])
            dn_seed = (Nn - tw_nf).mean(axis=1); dw = (Nn.mean(axis=0) - tw_nf)
            cv = float(np.mean([cvar95(f) for f in F]) - cvar95(tw_full))
            P(f"{arm:26s} {len(ks):7d} {(F.mean() - tw_full.mean()):+11.1f} {dn_seed.mean():+10.1f} "
              f"{(dn_seed.std(ddof=1) if len(ks) > 1 else float('nan')):9.1f} {dw.std(ddof=1) / math.sqrt(a.n):8.1f} {cv:+11.1f}")
            arm_nf[arm] = Nn.mean(axis=0)
        P("\n--- head-to-head, noise-free cost, paired over worlds (arm seed-averaged minus benchmark; negative = agent cheaper)")
        P(f"{'arm':26s} " + " ".join(f"{b[:22]:>24s}" for b in H2H))
        for arm, x in sorted(arm_nf.items()):
            cells = []
            for b in H2H:
                d = x - nf[b]; se = d.std(ddof=1) / math.sqrt(a.n)
                cells.append(f"{d.mean():+8.1f} (t {d.mean() / se:+5.2f})")
            P(f"{arm:26s} " + " ".join(f"{c:>24s}" for c in cells))
    P("\nCAVEATS: world-sampling inference only for benchmarks; DDQN agents carry extra training-seed variance (sd(seeds)).")
    P("Barger-Lorig variants are misspecified for a regime-switching world and use simulator (uncalibrated) parameters.")
    P("ce_* = certainty-equivalent plan with exact mean forecast (not BL). ce_filtered assumes E[alpha;regime=j] ~ p_j*alpha_hat. Reference, not optimum.")
    P("Compare agents with benchmarks on the SAME metric: the noise-free column is the Stage 3 metric.")
    with open(os.path.join(a.outdir, "benchmarks.md"), "w") as f:
        f.write("```\n" + "\n".join(out) + "\n```\n")
    with open(os.path.join(a.outdir, "benchmarks.json"), "w") as f:
        json.dump(dict(rows=rows, n=a.n, bl_zero_clip=counters, arms=list(arms)), f, indent=1, default=jd)
    print("wrote", a.outdir)


if __name__ == "__main__":
    main()