"""
validate_interfaces.py  (STAGE 0)
---------------------------------
Runs REAL episodes through the contracts in src/interfaces.py using the existing
validated code (RegimeSwitchingSimulator, the three observation models, the
bootstrap ParticleFilter, the heuristic rule) and checks that everything that
crosses a module boundary conforms. Run from EDI/validation/:  python validate_interfaces.py

What it proves:  the contracts FIT the code that exists today.
What it does NOT prove: anything about the IMM-PF or a learned policy (Stages 2-3
do not exist yet); the K>1 belief is only checked on a synthetic object.
"""
import sys, os, json
from dataclasses import asdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from simulator import SimParams
from regime_simulator import RegimeSwitchingSimulator, RegimeParams
from observation_model import ObsParams
from spread_hawkes import make_observation_model_v2
from particle_filter import ParticleFilter, PFParams
from heuristic_policy import heuristic_velocity, HeuristicParams
import interfaces as I

N_EP = 6
PF = PFParams(n_particles=1000)          # small: this is an interface test, not a filter benchmark
results = []


def check(name, errs):
    ok = len(errs) == 0
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {errs[:3]}"))


# ---- minimal policies that implement the Policy protocol (Stage 3 owns the real ones) ----
class TWAP:
    name, uses_oracle_truth = "twap", False
    def reset(self, seed=None): pass
    def act(self, x): return 0.0 if x.q <= 0 else x.q / (x.N - x.t)      # equal split of what remains

class BeliefHeuristic:
    name, uses_oracle_truth = "belief", False
    def __init__(self, alpha_ref): self.ar, self.hp = alpha_ref, HeuristicParams()
    def reset(self, seed=None): pass
    def act(self, x):
        b = x.belief
        return heuristic_velocity(x.t, x.N, x.q, b.alpha_mean, b.alpha_std, self.ar, self.hp)


def run_episode(policy, seed, obs_kind):
    """Harness: the ONLY place truth and belief meet, and it enforces the info barrier."""
    sp = SimParams()
    sim = RegimeSwitchingSimulator(sp, RegimeParams.default(), seed=seed)
    st = I.seed_streams(seed)
    om = make_observation_model_v2(obs_kind, seed=st["obs"], tau=sp.tau)
    om.reset(st["obs"])
    pf = ParticleFilter(sp, ObsParams(), PF, seed=st["filter"])
    policy.reset(st["policy"])

    gen = sim.run_stepwise(seed=st["sim"])
    state = next(gen)
    obs_rows, beliefs, truths = [], [], []
    while True:
        truth = I.TruthStep.from_state(state)
        raw = om.observe_step(truth.kappa, truth.alpha, truth.regime)      # regime -> generator only
        obs = I.Observation.from_dict(truth.t, raw)
        pf.step(obs.core_dict())                                          # filter sees core channels only
        belief = I.belief_from_bootstrap_pf(pf, truth.t)
        x = I.PolicyInput(t=truth.t, N=truth.N, q=truth.q, belief=belief,
                          truth=truth if policy.uses_oracle_truth else None)
        v = policy.act(x)
        obs_rows.append({**obs.core_dict(), **obs.extras}); beliefs.append(belief); truths.append(truth)
        try:
            state = gen.send(v)
        except StopIteration as stop:
            ep = stop.value
            break
    cfg = I.WorldConfig(sim=asdict(sp), regime={"P": sim.rp.P, "names": sim.rp.names,
                                                "specs": [asdict(s) for s in sim.rp.specs]},
                        obs_kind=obs_kind)
    log = I.episode_log_from_episode(policy.name, seed, cfg, ep, obs_rows=obs_rows, beliefs=beliefs)
    return log, ep, truths, obs_rows, beliefs, cfg


if __name__ == "__main__":
    print("validate_interfaces (Stage 0 conformance)")
    sp = SimParams()

    for kind in ("gaussian", "hawkes", "spread_hawkes"):
        logs = {"twap": [], "belief": []}
        alpha_paths = {"twap": [], "belief": []}
        c1_errs, c2_errs, c3_errs, c4_errs = [], [], [], []
        last = None
        for seed in range(100, 100 + N_EP):
            for pol in (TWAP(), BeliefHeuristic(sp.theta_alpha)):
                log, ep, truths, obs_rows, beliefs, cfg = run_episode(pol, seed, kind)
                logs[pol.name].append(log); alpha_paths[pol.name].append(ep.alpha.copy())
                # C1: truth steps are well-formed, indexed 0..N-1, regime ids valid
                if [t.t for t in truths] != list(range(sp.N)): c1_errs.append("truth index")
                if any(t.regime not in (0, 1) for t in truths): c1_errs.append("regime id")
                # C2: core channels present, finite, positive
                for r in obs_rows:
                    if not all(np.isfinite(r[c]) and r[c] > 0 for c in I.CORE_CHANNELS): c2_errs.append("core channel")
                # C3: every belief obeys the belief contract with K=1 (bootstrap PF)
                for b in beliefs: c3_errs += I.validate_belief(b, K=1)
                # C4: every log obeys the episode contract
                c4_errs += I.validate_episode_log(log)
                last = (log, truths[-1], obs_rows[-1], beliefs[-1], cfg)
        check(f"C1 truth steps [{kind}]", c1_errs)
        check(f"C2 observation core channels [{kind}]", c2_errs)
        check(f"C3 belief state K=1 [{kind}]", c3_errs)
        check(f"C4 episode logs (indices, v in [0,q], IS recompute, liquidated) [{kind}]", c4_errs)

        # common random numbers: the market path must not depend on the policy
        crn = [np.allclose(a, b) for a, b in zip(alpha_paths["twap"], alpha_paths["belief"])]
        check(f"C0 common random numbers: alpha path identical across policies [{kind}]",
              [] if all(crn) else ["alpha paths differ between policies for same seed"])

        m = I.paired_metrics(logs, baseline="twap")
        print(f"       paired IS vs TWAP [{kind}]: belief diff = {m['belief']['paired_diff_mean']:+.1f} "
              f"(se {m['belief']['paired_diff_se']:.1f}), TWAP mean IS = {m['twap']['IS_mean']:.0f}, n={N_EP}")

        # C5: build a StepResponse from the last step and validate it (incl. JSON round-trip)
        log, tr, orow, bel, cfg = last
        obs = I.Observation.from_dict(tr.t, orow)
        pol_state = {"twap": {"q": 0.0, "v": 0.0, "S": tr.S, "S_exec": tr.S, "cum_cost": 0.0}}
        resp = I.step_response(episode_id="ep-test", seed=100, truth=tr, obs=obs, belief=bel, policies=pol_state,
                               config={"obs_kind": kind, "n_regimes": 2, "regime_names": ["calm", "stressed"],
                                       "q0": sp.q0, "S0": sp.S0}, done=True)
        errs = I.validate_step_response(resp)
        # bootstrap PF is single-regime => its K=1 belief conflicts with config.n_regimes=2: record honestly
        expected = ["len(belief.regime_probs) != config.n_regimes"]
        check(f"C5 StepResponse structure [{kind}] (only expected mismatch: K=1 filter vs 2-regime world)",
              [e for e in errs if e not in expected])
        json.dumps(resp)   # must be JSON-serialisable

    # K=2 belief (what the IMM-PF will produce) -- synthetic, checks the contract itself
    b2 = I.BeliefState(t=3, regime_probs=[0.3, 0.7], kappa_mean=1.3e-5, kappa_std=6e-7, alpha_mean=2e-4,
                       alpha_std=1e-5, ess=900.0,
                       per_regime=[I.RegimeBelief(0, 0.3, 1.2e-5, 5e-7, 1.6e-4, 9e-6, 400.0),
                                   I.RegimeBelief(1, 0.7, 1.35e-5, 6e-7, 2.4e-4, 1e-5, 500.0)])
    check("C3 synthetic K=2 belief + feature vector length", I.validate_belief(b2, K=2) +
          ([] if len(b2.to_vector()) == 4 + 2 else ["to_vector length"]))
    bad = I.BeliefState(t=0, regime_probs=[0.5, 0.6], kappa_mean=1, kappa_std=1, alpha_mean=1, alpha_std=1, ess=1)
    check("C3 validator rejects regime_probs that do not sum to 1", [] if I.validate_belief(bad) else ["not rejected"])

    print("\nALL CONFORMANCE CHECKS PASS:", all(results), f"({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)