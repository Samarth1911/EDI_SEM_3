"""
diag_stage3b.py -- Stage 3B diagnostics.  NOT RUN by its author; run it from EDI/validation.

Why this exists: the 1500-episode oracle run lost to TWAP (+843, s.e. 37.7). Before anything else is
built, three questions need numbers, not guesses:
  A. Is TWAP already (near-)optimal among STATE-INDEPENDENT schedules on this action grid?
     -> paired IS of every constant multiplier vs TWAP.
  B. How large is the reward noise compared with the value gaps the agent has to resolve?
     -> std of (IS - IS_decomp) vs the constant-multiplier gaps, in dollars and in Q units.
  C. (with --agent) What did a trained agent learn? Timing profile of its multipliers, and whether its
     Q-gaps are larger than the noise / TD error.

Only the BeliefMDPEnv API already used by ddqn_train.py is assumed:
  env.reset(seed=) -> (obs, info); env.step(a) -> (obs, r, done, trunc, info); env.n_actions;
  env.twap_action; env.cfg.action_multipliers; env.get_episode() -> {"IS","sum_decomp","steps":[{"action",...}]}.

UNIT CONVERSION (INFERRED, not read from the env code): from your log, mean return -28.3 <-> mean IS 5610 at
reward_scale 0.1, i.e. return ~= -IS / (S0*q0) * 1e4 * reward_scale.  So one Q/return unit is
S0*q0/1e4/reward_scale dollars (= $200 for S0=10, q0=20000, reward_scale=0.1). Pass --reward-scale to match.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np
from belief_mdp_env import EnvConfig
from env_factory import make_env
from imm_particle_filter import IMMParams
from simulator import SimParams
from ddqn import DDQNAgent
from ddqn_train import EVAL_BASE


def rollout(env, seeds, chooser):
    """chooser(obs, t) -> action index. Returns arrays over episodes (and states over all steps)."""
    IS, dec, mults, states = [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done, t = False, 0
        while not done:
            states.append(np.array(obs, dtype=float))
            obs, _, done, _, _ = env.step(int(chooser(obs, t)))
            t += 1
        e = env.get_episode()
        IS.append(e["IS"]); dec.append(e["sum_decomp"])
        mults.append([env.cfg.action_multipliers[x["action"]] for x in e["steps"][:-1]])  # last step is forced
    return dict(IS=np.array(IS), dec=np.array(dec), mult=np.array(mults), states=np.array(states))


def pdiff(x, base):
    d = np.asarray(x) - np.asarray(base)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--arm", default="oracle", choices=["oracle", "raw_proxy", "filtered"])
    ap.add_argument("--no-unc", action="store_true")
    ap.add_argument("--no-regime", action="store_true")
    ap.add_argument("--filter", default="imm", choices=["imm", "single"])
    ap.add_argument("--obs-kind", default="gaussian", choices=["gaussian", "spread_hawkes"])
    ap.add_argument("--imm-particles", type=int, default=500)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", default=None, help="path to a saved .npz agent (must match --arm/--no-* flags)")
    a = ap.parse_args()

    env = make_env(EnvConfig(obs_mode=a.arm, use_uncertainty=not a.no_unc, use_regime=not a.no_regime,
                             n_regimes=1 if a.filter == "single" else 2, reward_scale=a.reward_scale),
                   obs_kind=a.obs_kind, filter_regimes=a.filter, seed=a.seed,
                   imm_params=IMMParams(n_particles=a.imm_particles), stationary_init=True)
    sp = SimParams()
    unit_usd = sp.S0 * sp.q0 / 1e4 / a.reward_scale
    seeds = list(range(EVAL_BASE, EVAL_BASE + a.eval_episodes))
    mults = env.cfg.action_multipliers
    out = {"unit_usd_per_q_unit(inferred)": unit_usd}

    # ---- A. constant-multiplier sweep ------------------------------------------------------------
    print(f"== A. constant multiplier vs TWAP  (n={len(seeds)} held-out worlds; last step always forced) ==")
    base = rollout(env, seeds, lambda o, t: env.twap_action)
    print(f"   TWAP: mean IS {base['IS'].mean():.1f}  (twap_action index {env.twap_action}, mult {mults[env.twap_action]})")
    sweep = []
    for i, m in enumerate(mults):
        r = rollout(env, seeds, lambda o, t, i=i: i)
        d_full, se_full = pdiff(r["IS"], base["IS"])
        d_nf, se_nf = pdiff(r["dec"], base["dec"])
        sweep.append(dict(action=i, mult=float(m), diff_full=d_full, se_full=se_full, diff_nf=d_nf, se_nf=se_nf))
        print(f"   mult {m:4.2f}: paired diff full {d_full:+9.1f} (se {se_full:6.1f})   noise-free {d_nf:+9.1f} (se {se_nf:6.1f})")
    out["constant_sweep"] = sweep
    best = min(sweep, key=lambda r: r["diff_nf"])
    print(f"   best constant multiplier (noise-free): {best['mult']} ({best['diff_nf']:+.1f}).")
    print("   If no constant multiplier beats TWAP, any gain must come from STATE-DEPENDENT timing.")

    # ---- B. noise vs gaps -----------------------------------------------------------------------
    noise_sd = float(np.std(base["IS"] - base["dec"], ddof=1))
    adj = [abs(sweep[i + 1]["diff_nf"] - sweep[i]["diff_nf"]) for i in range(len(sweep) - 1)]
    print("\n== B. reward noise vs value gaps ==")
    print(f"   std of price-diffusion term (IS - IS_decomp), TWAP: ${noise_sd:.0f} = {noise_sd / unit_usd:.2f} Q units per episode")
    print(f"   |gap| between adjacent constant multipliers (noise-free): min ${min(adj):.0f}, median ${np.median(adj):.0f}, "
          f"max ${max(adj):.0f}  (= {np.median(adj) / unit_usd:.2f} Q units median)")
    out["noise_sd_usd"] = noise_sd
    out["adjacent_gap_usd"] = adj

    # ---- C. trained agent ------------------------------------------------------------------------
    if a.agent:
        ag = DDQNAgent.load(a.agent)
        if ag.obs_dim != env.obs_dim:
            sys.exit(f"agent obs_dim {ag.obs_dim} != env obs_dim {env.obs_dim}: pass the same --arm/--no-* flags")
        r = rollout(env, seeds, lambda o, t: ag.greedy(o))
        d_full, se_full = pdiff(r["IS"], base["IS"]); d_nf, se_nf = pdiff(r["dec"], base["dec"])
        print(f"\n== C. agent {os.path.basename(a.agent)} ==")
        print(f"   paired diff vs TWAP: full {d_full:+.1f} (se {se_full:.1f}), noise-free {d_nf:+.1f} (se {se_nf:.1f})")
        chunks = np.array_split(np.arange(r["mult"].shape[1]), 7)
        prof = [float(r["mult"][:, c].mean()) for c in chunks]
        print("   mean multiplier by time block (early -> late): " + " ".join(f"{x:.2f}" for x in prof))
        print("   (a profile below 1.0 early means back-loading: the leftover is dumped in the forced last step)")
        tw = np.array([mults[env.twap_action]] * r["mult"].shape[1])
        print(f"   share of steps at the TWAP multiplier: {(np.isclose(r['mult'], tw[None, :])).mean():.2f}")
        Q = ag.q_values(base["states"])                       # states visited by TWAP on the held-out worlds
        srt = np.sort(Q, axis=1)
        gap12 = srt[:, -1] - srt[:, -2]
        gap_tw = Q.max(1) - Q[:, env.twap_action]
        rng_q = srt[:, -1] - srt[:, 0]
        print("   Q-value structure along TWAP paths (Q units | dollars):")
        print(f"     best-vs-2nd gap   mean {gap12.mean():.3f} | ${gap12.mean() * unit_usd:.0f}")
        print(f"     best-vs-TWAP gap  mean {gap_tw.mean():.3f} | ${gap_tw.mean() * unit_usd:.0f}   "
              f"(agent prefers a non-TWAP action in {(Q.argmax(1) != env.twap_action).mean():.0%} of states)")
        print(f"     max-min over actions mean {rng_q.mean():.3f} | ${rng_q.mean() * unit_usd:.0f}")
        print("   Compare with B: if these gaps are no larger than the TD error (training loss ~0.3 Huber => RMS ~0.8 Q units)")
        print("   the argmax is mostly noise.")
        out["agent"] = dict(path=a.agent, diff_full=d_full, se_full=se_full, diff_nf=d_nf, se_nf=se_nf,
                            time_profile=prof, gap_best_second=float(gap12.mean()), gap_best_twap=float(gap_tw.mean()),
                            q_range=float(rng_q.mean()))

    od = os.path.join(HERE, "outputs", "stage3b"); os.makedirs(od, exist_ok=True)
    tag = "diag_" + a.arm + ("_agent" if a.agent else "")
    with open(os.path.join(od, tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", os.path.join(od, tag + ".json"))


if __name__ == "__main__":
    main()