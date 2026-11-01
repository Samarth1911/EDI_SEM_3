# STATUS SUMMARY

_Updated from the Stage 3, Stage 3 revisit and Stage 4 handoffs (3 Oct 2026). Rows for Stages 0-2 and 5 are carried over from the earlier summary and were not re-checked. Update the row for your stage when its status changes._

| Stage | Owner | Status | Evidence in the repo |
|---|---|---|---|
| Milestone 1 (~35%): simulator, Gaussian observation model, bootstrap PF, heuristic, dashboard | (existing) | Validated | `validation/validation_summary.json`, `validation/policy_comparison_summary.json`, plots 01-03 |
| 0 -- Shared interfaces | team | **Draft, not ratified.** `validate_interfaces.py`: 20/20 conformance checks (before the `obs` field was added). `PolicyInput` now has an optional trailing `obs` field (resolved in Stage 4C; `check_agent_policies.py` X1 passes). **Still needed: team ratification of the `obs` field and Decisions 1-5, and a rerun of `validate_interfaces.py`.** | `STAGE0_INTERFACES.md` (Section 10.2), `src/interfaces.py` |
| 1 -- World generation | Person 1 | Code complete; HARD checks pass per its handoff; parameters uncalibrated | `STAGE1_HANDOFF.md`, `validation/outputs/` figures 04-06, 08 |
| 2 -- Estimation (IMM-PF) | Person 2 | **Complete for the brief. 10/10 HARD checks pass. Gaussian arm: alpha RMSE 2.5e-6 vs 1.8e-5 bootstrap, cov90 0.94, regime accuracy 0.984. Spread-Hawkes arm: overconfident (alpha cov90 0.54).** | `src/imm_particle_filter.py`, `src/belief_metrics.py`, `validation/validate_imm_pf.py`, `STAGE2_HANDOFF.md`, `validation/outputs/stage2_results.md` |
| 3 -- Decision (belief-MDP, Double DQN) | Person 3 | **Complete, including the revisit.** 8-arm Gaussian ablation, 5 seeds, 30000 episodes: validation-selected checkpoint beats TWAP in 7 of 8 arms (oracle -67.7 $, filt_full -61.0 $, raw_proxy -9.7 $ n.s.; fresh -76.9, -70.0, -11.3). Only significant ablation: filtered belief vs raw proxy (-51.3 $, t -3.37; fresh -58.7 $, t -4.72). Uncertainty features, bonus and regime layer: no significant effect. Final checkpoint never beats TWAP; CVaR95 worse than TWAP for every arm. **Revisit:** a finer action grid gave a null; averaging the 5 agents' Q-values (ensemble) beats the average member in every arm and for the best arms is statistically indistinguishable from the CE rule on the grid (exploratory, not pre-registered). **Spread-Hawkes arm now trained** (oracle, raw_proxy, filt_full, filt_mean_only; 5 seeds, seeds 3-4 added after the 3-seed results were seen) with ensembles: single agents -31.5 to -60.9 $ (test), ensembles -61.8 to -77.8 $; no contrast confirmed. | `STAGE3_HANDOFF.md`, `STAGE3_REVISIT_HANDOFF.md`, `src/belief_mdp_env.py`, `src/ddqn.py`, `src/stage3_adapter.py`, `validation/run_stage3c_multiseed.py`, `validation/compare_arms.py`, `validation/score_ensemble.py`, `validation/export_ensemble_logs.py`, `validation/outputs/stage3c/` (incl. `hawkes_report.md`, `logs_ensemble_valsel*.pkl`) |
| 4 -- Evaluation + backend | Person 4 | **4A done, 4B done, 4C hosting built and checked; reporting script open.** 4A: TWAP, heuristic, Barger-Lorig and CE benchmarks on 200 fresh worlds; CE (-99.6 $ continuous, -92.2 $ on the grid) is the strong reference; BL loses to TWAP (+337 $) and `test_bl_consistency.py` shows this is the expansion's range (lam*T of 3-6), not a coding error; paired ensemble-vs-CE for all 8 Gaussian arms (only filt_full_bonus +1.2 and oracle +4.7 are indistinguishable from CE on the grid) and for the 4 Hawkes ensembles (all trail CE by +14 to +28 $); Hawkes benchmarks; first-look CE misspecification (theta errors matter, lam errors do not). 4B: exact Shapley attribution of the filter update (spread and volume dominate alpha; spread leads regime detection), Gaussian and Hawkes. 4C: agents and ensembles hosted behind the Policy protocol, `check_agent_policies.py` ALL CHECKS PASSED; `dashboard.html` edited for Hawkes but not opened in a browser. | `STAGE4_HANDOFF.md`, `src/barger_lorig.py`, `validation/run_stage4_benchmarks.py`, `validation/run_stage4_shap.py`, `validation/test_bl_consistency.py`, `validation/test_ce_misspec.py`, `agent_policies.py`, `check_agent_policies.py`, `dashboard.html`, `validation/outputs/stage4/`, `stage4_ens_hawkes/` |
| 5 -- Frontend rebuild | whole team | Not started per the earlier summary (milestone-1 dashboard is client-side, single-regime Gaussian only); not re-checked | `frontend/` |

## Stage 2: what is left

1. Team ratification of Stage 0 Decisions 2 and 3 (Stage 2 implements them as proposed).
2. The Hawkes arm has so far been run as an overconfident-filter ablation (default `obs_noise_inflation = 1.0`); no inflation sweep or bias-aware likelihood was used downstream. Decide whether that is the final treatment.
3. Optional follow-ups listed in `STAGE2_HANDOFF.md` (d) item 11.

## Known provisional items (Stage 2)

- Defaults not tuned: 2000 particles per bank, `jump_prob = 0.03`, `obs_noise_inflation = 1.0` (sweeps are in-sample).
- Gaussian likelihood on core channels only; Hawkes extras unused; no price-return input. Measured consequence: biased, overconfident filter on the spread-Hawkes arm.
- Filter knows the true regime model; two misspecified-model sweeps are included as sensitivity tests.
- No oracle-regime filter as an upper bound for detection quality.

## Stage 3: what is left

1. Hawkes arm: only 4 arms exist (no K = 1, no-regime or bonus arms); no seed-level contrasts on the Hawkes ensembles; fresh-world contrasts for the 5 seeds not seen (`fresh_scores_hawkes.json` may still hold 3 seeds). Unexplained: the raw_proxy ensemble is far better on Hawkes than on Gaussian (-70.8 vs -45.9 $, test), which shrinks the filtered-vs-raw gap; do not interpret without a Hawkes contrast run.
2. Optional: other bonus beta values, a dueling head or other remedy for checkpoint oscillation, more training seeds (limited to 0-6; disclose as optional stopping).
3. Hosting at 2000 particles instead of 500 (see Stage 4).

## Known provisional items (Stage 3)

- Primary metric (validation-selected checkpoint) was fixed before any ablation result existed but after the oracle's final-checkpoint metric failed; report both metrics for every arm.
- 5 training seeds per arm; contrast se is 4-12 $, so only effects above about 12-27 $ are detectable. 14 paired contrasts examined, about 0.7 chance positives expected.
- Fresh-world scores re-score the same trained agents on new worlds: they remove world-sampling noise but are not independent replications over training seeds.
- The ensemble idea arose after results on test and fresh worlds were seen; nothing was tuned, but it is not pre-registered. Ensemble inference is over worlds only (one ensemble per arm) and costs 5x. Hawkes seeds 3-4 were added after the 3-seed results were seen.
- Exploration bonus is a heuristic (beliefs do not depend on the agent's trades); only beta = 1.0 was run.
- Network and hyperparameters are untuned and do not claim to match Macri and Lillo (2025).
- Cached beliefs rely on world/proxy/belief independence from the agent's actions, verified for the Gaussian and the Hawkes arm (`check_independence_hawkes.py` PASS).
- The CE rule and the filter use the simulator's true parameters, and the CE comparator uses the true state; the first-look misspecification test (alpha-only, deterministic) says true-parameter CE is an optimistic reference, mainly through theta.

## Stage 4: what is left

1. Rerun the Gaussian benchmark into its own folder (the Gaussian files in `outputs/stage4` were overwritten by a Hawkes run): `python run_stage4_benchmarks.py --n 200 --agent-logs outputs\stage3c\logs\logs_ensemble_valsel.pkl --outdir outputs\stage4_gaussian`.
2. 4C: open `dashboard.html` in a browser; confirm CORS (page on port 5500 calls `http://localhost:8000`) and the 500- vs 2000-particle live hosting; confirm the purpose and default state of the stressed-regime multiplier band guard.
3. Single (a)/(b)/(c) ablation reporting script (pieces exist: `compare_arms.py`, `ens_contrasts.py`).
4. Optional: simulated CE misspecification (needs a `param_scale` option in the `ce_*` policy code), Hawkes `ens_contrasts` and paired ensemble-vs-CE on the 500-world blocks, re-reading the Barger-Lorig paper's printed formula.

## Known provisional items (Stage 4)

- BL is a first-order approximation that underperforms TWAP here; describe it that way, not as "the benchmark". Damped-BL rows are exploratory (damping chosen on the same worlds). Test thresholds in `test_bl_consistency.py` were set after seeing output.
- CE is a reference, not the optimum (non-causal ceiling about -194 $). Best ensembles match CE on the grid, none beats it.
- 200-world and 500-world numbers must not be mixed (the first 200 fresh worlds are 4-7 $ harder for every policy).
- SHAP describes how this filter reacts, one step at a time, not causal informativeness or what the agents use; channel shares come with large interaction terms; one filter seed.
- Hawkes ensemble comparisons use the first 200 fresh worlds only.