# STAGE 4 -- HANDOFF SYNOPSIS (Person 4: Benchmarks, SHAP) -- 4A and 4B done, 4C hosting built and checked

_Status: 4A (benchmarks + BL validation + ensemble-vs-CE + Hawkes benchmarks + CE misspecification first look) done. 4B (SHAP) run, checks passed. 4C (backend hosting of agents/ensembles) built and checked, see (b)9. Parameters are the simulator's (uncalibrated)._
_Convention: noise-free cost difference vs TWAP in $, negative = cheaper (Stage 3 metric). Compare numbers only on the SAME worlds._

## (a) What was built
- `src/barger_lorig.py`: first-order Barger-Lorig (BL) closed form in simulator units (`f = tau*alpha`, `b = kappa`), truncated at 0. Self-test checks the unit mapping against the code's own algebra.
- `validation/test_bl_consistency.py` (no simulation, all checks pass): checks both BL terms against independent references. Alpha-term vs the continuous-time CE plan (closed form); kappa-term vs a numerical QP plan on grids refined by 1/4/16 with Richardson extrapolation `(4*QP16 - QP4)/3` (first-order convergence assumed from two levels). Test tolerances (C: 0.3*x_k + 0.03) were set after seeing output: calibrated, not independent.
- `validation/run_stage4_benchmarks.py`: plays benchmark policies through the real simulator/IMM-PF on seeds 1,000,000+i, emits `interfaces.EpisodeLog`s (0 contract errors), scores them next to Stage 3 logs. Policies: `twap`, `heuristic_{belief,oracle}`, `bl1_*`, `ce_{oracle,filtered}` and `_grid`. `--obs-kind spread_hawkes` supported. Accepts any logs `.pkl` (`--agent-logs`), including ensemble logs (`ens_<arm>` names left ungrouped).
- `validation/diag_bl.py`: BL diagnostics (single-regime vs regime world, multiplier percentiles, clipped/damped variants).
- **CE benchmark** (`ce_*`): certainty-equivalent plan v_s ~ 1/E[alpha_s] with exact mean forecast (CIR decay + Markov regime), receding horizon. NOT Barger-Lorig; ignores permanent impact and Jensen effects: a reference, not the optimum. `ce_oracle` uses true alpha/regime; `ce_filtered` uses filter beliefs (E[alpha; regime=j] ~ p_j*alpha_hat). `_grid` variants snap to the 7-point Stage 3 grid (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0).
- `validation/test_ce_misspec.py` (no simulation): CE alpha-plan built with misestimated lam/theta (x0.7/1.0/1.3) evaluated under the true mean forecast.
- `validation/run_stage4_shap.py` (4B): exact Shapley (8 coalitions) of the IMM-PF one-step belief update over the 3 proxy channels (spread, volume, realvol); functionals: posterior mean of alpha (units of theta_alpha) and P(stressed). Checks run first and stop the script on failure.

## (b) What the results show

**1. Benchmarks (200 fresh worlds, seeds 1,000,000..1,000,199; TWAP mean IS 4993.1; Gaussian obs; noise-free diff vs TWAP, paired se; CVaR95 on full IS):**
| Policy | d_noisefree (se) | CVaR95 - TWAP |
|---|---|---|
| heuristic_belief / oracle | +8.3 (13.0) / +4.0 (12.8) | +345 / +340 |
| bl1_oracle_regime | +336.9 (28.8) | +110 |
| bl1_filtered_mix | +316.5 (28.7) | +93 |
| bl1_oracle_calm | +427.8 (46.6) | +2263 |
| bl1_oracle_regime_grid | +98.3 (11.9) | +99 |
| ce_oracle | -99.6 (7.3) | +18 |
| ce_oracle_grid | -92.2 (7.1) | +53 |
| ce_filtered | -98.7 (7.3) | +20 |
| ce_filtered_grid | -90.2 (7.1) | +57 |

**2. Barger-Lorig underperforms TWAP** in the regime world (+337). Multiplier ranges 0.01 (1st pct) to 4.18 (99th); 5.1% of steps above 2.0, 0.9% truncated at 0. Cost falls with damping (raw +337, damp 0.5 +31, damp 0.25 -23); the damping factor was chosen on these worlds, so damped rows are exploratory. **Validation (`test_bl_consistency.py`):** BL is the correct small-horizon limit of both references, and its underperformance comes from using the expansion far outside its range (lam*T = 3 calm, 6 stressed), not from a coding error.
- Alpha-term: BL/CE response ratio 1.04 at lam*T_rem = 0.12 (calm), matching the series 1 + x/3 + x^2/36; at horizon start 2.2x (calm) / 3.6x (stressed), even for infinitesimal deviations. Nonlinear case (x = 3, alpha = 2*theta): BL 0.25 vs CE 0.64.
- Kappa-term: extrapolated QP/BL ratio 0.965 (calm/single) and 0.938 (stressed) at n=2; about 0.54 (calm) and 0.36 (stressed) at n=50, i.e. BL overshoots there too.
- Not verified: the paper's printed formula (not re-read here); optimality under stochastic dynamics (references are certainty-equivalent). "single" and "calm" parameter sets are identical, so they count as one piece of evidence. Do not describe BL as "the benchmark"; describe it as a first-order approximation that underperforms TWAP here.

**3. CE is a strong, verified reference.** Beats TWAP by about 2% (-99.6 continuous, -92.2 on grid) with a mild tail. `ce_filtered` is within about 1 $ of `ce_oracle` (paired se not printed). Independently reproduced in the Stage 3 revisit (200 test worlds: -101.5 / -90.8; 500 fresh worlds: grid -95.1 (se 4.3), continuous -102.1 (se 4.4)). The first 200 fresh worlds are 4-7 $ harder for every policy: do not mix 200- and 500-world numbers.

**4. Single trained agents trail CE** (5 training seeds per arm, val-selected; gap to `ce_filtered_grid`, world-only t): filt_full_bonus -78.2 (+12.0, t 4.5), oracle -73.1 (+17.0), filt_mean_only -68.1 (+22.1), filt_full -65.5 (+24.7), filt_mean_bonus -61.7 (+28.5), filt_single -51.2 (+39.0), filt_no_regime -43.9 (+46.2), raw_proxy -3.8 (+86.4). Seed sd 4.6 to 27.3.

**5. Ensembles (mean Q of an arm's 5 val-selected agents; Stage 3 revisit) close most of the gap. Paired ensemble minus `ce_filtered_grid`, ALL 8 arms, first 200 fresh worlds (negative = agent cheaper; CE uses the TRUE state):**
| Arm | ensemble d_nf | gap vs ce_filtered_grid (t) | CVaR95-TWAP |
|---|---|---|---|
| filt_full_bonus | -88.9 | +1.2 (0.46) | +82 |
| oracle | -85.5 | +4.7 (1.15) | +69 |
| filt_full | -80.6 | +9.6 (4.41) | +99 |
| filt_mean_only | -77.4 | +12.8 (3.04) | +85 |
| filt_mean_bonus | -73.3 | +16.9 (4.86) | +100 |
| filt_single | -68.5 | +21.7 (4.13) | +95 |
| filt_no_regime | -48.9 | +41.3 (9.09) | +56 |
| raw_proxy | -43.6 | +46.6 (9.06) | +196 |
- Only filt_full_bonus and oracle are statistically indistinguishable from CE on the grid; the other six trail it. Vs continuous `ce_oracle` all ensembles trail by +10.7 to +56.0 (t > 3.9), about 10 $ of which is the 7-point grid cost. All ensembles beat `heuristic_belief` by 52-97 $.
- On 500 fresh worlds (Stage 3 revisit) the two-arm paired figures were oracle +5.8 (t 2.38), filt_full_bonus +1.4 (t 0.85). filt_full_bonus was chosen as best after seeing results: the most optimistic reading.
- Claim to make: "the best ensembles are statistically indistinguishable from CE on the same grid". Do not claim they beat CE, or that filtered arms in general match it.
- Seed-level ablation contrasts on ensembles (Welch over 5 members, Holm over 7): only filtered-vs-raw-proxy is significant (fresh p 0.032; test 0.066). Regime layer, uncertainty features and exploration bonus are not established (see revisit handoff).

**6. Hawkes observations (`--obs-kind spread_hawkes`, same 200 fresh worlds).**
Benchmarks: oracle rows identical to Gaussian. `ce_filtered` -98.5 vs `ce_oracle` -99.6 (Gaussian -98.7); `ce_filtered_grid` -90.8 (Gaussian -90.2); differences far below the world se (~7). CVaR95 vs TWAP of `ce_filtered` rises from +20.0 to +29.9 and of `ce_filtered_grid` from +56.6 to +68.1 (no se). So the Gaussian filter under Hawkes observations costs the CE rule little in mean cost and modestly in the tail.

Hawkes-trained ensembles (4 arms, 5 seeds; `--agent-logs ...logs_ensemble_valsel_spread_hawkes.pkl`; ensemble minus CE, paired over worlds, CE uses the TRUE state):
| Arm | ensemble d_nf | gap vs ce_filtered_grid (t) | gap vs ce_oracle (t) | CVaR95-TWAP | Gaussian ensemble gap vs ce_filtered_grid |
|---|---|---|---|---|---|
| filt_full | -76.8 | +14.0 (2.94) | +22.8 (5.02) | +106.0 | +9.6 |
| oracle | -74.9 | +15.9 (3.85) | +24.8 (6.13) | +86.6 | +4.7 |
| raw_proxy | -68.9 | +21.9 (7.97) | +30.7 (9.72) | +66.5 | +46.6 |
| filt_mean_only | -63.3 | +27.5 (5.94) | +36.3 (7.58) | +148.4 | +12.8 |
All four trail CE significantly and all beat `heuristic_belief` by 75-88 $. Versus Gaussian the Hawkes ensembles trail CE more for the filtered arms and oracle, and far less for raw_proxy (unexplained, see `STAGE3_HANDOFF.md`; separately trained agents). Single-agent and ensemble Hawkes numbers on the test/500-fresh worlds are in `STAGE3_HANDOFF.md` and `STAGE3_REVISIT_HANDOFF.md`. Seeds 3-4 of the Hawkes agents were added after the 3-seed results were seen.
**An earlier run of the same command WITHOUT `--agent-logs` printed the Gaussian-trained agent logs under the Hawkes heading: those agent rows and its head-to-head table are not Hawkes results; do not use them.**

**7. CE parameter misestimation, first look** (`test_ce_misspec.py`; alpha-only, current alpha known, deterministic mean-path cost). Plans with lam or theta off by +-30% lose at most 1.3% of cost vs the correct plan and never fall below TWAP. Lam errors alone keep 97-99% of CE's gain over TWAP; theta errors alone keep 32-87%; both in the same direction keep as little as 4% (stressed, alpha 50% above mean, both x0.7). Theta sensitivity is not monotone in the alpha deviation (kappa-free ratio of two moving quantities): write "theta errors can remove up to about two thirds of the gain in some states". CE's gain is only about 2% of cost, so a 1% planning error removes a large share. Conclusion: true-parameter CE is an optimistic reference, mainly through theta. This does NOT show agents are advantaged (they are trained inside the same simulator). Not tested: kappa, regimes, the filter, simulator IS, tails.

**8. 4B SHAP (Stage 4B, run).** 200 worlds (seeds 2,000,000+), 2000 particles/regime, bootstrap CIs over worlds. Checks: Shapley toy games PASS; full coalition vs the filter's real belief max error 0; efficiency error <= 9e-16.
| Share of mean abs Shapley (all steps) | spread | volume | realvol |
|---|---|---|---|
| Gaussian, alpha posterior mean | 0.44 | 0.42 | 0.14 |
| Gaussian, P(stressed) | 0.52 | 0.29 | 0.18 |
| Hawkes, alpha posterior mean | 0.48 | 0.42 | 0.10 |
| Hawkes, P(stressed) | 0.51 | 0.30 | 0.19 |
- Spread leads regime detection, up to 0.66 (Gaussian) / 0.64 (Hawkes) on switches into calm; realvol matters most in the stressed regime for alpha (0.17 vs 0.12 calm).
- Hawkes shares barely move but absolute magnitudes and CIs rise (spread mean |phi| for alpha 0.037 -> 0.064; volume 0.036 -> 0.056, unexplained: do not claim the change is spread-only). Possible link to the larger CE tail under Hawkes is a hypothesis, untested.
- Mean |interaction| (0.091 Gaussian, 0.133 Hawkes) exceeds mean |update| (0.077 / 0.083) for alpha: shares are summaries of Shapley credit, not additive contributions; quote that caveat with any share.
- Signed means describe how the filter's update tends to move, not world bias.

**9. 4C (backend hosting; built by the team, reviewed here statically except where a check is cited).** `agent_policies.py` (188 lines) hosts single agents (`ddqn_<arm>_s<seed>`) and ensembles (`ens_<arm>`) behind the Policy protocol (`PolicyInput.obs`). Gaussian arms hosted: oracle, raw_proxy, filt_full, filt_no_regime, filt_full_bonus (checked); Hawkes arms: oracle, raw_proxy, filt_full, filt_mean_only. An ensemble needs all 5 `_hawkes`/Gaussian checkpoints, otherwise it is absent from `agent_names()` and a direct request fails with "checkpoint missing". Bonus, no-regime and mean-bonus arms are rejected on Hawkes. `check_agent_policies.py`: ALL CHECKS PASSED: live obs, trade path and IS equal the environment's exactly (max differences 0.0) for single agents and ensembles on both observation kinds; information barrier holds; step responses pass the contract; live noise-free cost equals the saved `ens_nf` (Gaussian oracle and filt_full_bonus, Hawkes oracle and filt_full; max gap 2.7e-12); stressed-regime band guard on an ensemble (H1: multiplier = clip(raw, 0.75, 1.25) when stressed; 0 violations in 42 stressed steps; one world); error cases E1-E7. `dashboard.html` (edited for Hawkes, only the inline script syntax-checked, NOT opened in a browser): the Agents default follows the Obs arm (gaussian `ens_filt_full_bonus,ens_oracle,ens_raw_proxy`; spread_hawkes `ens_filt_full,ens_oracle,ens_raw_proxy,ens_filt_mean_only`; user-typed lists are kept), a Hawkes note says the agents were trained on Hawkes observations, exploratory, filter overconfident. Do not read the card order as a finding (`ens_filt_mean_only` is the weakest Hawkes ensemble). Caveats: live hosting runs the filter at 500 particles per regime (agents trained/scored on 2000; `AGENT_TRAIN_PARTICLES = 2000` is defined in `agent_policies.py` but unused there); the dashboard calls `http://localhost:8000` from a page served on port 5500, so the backend must allow CORS. Not seen by the editor of this file: `live_session.py`, the HTTP server code, the guard's code and whether it is active by default (equality with the saved scores suggests it is not).

## (c) Interface produced
- `outputs/stage4/benchmarks.json` and `benchmarks.md` (rows: name, IS_mean, d_full, se_full, d_nf, se_nf, cvar_diff; BL zero-clip shares; arm list). The JSON holds the benchmark rows and the ensemble arm list only; the ensemble rows and head-to-head tables are only in the `.md`. **Files in `outputs/stage4` come from the LAST run written there, which was a Hawkes run, so the Gaussian benchmark/ensemble files were overwritten.** The Hawkes ensemble run is preserved in `outputs\stage4_ens_hawkes` (`benchmarks.json/.md`, obs=spread_hawkes, 4 `ens_*` arms). Always pass `--outdir` per run.
- `outputs/stage4/shap_gaussian.{json,png}`, `shap_spread_hawkes.{json,png}`.
- Benchmark logs are `interfaces.EpisodeLog`s (`episode_log_from_episode(name, seed, wcfg, ep)`); `nf_cost` equals the Stage 3 noise-free metric (TWAP match 4.5e-12).
- Stage 3 single-agent logs `outputs/stage3c/logs/logs_valsel.pkl`; ensemble logs `outputs/stage3c/logs/logs_ensemble_valsel.pkl` (TWAP + `ens_<arm>`, 200 fresh worlds); unpickling needs `..\src` on the path.
- Hawkes ensemble logs: `outputs/stage3c/logs/logs_ensemble_valsel_spread_hawkes.pkl` (TWAP + `ens_oracle`, `ens_raw_proxy`, `ens_filt_full`, `ens_filt_mean_only`; verified 3.6e-12). Gaussian and Hawkes logs must be scored with the matching `--obs-kind`.
- World blocks (do not mix): train 100000*(seed+1)+episode, validation 800000+, test 900000+ (200), fresh 1,000,000+ (200 in 4A, 500 in the revisit), grid design 1,500,000+, SHAP 2,000,000+.
- Commands: `python run_stage4_benchmarks.py --n 200 [--obs-kind spread_hawkes] [--agent-logs <pkl>]`; `python run_stage4_shap.py [--obs-kind spread_hawkes] [--n N]`; Hawkes ensembles: `python run_stage4_benchmarks.py --n 200 --obs-kind spread_hawkes --agent-logs outputs\stage3c\logs\logs_ensemble_valsel_spread_hawkes.pkl --outdir outputs\stage4_ens_hawkes`.

## (d) Assumptions, limits, open items
- CE rule and filter use the simulator's true parameters; CE comparators use the TRUE state, so agent-vs-CE gaps mix agent quality with filter noise. Parameter misestimation tested only in the alpha-only first look (item 7).
- Ensemble idea arose after results were seen; nothing tuned, not pre-registered; ensemble contrasts are exploratory. One ensemble of 5 per arm: world se says nothing about retraining variability. Inference cost 5x.
- Hawkes: 4 arms only (oracle, raw_proxy, filt_full, filt_mean_only), 5 seeds (seeds 3-4 added after seeing 3-seed results); no K = 1, no-regime or bonus arms; no seed-level contrasts on Hawkes ensembles; the filter is overconfident there (alpha cov90 0.54).
- CE is a reference, not the optimum; non-causal ceiling about -194, unreachable causally. Tail risk: ensembles are worse than TWAP in CVaR95 for every arm (+51 to +196) and often worse than CE.
- Benchmark inference is over worlds only; noise-free and full-IS columns agree within about 2 $. Heuristic knobs were tuned on a single-regime world; its oracle mean is unstable across world sets (+4.0 vs +50.6): never quote it as one number.
- SHAP describes how THIS filter reacts (one step, along the true filter path), not causal informativeness or what agents rely on; one filter seed, so filter-seed variability is not in the CIs; do not rank spread vs volume where CIs overlap (stressed, Hawkes).
- **Still open:** (1) rerun the Gaussian benchmark with its own output dir: `python run_stage4_benchmarks.py --n 200 --agent-logs outputs\stage3c\logs\logs_ensemble_valsel.pkl --outdir outputs\stage4_gaussian`, then restore/rename `outputs/stage4` files; (2) 4C: open `dashboard.html` in a browser, confirm CORS and the 500- vs 2000-particle choice, confirm the band guard's default state and purpose, ratify `PolicyInput.obs` and rerun `validate_interfaces.py` (Stage 3 (d)10); (3) single (a)/(b)/(c) ablation reporting script (pieces exist: `compare_arms.py`, `ens_contrasts.py`); (4) optional: Hawkes `ens_contrasts` and paired ensemble-vs-CE on the 500-world blocks, simulated CE misspecification (needs a `param_scale` option in the `ce_*` policy code), re-reading the paper's BL formula.