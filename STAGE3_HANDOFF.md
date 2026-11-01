# STAGE 3 -- DECISION (Person 3): belief-MDP + Double DQN -- FINAL

**Status: COMPLETE.** An 8-arm ablation covering (a) oracle vs raw-proxy vs filtered, (b) mean vs mean + uncertainty (and the exploration bonus), (c) with vs without the regime layer was run with 5 training seeds per arm on Gaussian observations. A follow-up revisit (action grid, ensembling, CE comparison) is documented in `STAGE3_REVISIT_HANDOFF.md`. The spread-Hawkes observation model was run as an exploratory 4-arm robustness check (3 seeds first, seeds 3-4 added later, so 5 seeds now) and its 5-checkpoint ensembles were scored (section below). Depends on Stage 0 (`src/interfaces.py`), Stage 1 (simulator, observation models) and Stage 2 (`IMMParticleFilter`: `STAGE2_HANDOFF.md`). Consumed by Stage 4: 4A (benchmarks), 4B (SHAP) and 4C (backend with the trained agents hosted) are reported done.

## Files

| File | Role | Status |
|---|---|---|
| `src/belief_mdp_env.py` | `BeliefMDPEnv`, `EnvConfig`: Gym-style env wrapping `RegimeSwitchingSimulator.run_stepwise` + observation model + `IMMParticleFilter`. Observation layout per arm, reward, action grid | self-test passes; used in every run |
| `src/env_factory.py` | `make_env(cfg, obs_kind, filter_regimes, seed, imm_params, stationary_init=...)` | used |
| `src/ddqn.py` | Double DQN in numpy (MLP 64x64, Huber loss, Adam + global-norm clipping, uniform replay, hard target sync counted in GRADIENT steps); `q_values`, `greedy`, `save`, `load` | self-test 7/7 passes |
| `src/ddqn_train.py`, `ddqn_train2.py`, `ddqn_train3.py` | training loops: baseline, validation protocol + best-checkpoint, TWAP reward baseline + optional bonus. `run_episodes`, `paired` = the evaluation harness | used |
| `src/ddqn_train4.py` | `train_v4` = `train_v3` + optional EMA of the online weights | run once (oracle, 3 seeds); no benefit shown; not used for the ablation |
| `src/stage3c_bonus.py` | uncertainty bonus tied to the filter's posterior std, added to Q at action selection only | used in the two bonus arms |
| `src/belief_cache.py` | cached-belief env (replays stored world/proxy/belief per world seed, rebuilds only the price path) | exactness test passes (Gaussian and spread-Hawkes) |
| `src/stage3_adapter.py` | env episode -> `interfaces.EpisodeLog`; `run_logs(env, seeds, agent, name, obs_kind)` | self-test passes; used by both exports |
| `validation/run_stage3c_multiseed.py` | multi-seed driver for all 8 arms (`--use-cache`, `--ema`, `--grid`, `--obs-kind`, checkpoints saved) | used for all results |
| `validation/build_belief_cache.py` | builds belief-cache chunks with the real filter (resumable) | used |
| `validation/report_stage3c.py`, `compare_arms.py` | per-arm vs TWAP and paired contrasts for (a), (b), (c), both metrics | used (`ablation_report.md`, `hawkes_report.md`) |
| `validation/score_fresh.py`, `fresh_contrasts.py` | checkpoints on fresh worlds (seeds 1,000,000+) and paired contrasts | used |
| `validation/export_agent_logs.py` | single agents + TWAP -> `outputs/stage3c/logs/logs_valsel.pkl` | used |
| `validation/score_ensemble.py`, `ens_vs_ce.py`, `ens_contrasts.py`, `export_ensemble_logs.py`, `check_ens_logs.py` | revisit tools: 5-checkpoint mean-Q ensembles, paired ensemble vs CE, ensemble ablation contrasts, ensemble `EpisodeLog` export and its check | used (see `STAGE3_REVISIT_HANDOFF.md`) |
| `validation/compare_grids.py`, `diag_ce_ref.py`, `check_grid_env.py`, `diag_grid.py`, `diag_agent_vs_ce.py` | revisit diagnostics (grid experiment, CE reference) | used (`diag_grid.py` and `diag_agent_vs_ce.py` outputs not re-checked in this document) |
| `validation/selftest_env.py`, `selftest_ddqn.py`, `selftest_cached_env.py`, `check_independence_hawkes.py` | self-tests | all pass |
| `validation/patch_driver_cache.py`, `patch_fix1.py` | one-off patch scripts | `patch_fix1.py` application not confirmed; the exports ran correctly |

`diag_headroom.py` (original version that produced the first CE reference numbers) was overwritten by a different tool; `diag_ce_ref.py` reproduces those numbers and matches Stage 4A. Keep the original in version control if it exists there.

## Run commands (PowerShell, from `EDI\validation`, venv active)

```
# caches (resumable; ~85 s per 1000 worlds with the IMM filter, ~48 s with the single filter)
python build_belief_cache.py --filter imm --what val test
python build_belief_cache.py --filter single --what val test
python build_belief_cache.py --filter imm --what train --train-seed <0..4> --episodes 30000
python build_belief_cache.py --filter single --what train --train-seed <0..4> --episodes 30000

# ablation arms (about 10-11 min per arm-seed with the cache)
python run_stage3c_multiseed.py --arms oracle,raw_proxy --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --tag _abl
python run_stage3c_multiseed.py --arms filt_full,filt_mean_only,filt_no_regime,filt_single --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --use-cache --tag _abl
python run_stage3c_multiseed.py --arms filt_full_bonus,filt_mean_bonus --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --use-cache --bonus-beta 1.0 --tag _abl_b1

# spread-Hawkes arm (3 seeds first; seeds 3-4 were added later: build the train caches for --train-seed 3 4 and rerun with --seeds 5, finished pairs are skipped; the exact seed 3-4 commands were not shown to the editor of this file, confirm)
python build_belief_cache.py --filter imm --obs-kind spread_hawkes --what val test
python build_belief_cache.py --filter imm --obs-kind spread_hawkes --what train --train-seed <0..2> --episodes 30000
python run_stage3c_multiseed.py --arms oracle,raw_proxy,filt_full,filt_mean_only --seeds 3 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --obs-kind spread_hawkes --use-cache --tag _hawkes

# reports and exports
python compare_arms.py --tags _abl _abl_b1 --out-name ablation_report
python score_fresh.py --tags _abl _abl_b1 --n-fresh 500 --use-cache --out-name fresh_scores
python fresh_contrasts.py
python export_agent_logs.py --tags _abl _abl_b1 --which valsel --n 200
python export_ensemble_logs.py --tags _abl _abl_b1
python check_ens_logs.py

# spread-Hawkes reports, ensembles and exports (5 seeds)
python compare_arms.py --tags _hawkes --out-name hawkes_report
foreach ($a in "oracle","raw_proxy","filt_full","filt_mean_only") { python score_ensemble.py --arm $a --obs-kind spread_hawkes }
python export_ensemble_logs.py --tags _hawkes
python check_ens_logs.py --obs-kind spread_hawkes
python run_stage4_benchmarks.py --n 200 --obs-kind spread_hawkes --agent-logs outputs\stage3c\logs\logs_ensemble_valsel_spread_hawkes.pkl --outdir outputs\stage4_ens_hawkes
python check_agent_policies.py
```
Results append to `outputs/stage3c/results_base<0|1><tag>.jsonl` and are keyed by tag only: **change `--tag` whenever a hyperparameter changes.** Finished (arm, seed) pairs are skipped on rerun. Training seeds are limited to 0-6 by the world-seed range assertions. Keep the machine awake: one run took 16712 s instead of about 660 s when it was suspended (training is deterministic, so results were unaffected). `score_ensemble.py` now has `--obs-kind` (default gaussian): for the four Hawkes arms use `--obs-kind spread_hawkes` (tag `_hawkes`, saves `ens_nf_<arm>_hawkes.npz`; other arms exit with an error). `export_ensemble_logs.py` refuses to mix observation kinds: export Gaussian and Hawkes tags separately.

## Problem definition (as implemented)

- **Action**: index into `action_multipliers = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)`; shares sold `v_t = min(mult * q_t / (N - t), q_t)`; multiplier 1.0 is TWAP. At the last step everything left is sold (a forced sale, not a choice).
- **Reward**: `-1e4 * cost_t / (S0*q0) * reward_scale` (bps of initial value; `reward_scale` 0.1 in the driver). `cost_t` = the drift-only decomposition `alpha_t v^2 + kappa_t v (q - v)` (`reward_mode="decomp"`), whose sum equals the noise-free part of implementation shortfall (self-test: IS minus sum of decomposition has z = -0.27 over 20 seeds). The exact IS increment is available as `reward_mode="is"`.
- **TWAP reward baseline** (`baseline=True`, driver default): each step's reward has the cost TWAP would have paid in that step subtracted, using the true `alpha_t, kappa_t` in the reward only. It depends on the exogenous world and `t`, not on the agent's action, so it cannot change the optimal policy (`_baseline_reward` uses only t, q0, N and the true alpha_t, kappa_t; `twap_baseline_residual` prints 0.00e+00 but is tautological).
- **Observation** (normalised by fixed reference scales; `std_scale = 10` on the std features, not tuned):

| arm (driver name) | layout | dim |
|---|---|---|
| `oracle` | q/q0, time left, spread proxy, TRUE alpha, TRUE kappa, one-hot TRUE regime | 7 |
| `raw_proxy` | q/q0, time left, spread, volume, realised vol | 5 |
| `filt_full` | q/q0, time left, spread, per-regime (alpha_mean, alpha_std, kappa_mean, kappa_std), regime probabilities | 13 |
| `filt_mean_only` | as `filt_full` without the std features | 9 |
| `filt_no_regime` | q/q0, time left, spread, POOLED alpha/kappa mean and std, no regime probabilities | 7 |
| `filt_single` | as `filt_full` with a K = 1 bootstrap-style filter | 8 |

  `filt_full_bonus` / `filt_mean_bonus` add the exploration bonus to `filt_full` / `filt_mean_only`. The oracle arm sees the true regime; filtered arms see only the IMM-PF belief.
- **Seeds**: training worlds `100000*(seed+1)+episode`, validation worlds `800000+i` (300), test worlds `900000+i` (200), fresh-score worlds `1000000+i` (500), design worlds for the grid diagnostic `1500000+`. Disjoint by assertion. Within a world: observation model `seed+10000`, filter `seed+20000`.
- **Agent defaults** (untuned): MLP 64x64, lr 1e-3, gamma 1, batch 128, buffer 100k, warm-up 2000 steps, one gradient step per 4 env steps, target sync 2000 gradient steps, epsilon 1.0 -> 0.05 over 30% of episodes, Huber delta 1, gradient clip 10, validation every 200 episodes with best-checkpoint selection on the paired noise-free difference to TWAP. Macri & Lillo's network and hyperparameters are unknown to the author.
- **Exploration bonus** `bonus_a = (sigma_alpha v_a^2 + sigma_kappa v_a (q - v_a)) * 1e4/(S0 q0) * reward_scale`, beta = 1.0 annealed to zero over the first 80% of training; added at action selection only, never in the TD target. It is a training-time diversification heuristic, not information gathering: the agent's trades do not change observations or beliefs in this simulator.
- **Cached beliefs**: world, proxies and beliefs do not depend on the agent's trades (verified for the Gaussian and the spread-Hawkes observation models); only the price path does and it is rebuilt from the stored TWAP price noise. Cached env vs real env: max |obs|, |IS|, |S| differences 0.0 (K = 2 and K = 1 filters, all observation modes); a 400-episode run gives identical results with the real and cached env (175 s vs 9 s).

## Results (single agents)

Setup: Gaussian observations, 30000 training episodes, 5 training seeds per arm (0-4), target sync 2000, no EMA, 200 test worlds, 500 fresh worlds. Values are paired noise-free differences vs TWAP in $ (negative = better). **PRIMARY metric** = validation-selected checkpoint scored on held-out worlds (VAL-SEL); **SECONDARY** = final checkpoint (LAST). The primary metric was fixed before any ablation result existed but after the oracle's final-checkpoint metric failed.

| Arm | VAL-SEL test | VAL-SEL fresh | LAST test (sd over seeds) | VAL-SEL CVaR95 diff, fresh |
|---|---|---|---|---|
| oracle | -67.7 | -76.9 | +4.5 (16.7) n.s. | +101 |
| raw_proxy | -9.7 (n.s.) | -11.3 | +72.4 (35.9) worse | +135 |
| filt_full | -61.0 | -70.0 | +70.4 (182.9) n.s. | +91 |
| filt_mean_only | -63.4 | -73.0 | -10.7 (28.2) n.s. | +117 |
| filt_no_regime | -49.0 | -48.5 | +58.9 (22.6) worse | +84 |
| filt_single | -50.1 | -54.9 | +12.0 (18.6) n.s. | +130 |
| filt_full_bonus | -73.3 | -82.0 | +0.6 (53.8) n.s. | +103 |
| filt_mean_bonus | -56.2 | -68.1 | +137.0 (309.2) n.s. | +166 |

Every arm except raw_proxy has a 95% CI (test worlds) excluding zero at the validation-selected checkpoint. The ranking of the arms is identical on test and fresh worlds and fresh values are 0-12 $ better, so configuration choices made on the test worlds did not inflate the headline numbers.

### Paired contrasts by training seed (A - B, negative = A cheaper; n = 5; |t| > 2.776 needed)

| Ablation | Contrast | test: mean (t) | fresh: mean (t) |
|---|---|---|---|
| (a) | filt_full - raw_proxy | -51.3 (-3.37) **significant** | -58.7 (-4.72) **significant** |
| (a) | oracle - filt_full | -6.7 (-1.06) n.s. | -6.9 (-0.93) n.s. |
| (b) | filt_full - filt_mean_only (uncertainty features) | +2.5 (+0.56) n.s. | +3.0 (+0.48) n.s. |
| (b) | filt_mean_bonus - filt_mean_only (bonus) | +7.2 (+1.03) n.s. | +4.9 (+0.74) n.s. |
| (b) | filt_full_bonus - filt_full (bonus) | -12.3 (-2.16) n.s. | -12.0 (-2.22) n.s. |
| (c) | filt_full - filt_no_regime (regime probs masked) | -12.0 (-1.22) n.s. | -21.4 (-1.77) n.s. |
| (c) | filt_full - filt_single (K = 1 filter) | -10.9 (-2.31) n.s. | -15.0 (-2.66) n.s. (just under threshold) |

Final-checkpoint contrasts: all |t| <= 1.12 on both world sets; no arm pair is separable on that metric. With a Welch test and Holm adjustment over the 7 contrasts (`ens_contrasts.py` seed level) only filtered-vs-raw-proxy survives: test Holm p 0.066, fresh 0.032. Name which test you quote.

### Revisit findings (details in `STAGE3_REVISIT_HANDOFF.md`)

- **Action grid**: a finer, narrower grid gave no improvement (oracle arm, 5 seeds: VAL-SEL -63.3 vs -67.7, Welch p 0.49; difference +4.3 with 95% CI -9.8 to +18.4; final checkpoint -20.8 vs +4.5, p 0.13). CE reference on the new grid (test worlds 900000+; cost-model identity residual 1.4e-14, 0/200 non-converged solves): CE snapped to the grid -98.5 (se 8.0) against continuous CE -101.5 (se 7.6), so the grid is not the bottleneck for CE, and the agent recovers about 64% of the new-grid CE reference against 75% of the old one (-67.7 / -90.8): the shortfall is a learning problem, not an action-space problem. Default grid kept.
- **Ensembling** (mean Q of an arm's 5 validation-selected agents): beats the average member in all 8 arms on both world sets (6-39 $, |t| >= 3.1). Best arm (filt_full_bonus) ensemble: -87.9 (test) / -93.7 (fresh), paired gap to a CE rule on the same grid +2.9 / +1.4 (not significant); oracle ensemble still trails CE by +8.6 / +5.8 (t about 2). Exploratory: the idea came after seeing results.
- **CE reference** (diag_ce_ref.py; matches Stage 4A to the decimal): CE on the grid -90.8 (test) / -92.2 (first 200 fresh) / -95.1 (500 fresh); continuous CE -101.5 / -99.6 / -102.1; non-causal ceiling -193.4 / -194.4. The hand-designed heuristic fed the true alpha costs +50.6 (test) and +4.0 (first 200 fresh) vs TWAP (about 2.6 se apart; heavy-tailed): never better than TWAP, not competitive with CE.
- **Other**: EMA 0.999 (oracle, 3 seeds) did not help (LAST -6.0 paired se about 5.9; VAL-SEL +4.9). Oracle-arm development runs (3 seeds, test worlds): sync 500 (15000 ep) LAST +41.8 / VAL-SEL -22.2; sync 2000 (15000 ep) +19.6 / -48.4; sync 5000 (15000 ep) +574.2 / +378.4 (did not learn in the budget); sync 2000 (30000 ep) -0.4 / -71.4. The lr 3e-4 rows used only 5000 episodes and cannot be compared.
- Tail risk is worse than TWAP for every arm (CVaR95 of cost +84 to +166 $ for single agents; +51 to +188 $ for ensembles).
- The oracle reaches about 75-85% of the CE-on-grid reference; filtered agents learn regime-dependent trading (calm/stressed multipliers 1.39/1.06 for filt_full seed 0 vs 1.39/1.10 for the oracle).

### Spread-Hawkes observations (4 arms; exploratory)

Training history: 3 seeds (0-2) were run first; seeds 3-4 were added AFTER the 3-seed results had been seen (optional stopping, disclosed). The 5-seed numbers are the current ones; the 3-seed numbers are kept for the record. With 5 seeds |t| > 2.776 is needed and no contrast below is confirmed.

`check_independence_hawkes.py` (K = 2 IMM filter, 300 particles, worlds 900001-900003): determinism, independence of world/proxies/beliefs from the agent's actions and cached-env exactness all PASS (max differences 0.0). The Stage 2 filter is overconfident on this arm (alpha cov90 0.54).

**Single agents, 5 seeds, validation-selected** (paired noise-free diff vs TWAP, $; `compare_arms.py --tags _hawkes`):

| Arm | VAL-SEL test (sd over seeds) | VAL-SEL fresh-500 (mean of members) | earlier 3-seed test / fresh | LAST test (sd) |
|---|---|---|---|---|
| oracle | -60.9 (15.9) better than TWAP | -66.9 | -61.1 / -67.2 | +8.8 (41.3) n.s. |
| filt_full | -56.1 (6.1) better than TWAP | -64.5 | -57.3 / -62.9 | +17.4 (16.9) n.s. |
| filt_mean_only | -39.1 (14.8) better than TWAP | -48.0 | -34.6 / -42.9 | +19.8 (81.4) n.s. |
| raw_proxy | -31.5 (33.0) n.s. | -36.0 | -21.2 / -26.1 | +63.9 (72.2) n.s. |

Contrasts by training seed (A - B, VAL-SEL, test, n = 5), all n.s.: filt_full - raw_proxy -24.6 (se 13.5, t -1.82); oracle - filt_full -4.7 (t -0.51); filt_full - filt_mean_only -17.0 (t -1.92; the Gaussian arm has +2.5: opposite sign, hypothesis only that one weak filt_mean_only seed drives it). Final checkpoint: -46.5 (t -1.24), -8.6 (t -0.58), -2.4 (t -0.06). No detectable degradation of filt_full relative to the Gaussian arm (-56.1 vs -61.0). Fresh-world contrasts for the 5 seeds were not seen by the editor of this file (`fresh_scores_hawkes.json` may still hold 3 seeds).

**Ensembles (mean Q of the 5 validation-selected Hawkes agents; `score_ensemble.py --obs-kind spread_hawkes`, same worlds as the Gaussian ensembles: test 900000+ (200), fresh 1,000,000+ (500)).** Noise-free diff vs TWAP, members' mean, ensemble minus members (paired se, t), ensemble CVaR95 vs TWAP:

| Arm | test: ensemble / members / ens-members (se, t) | fresh-500: ensemble / members / ens-members (se, t) | CVaR95 ens (test / fresh) |
|---|---|---|---|
| oracle | -72.4 / -60.9 / -11.5 (2.0, -5.61) | -81.0 / -66.9 / -14.1 (1.2, -12.09) | +118.5 / +78.9 |
| raw_proxy | -70.8 / -31.5 / -39.3 (4.0, -9.83) | -74.0 / -36.0 / -38.0 (2.2, -17.51) | +81.0 / +64.4 |
| filt_full | -77.8 / -56.1 / -21.7 (2.1, -10.15) | -83.4 / -64.5 / -18.9 (1.2, -15.70) | +127.9 / +95.1 |
| filt_mean_only | -61.8 / -39.1 / -22.6 (2.1, -10.97) | -70.2 / -48.0 / -22.2 (1.4, -15.86) | +171.1 / +137.1 |

- The ensemble beats the average member in all 4 arms on both blocks (|t| >= 5.6); raw_proxy gains most (about -38 to -39 $). Mean multiplier by true regime (calm / stressed, test): oracle 1.25 / 0.96, raw_proxy 1.14 / 0.98, filt_full 1.28 / 1.03, filt_mean_only 1.26 / 1.02.
- Tail: ensemble CVaR95 is not lower than the members' mean for oracle (test +118.5 vs +109.2) and filt_mean_only (test +171.1 vs +169.8); elsewhere lower. All are worse than TWAP (+64 to +171).
- Filtered-vs-raw gap of the ENSEMBLES is much smaller here (-7.0 test / -9.4 fresh, from the printed means, no se) than on Gaussian (-31.0 / -34.7), because the raw_proxy ensemble is far better on Hawkes (-70.8 vs -45.9 on test) while filt_full is similar (-77.8 vs -76.9). Unexplained (separately trained agents); do not claim the filter matters less under Hawkes without a Hawkes `ens_contrasts` run.
- First 200 fresh worlds (Stage 4 benchmarks with Hawkes observations): ensemble diffs filt_full -76.8, oracle -74.9, raw_proxy -68.9, filt_mean_only -63.3; paired gap to `ce_filtered_grid` (-90.8): +14.0 (t 2.94), +15.9 (3.85), +21.9 (7.97), +27.5 (5.94): all significantly behind CE (Gaussian ensembles: filt_full +9.6, oracle +4.7, raw_proxy +46.6, filt_mean_only +12.8). CE uses the true state.
- Not run on the Hawkes arm: K = 1, no-regime and bonus arms; seed-level ablation contrasts on ensembles; paired ensemble-vs-CE on the test/500-fresh worlds; a Hawkes CE reference on those worlds (the reference printed by `score_ensemble.py` is for the first 200 fresh worlds only).

### Claims supported / not supported

Supported: the filtered belief is more useful than raw proxies (Gaussian arm; same direction on the Hawkes arm); the filtered agent is statistically indistinguishable from the oracle-input agent; the DDQN beats TWAP on average at the validation-selected checkpoint; averaging five agents recovers most of the single-agent gap to a CE rule.
NOT supported: the uncertainty features help (null on the Gaussian arm); the exploration bonus helps (beta = 1.0 only, inconsistent signs); the regime layer helps (consistent direction, about -11 to -21 $, not significant); the final checkpoint beats TWAP.

## Interface for Stage 4 (Persons 4A/4B/4C)

- **Agents**: `DDQNAgent.load(path).greedy(obs)`; checkpoints in `validation/outputs/stage3c/ckpt/<arm>_base1<tag>_s<seed>_valsel.npz` (the deployable policy) and `..._last.npz` (tags `_abl`, `_abl_b1`, `_hawkes`). The network is rebuilt from layer sizes only, so **score with an env whose `EnvConfig` equals the training config** (obs_mode, use_uncertainty, use_regime, n_regimes, include_kappa, action_multipliers, reward_scale 0.1). The ensemble policy is the argmax of the mean of `q_values(obs)` over `<arm>_base1<tag>_s{0..4}_valsel.npz`.
- **Single-agent logs**: `validation/outputs/stage3c/logs/logs_valsel.pkl` = dict policy name -> list[`interfaces.EpisodeLog`], 41 policies (TWAP + 8 arms x 5 seeds) x 200 worlds (seeds 1,000,000+). Names like `filt_full_abl_s0_valsel`.
- **Ensemble logs**: `validation/outputs/stage3c/logs/logs_ensemble_valsel.pkl` = dict `twap` and `ens_<arm>` (8 arms) -> list[`EpisodeLog`], 200 worlds from 1,000,000. Checked against the saved ensemble scores (`ens_nf_<arm>.npz`, first 200 fresh worlds): max |difference| 3.6e-12 for every arm. Noise-free diff vs TWAP on those worlds: filt_full_bonus -88.9, oracle -85.5, filt_full -80.6, filt_mean_only -77.4, filt_mean_bonus -73.3, filt_single -68.5, filt_no_regime -48.9, raw_proxy -43.6. Names have no seed suffix: the Stage 4 arm-grouping must score each `ens_<arm>` as its own policy. Unpickling needs `..\src` on the path.
- **Hawkes ensemble logs**: `validation/outputs/stage3c/logs/logs_ensemble_valsel_spread_hawkes.pkl` = `twap` + `ens_oracle`, `ens_raw_proxy`, `ens_filt_full`, `ens_filt_mean_only` (200 worlds from 1,000,000; `export_ensemble_logs.py --tags _hawkes`). `check_ens_logs.py --obs-kind spread_hawkes`: logs match the saved scores to 3.6e-12 for all four. Noise-free diff vs TWAP on those worlds: filt_full -76.8, oracle -74.9, raw_proxy -68.9, filt_mean_only -63.3. Per-world arrays: `ens_nf_<arm>_hawkes.npz`. Gaussian and Hawkes logs must be scored with the matching `--obs-kind`.
- `EpisodeLog` fields: policy, seed, world, steps, IS, q0, S0, q_final, shock_step, schema_version. `StepLog.belief` is a partial dict (`t, regime_probs, alpha_mean, alpha_std, kappa_mean, kappa_std`; no `per_regime`, `ess`, `loglik_increment`); `obs` is `None`; the action index is not logged; belief is `None` for the oracle and raw_proxy arms. Proxies for SHAP (4B) therefore have to be regenerated from the observation model.
- **Arm identifiers**: oracle, raw_proxy, filt_full, filt_mean_only, filt_no_regime, filt_single, filt_full_bonus, filt_mean_bonus.
- **Result files**: `validation/outputs/stage3c/results_base1_abl.jsonl`, `..._abl_b1.jsonl`, `..._hawkes.jsonl`, `fresh_scores.json`, `fresh_scores_hawkes.json`, `ablation_report.md`, `hawkes_report.md`, `ens_nf_<arm>.npz`.
- **Hosting in the Stage 4C backend (resolved)**: `PolicyInput` now carries the agent's observation vector (`obs`, Stage 0 Decision 6; see `STAGE0_INTERFACES.md` Section 10.2), so the trained agents and the ensembles are hosted as ordinary policies by the live API. Reported checks (`check_agent_policies.py`, world 900000): for the `oracle`, `raw_proxy`, `filt_full`, `filt_no_regime` and `filt_full_bonus` arms the live observation vector, trade path and IS equal the environment's exactly (max differences 0.0), for single agents and ensembles; the information barrier holds; the live noise-free cost equals the saved ensemble scores (`ens_nf`) for the `oracle` and `filt_full_bonus` ensembles (max gap 3e-12). **Live hosting runs the filter at 500 particles per regime, whereas the agents were trained and scored on beliefs from 2000**, so a live run sees slightly different beliefs from the evaluated ones; say so on any dashboard, or host at 2000. The ensemble needs the same normalised vector that `BeliefMDPEnv._obs()` builds. In the backend the four Hawkes arms (oracle, raw_proxy, filt_full, filt_mean_only) are hosted as single agents and as ensembles on `spread_hawkes` episodes; Hawkes arms without agents and the bonus arms are rejected there and `filt_single` is not hosted (checks E1-E7). Later `check_agent_policies.py` run: ALL CHECKS PASSED, including exact live-vs-environment equality (obs, trade path, IS; max differences 0.0) for the four Hawkes arms, live noise-free cost equal to the saved `ens_nf` for the Hawkes oracle and filt_full ensembles (max gap 9e-13), and a stressed-regime multiplier band guard on ensembles (H1: effective multiplier = clip(raw, 0.75, 1.25) when stressed, 0 violations in 42 stressed steps). The guard's code was not seen by the editor of this file; the exact equality with the saved scores suggests it is inactive in default scoring (inference, confirm).
- **Stage 4A consistency** (checked): TWAP mean IS 4993.1 and the CE/heuristic benchmark values on the first 200 fresh worlds match 4A's handoff; agent numbers differ from the 500-world figures by 4-7 $ because those 200 worlds are slightly harder for every policy.

---

# HANDOFF SYNOPSIS (Stage 3) -- FINAL

**(a) What was built.** A belief-MDP environment around the Stage 1 regime-switching simulator, an observation model and the Stage 2 IMM particle filter, with six observation layouts covering ablations (a), (b), (c); a numpy Double DQN (van Hasselt et al., 2016); a training loop with validation-based best-checkpoint selection; a TWAP reward baseline; an uncertainty-driven exploration bonus tied to the filter's posterior std; a cached-belief environment (about 19x faster, bit-identical to the real env); a multi-seed driver for 8 arms; reporting and export tools; a 5-checkpoint ensemble policy with logs in the Stage 0 `EpisodeLog` format; and the revisit diagnostics (action grid, CE comparison).

**(b) What the results show.** With 5 training seeds per arm and 30000 episodes, the validation-selected single agent beats TWAP in 7 of 8 arms (oracle -67.7 $, filt_full -61.0 $, filt_full_bonus -73.3 $; raw_proxy -9.7 $ is not distinguishable from TWAP); fresh worlds give the same ranking. The only confirmed ablation is filtered belief vs raw proxy (-51.3 $, t -3.37; fresh -58.7 $, t -4.72; Holm-adjusted Welch p 0.066 / 0.032). The filtered agent is statistically indistinguishable from the oracle-input agent (-6.7 $). The uncertainty features (+2.5 $), the exploration bonus and the regime layer (-12.0 $ and -10.9 $; fresh -21.4 $ and -15.0 $) show no significant effect. The final checkpoint never beats TWAP, and tail risk (CVaR95) is worse than TWAP for every arm. Averaging five agents' Q-values beats the average member in every arm and recovers 85-99% of the CE-on-grid gain for the best arms (exploratory). On spread-Hawkes observations (4 arms, 5 seeds; seeds 3-4 added after the first 3-seed results) the filtered agent was not visibly degraded (filt_full -56.1 $ vs -61.0 $ Gaussian, test); the Hawkes ensembles beat their members in all four arms (-11 to -39 $) but trail the CE rule on the grid by +14 to +28 $ (first 200 fresh worlds).

**(c) Interface produced.** See "Interface for Stage 4": checkpoints, single-agent and ensemble `EpisodeLog` files, result files, arm names, and the agent-hosting contract (`PolicyInput.obs`, resolved in Stage 4C).

**(d) Assumptions, simplifications, open questions.**
1. **Primary metric**: validation-selected checkpoint, fixed before any ablation result existed but after the oracle's final-checkpoint metric failed. Report both metrics for every arm.
2. **Small n**: 5 training seeds per arm (5 on the 4-arm Hawkes check); contrast se is 4-12 $, so only effects above about 12-27 $ are detectable. Seeds are limited to 0-6 by the world-seed assertions; adding seeds after seeing these results would be optional stopping and must be disclosed. This was done for the Hawkes arm (seeds 3-4 added after the 3-seed results; both sets are reported).
3. **Multiple contrasts**: 14 contrasts were examined for the main ablation (7 per metric); about 0.7 chance positives expected. One (filter vs raw proxy) is significant and replicates on fresh worlds.
4. **Fresh-world contrasts are not independent replications**: they re-score the same trained agents on new worlds and remove world-sampling noise only.
5. **Exploratory items**: the ensemble (idea arose after seeing results; world-level inference describes the trained agents, not the arms), the grid experiment, and the Hawkes arm with its ensembles (seeds 3-4 added after seeing the 3-seed results).
6. **Exploration bonus** is a heuristic (beliefs do not depend on the agent's trades); only beta = 1.0 was run.
7. **The final checkpoint is not a usable policy**: the greedy policy oscillates between validation checkpoints and EMA did not fix it. Use the validation-selected checkpoint or the ensemble.
8. **Gaussian observations are the main arm**; the filter is overconfident on the Hawkes arm (Stage 2). The CE rule and the filter both use the simulator's true parameters; behaviour under misestimation is untested.
9. **Untuned network and hyperparameters**; no claim to match Macri and Lillo (2025).
10. **Stage 0 request (resolved in Stage 4C)**: `PolicyInput` gained an optional trailing `obs` field so the agents can be hosted behind the standard Policy protocol and the live API. It still needs team ratification and a rerun of `validate_interfaces.py`.
11. **Live hosting uses 500 particles per regime** (training and scoring used 2000), so live beliefs differ slightly from the evaluated ones.
12. **Not done here**: Barger-Lorig benchmark and SHAP (Stage 4), dueling head / n-step returns / soft target updates (not built).