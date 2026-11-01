# STAGE 3 REVISIT -- HANDOFF SYNOPSIS (Person 3) -- FINAL

_Why: Stage 4A reported that single trained agents trail a simple certainty-equivalent (CE) rule (4A: gap to `ce_filtered_grid` of +12.0 $ for the best arm up to +86.4 $ for raw_proxy; checked against our fresh-world numbers, see (b)6). This revisit asks why and what closes the gap. Test worlds = seeds 900000+ (200), fresh worlds = 1,000,000+ (500). Noise-free paired difference vs TWAP in $, negative = better. All 8 arms are covered._

## (a) What was built (run by the team; outputs below)
`diag_agent_vs_ce.py` (agent vs CE multiplier per step from logs), `diag_grid.py` (CE snapped to candidate grids on design worlds 1,500,000+), `check_grid_env.py`, `--grid` flag in `run_stage3c_multiseed.py` (EnvConfig only; requires `--tag`), `compare_grids.py`, `diag_ce_ref.py` (CE reference, any grid and world range), `score_ensemble.py` (mean-Q ensemble of an arm's 5 val-sel checkpoints; saves `ens_nf_<arm>.npz`; `--obs-kind spread_hawkes` for the four Hawkes arms saves `ens_nf_<arm>_hawkes.npz`), `ens_vs_ce.py` (paired ensemble vs CE; run for oracle and filt_full_bonus), `ens_contrasts.py` (7 pre-listed ablation contrasts at world and seed level, Holm-adjusted), `export_ensemble_logs.py` and `check_ens_logs.py` (ensemble `EpisodeLog` export and its verification).

## (b) What the results show
1. **Agents track CE but are noisy and over-reactive** (fresh worlds, per-step multiple of remaining-TWAP; from `diag_agent_vs_ce.py`, output not re-checked here): slope of log agent on log CE multiple +1.1 to +1.45, corr 0.64-0.83, mean |dlog| 0.11-0.22. Regime-level means match (filt_full calm/stress 1.10/0.95 vs CE 1.09/0.98); agent timing is flatter than CE; raw_proxy trades about 1.2x TWAP and sits at the 2.0 grid end on 13% of steps.
2. **Grid experiment = NULL.** On design worlds CE snapped to the default grid loses about 10 $ (-107.2 continuous vs -97.2; from `diag_grid.py`, output not re-checked here); finer log grids all recover 98%. Oracle arm, 5 seeds, grid (0.579,0.694,0.833,1.0,1.2,1.44,1.728): VAL-SEL -63.3 (sd 11.1) vs -67.7 (sd 7.3) on the default grid, +4.3, Welch p 0.49; LAST -20.8 (sd 27.9) vs +4.5 (sd 16.7), p 0.13 (paired by seed t about -2.7, below 2.776); VAL-SEL CVaR95 +222 vs +154 (p 0.094). CE on the new grid is -98.5 (se 8.0) vs -90.8 on the default grid (test worlds), and the agent recovers less of it (64% vs 75%). Default grid retained. Collapse fraction was 0.0 under both grids (an earlier remark that the narrower grid prevents collapse is withdrawn).
3. **Ensembling helps in every arm** (mean Q over the 5 val-sel agents, no tuning). Ensemble vs TWAP, mean of members vs TWAP, and ensemble minus mean of members (paired se over worlds):

| Arm | test: ensemble / members / ens-members (se) | fresh: ensemble / members / ens-members (se) | ensemble CVaR95 vs TWAP (test / fresh) |
|---|---|---|---|
| oracle | -82.2 / -67.7 / -14.5 (1.5) | -89.2 / -76.9 / -12.4 (0.9) | +117 / +66 |
| raw_proxy | -45.9 / -9.7 / -36.1 (3.0) | -50.6 / -11.3 / -39.3 (2.0) | +188 / +179 |
| filt_full | -76.9 / -61.0 / -15.9 (1.6) | -85.3 / -70.0 / -15.3 (1.0) | +113 / +90 |
| filt_mean_only | -73.2 / -63.4 / -9.7 (1.7) | -82.4 / -73.0 / -9.4 (1.0) | +134 / +84 |
| filt_no_regime | -57.9 / -49.0 / -8.9 (2.9) | -54.6 / -48.5 / -6.1 (1.7) | +54 / +51 |
| filt_single | -69.3 / -50.1 / -19.2 (2.8) | -73.0 / -54.9 / -18.0 (1.7) | +123 / +87 |
| filt_full_bonus | -87.9 / -73.3 / -14.6 (1.5) | -93.7 / -82.0 / -11.7 (0.9) | +104 / +75 |
| filt_mean_bonus | -67.7 / -56.2 / -11.5 (2.3) | -79.0 / -68.1 / -10.9 (1.3) | +150 / +94 |

   The ensemble beats the AVERAGE member in all 16 cases (|t| >= 3.1). It does not always beat the BEST member (it does not for filt_mean_only, filt_no_regime, filt_mean_bonus); best-of-five picked after the fact is not a fair comparator.
4. **Ensemble vs CE** (CE uses the TRUE state; paired over the same worlds; run for two arms). CE on the grid: -90.8 (se 7.5) on test, -95.1 (se 4.3) on the 500 fresh worlds; continuous CE -101.5 / -102.1.
   - oracle ensemble - CE grid: +8.6 (se 4.2, t 2.05) test, +5.8 (se 2.5, t 2.38) fresh: slightly worse than CE.
   - filt_full_bonus ensemble - CE grid: +2.9 (se 2.7, t 1.09) test, +1.4 (se 1.6, t 0.85) fresh: indistinguishable from CE. This is the best single arm and was chosen for this comparison after seeing results.
   - single agents (mean member) - CE grid: oracle +23.2 / +18.2, filt_full_bonus +17.5 / +13.1.
   - ensemble - continuous CE: +8.4 to +19.3 (all t > 5); about 10 $ of that is the cost of the 7-point grid.
   - Ensemble gain as a share of the CE-on-grid gain (unpaired ratio; test / fresh): oracle 91% / 94%, filt_full_bonus 97% / 99%, filt_full 85% / 90%, filt_mean_only 81% / 87%, filt_mean_bonus 75% / 83%, filt_single 76% / 77%, filt_no_regime 64% / 57%, raw_proxy 51% / 53%.
5. **Ablation contrasts on the ensembles** (`ens_contrasts.py`, X - Y, negative = X cheaper). World-level conditions on ONE ensemble per arm (world sampling only); seed-level is Welch over the 5 members (includes training variability), Holm over 7:

| Contrast | world-level test / fresh | seed-level test (Holm p) / fresh (Holm p) |
|---|---|---|
| filtered vs raw proxy (filt_full - raw_proxy) | -31.0 / -34.7 | -51.3 (0.066) / -58.7 (**0.032**) |
| uncertainty features (filt_full - filt_mean_only) | -3.7 / -2.9 (n.s.) | +2.5 (0.95) / +3.0 (1.0) |
| bonus with std features (filt_full_bonus - filt_full) | -11.0 / -8.5 | -12.3 (0.26) / -12.0 (0.22) |
| bonus mean-only (filt_mean_bonus - filt_mean_only) | +5.4 / +3.4 | +7.2 (0.95) / +4.9 (1.0) |
| regime layer (filt_full - filt_no_regime) | -19.0 / -30.6 | -12.0 (0.89) / -21.4 (0.53) |
| IMM vs K=1 filter (filt_full - filt_single) | -7.6 / -12.3 | -10.9 (0.54) / -15.0 (0.22) |
| oracle vs filtered input (oracle - filt_full) | -5.3 / -4.0 (n.s.) | -6.7 (0.89) / -6.9 (0.77) |

   Only filtered-vs-raw-proxy survives the seed-level correction (fresh), borderline on test. World-level significance applies to these particular trained agents, not to the arms. The bonus helps with std features (about -8 to -12 $) but not on mean-only agents (+3 to +7 $); no consistent bonus effect. Oracle vs filtered input is confounded by the bonus (the oracle arm has no posterior std); the fair comparison is oracle vs filt_full (oracle better by 4-5 $, n.s.).
6. **CE reference verified and consistent with Stage 4A.** `diag_ce_ref.py` on the first 200 fresh worlds reproduces 4A to the decimal: TWAP mean IS 4993.1; `rhce_oracle` (continuous CE) -99.6 (se 7.3); `rhce_grid` -92.2 (se 7.1); `heur_oracle` +4.0 (se 12.8); non-causal ceiling -194.4 (se 11.0). On the 200 test worlds the same script gives -101.5, -90.8, +50.6 (se 12.4) and -193.4. CE is stable across the two world sets; the oracle-input heuristic is not (+50.6 vs +4.0, about 2.6 se apart, heavy-tailed outcomes): it never beats TWAP and is not competitive with CE, but its mean should not be quoted as a single number. Stage 4A's single-agent numbers (mean over seeds on the first 200 fresh worlds) differ from the 500-world figures here by 4-7 $ because those 200 worlds are slightly harder for every policy, CE included.
7. **Ensemble logs verified.** `export_ensemble_logs.py` wrote `logs_ensemble_valsel.pkl` (TWAP + `ens_<arm>` for 8 arms, 200 worlds from 1,000,000). `check_ens_logs.py`: costs recomputed from the `EpisodeLog`s match the saved ensemble scores (`ens_nf_<arm>.npz`, first 200 fresh worlds) to 3.6e-12 for every arm. Noise-free diff vs TWAP on those worlds: filt_full_bonus -88.9, oracle -85.5, filt_full -80.6, filt_mean_only -77.4, filt_mean_bonus -73.3, filt_single -68.5, filt_no_regime -48.9, raw_proxy -43.6. Against 4A's references on the same worlds (unpaired consistency check): `ce_filtered_grid` -90.2, `ce_oracle_grid` -92.2.

8. **Hawkes ensembles** (`score_ensemble.py --obs-kind spread_hawkes`, 5 seeds per arm, four arms; same test/fresh worlds). Ensemble vs TWAP, members' mean, ensemble minus members (paired se), ensemble CVaR95 vs TWAP:

| Arm | test: ensemble / members / ens-members (se) | fresh-500: ensemble / members / ens-members (se) | CVaR95 ens (test / fresh) |
|---|---|---|---|
| oracle | -72.4 / -60.9 / -11.5 (2.0) | -81.0 / -66.9 / -14.1 (1.2) | +118.5 / +78.9 |
| raw_proxy | -70.8 / -31.5 / -39.3 (4.0) | -74.0 / -36.0 / -38.0 (2.2) | +81.0 / +64.4 |
| filt_full | -77.8 / -56.1 / -21.7 (2.1) | -83.4 / -64.5 / -18.9 (1.2) | +127.9 / +95.1 |
| filt_mean_only | -61.8 / -39.1 / -22.6 (2.1) | -70.2 / -48.0 / -22.2 (1.4) | +171.1 / +137.1 |

   The ensemble beats the average member in all 8 cases (|t| >= 5.6). Ensemble tail is not lower than the members' mean for oracle and filt_mean_only on test; every ensemble is still worse than TWAP (+64 to +171). The filtered-vs-raw gap of the ensembles (-7.0 test / -9.4 fresh, computed from the printed means, no se) is much smaller than on Gaussian (-31.0 / -34.7) because the raw_proxy ensemble is far better on Hawkes (-70.8 vs -45.9, test); unexplained, no Hawkes contrast run. First 200 fresh worlds (logs verified to 3.6e-12): noise-free diff filt_full -76.8, oracle -74.9, raw_proxy -68.9, filt_mean_only -63.3, against Hawkes `ce_filtered_grid` -90.8: paired gaps +14.0 / +15.9 / +21.9 / +27.5 (t 2.9 to 8.0), so all four trail CE (details in `STAGE4_HANDOFF.md`). Seeds 3-4 of the Hawkes agents were added after the 3-seed results were seen.

## (c) Interface
Ensemble policy = argmax of the mean of `DDQNAgent.q_values(obs)` over `<arm>_base1<tag>_s{0..4}_valsel.npz` (tag `_abl`, `_abl_b1` for bonus arms, `_hawkes` for the Hawkes arms, which need `--obs-kind spread_hawkes`); requires the same `EnvConfig` as training. It needs the agent observation vector, so hosting it behind the standard Policy protocol still needs the open `PolicyInput` decision (add the latest observation). Per-world arrays in `outputs/stage3c/ens_nf_<arm>.npz` (keys `{test,fresh}_{seeds,twap,ens,mem}`), all 8 arms present; Hawkes: `ens_nf_<arm>_hawkes.npz` for oracle, raw_proxy, filt_full, filt_mean_only, and `logs_ensemble_valsel_spread_hawkes.pkl` (`ens_<arm>` for those four). `EpisodeLog`s of the ensembles in `outputs/stage3c/logs/logs_ensemble_valsel.pkl`; names `ens_<arm>` have no seed suffix and need their own entry in the Stage 4 arm grouping. Unpickling needs `..\src` on the path.

## (d) Assumptions, limits, open items
- The ensemble idea arose after results on the test and fresh worlds had been seen; nothing was tuned (all 5 members, mean Q), but it is NOT pre-registered. Ensemble-level contrasts are exploratory.
- World-level se measures world sampling only; one ensemble of 5 per arm says nothing about retraining variability. Inference cost is 5x.
- The evidence supports "much of the single-agent gap to CE is training noise", not "variance, not bias, is the limiter": ensembles still trail continuous CE by 8-19 $, of which about 10 $ is the 7-point grid.
- The paired ensemble-vs-CE comparison on the test/500-fresh worlds exists only for oracle and filt_full_bonus (Gaussian). Paired comparisons on the first 200 fresh worlds now exist for all 8 Gaussian ensembles and the 4 Hawkes ensembles (`STAGE4_HANDOFF.md`).
- The CE rule and the filter both use the simulator's true parameters; behaviour under misestimation is untested. The Hawkes arm (4 arms, 5 seeds; seeds 3-4 added after the 3-seed results were seen) now has ensembles (item 8). Not run on Hawkes: K = 1, no-regime and bonus arms, ensemble ablation contrasts, a test/500-fresh CE reference. See `STAGE3_HANDOFF.md`.
- `diag_grid.py` and `diag_agent_vs_ce.py` outputs (design-world grid numbers, agent-vs-CE slopes) are quoted from an earlier session and were not re-checked here.
- The original `diag_headroom.py` was overwritten by a different tool; `diag_ce_ref.py` reproduces its numbers.
- Dueling head, n-step returns and soft target updates were not built.
- Ensemble tail risk is lower than that of the members but still worse than TWAP for every arm (+51 to +188 $).