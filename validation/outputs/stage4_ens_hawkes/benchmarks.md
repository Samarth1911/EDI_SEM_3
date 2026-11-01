```
worlds: 200 (seeds 1000000..1000199), obs=spread_hawkes.  diffs vs TWAP in $, negative = cheaper.

policy                       meanIS       d_full(se)    d_noisefree(se) CVaR95-TWAP
twap                         4993.1     +0.0 (  0.0)      +0.0 (  0.0)        +0.0
heuristic_belief             5010.8    +17.7 ( 13.6)     +11.6 ( 13.3)      +360.8
heuristic_oracle             5002.9     +9.8 ( 13.0)      +4.0 ( 12.8)      +340.0
bl1_oracle_regime            5333.4   +340.3 ( 28.6)    +336.9 ( 28.8)      +110.3
bl1_oracle_calm              5434.2   +441.0 ( 47.8)    +427.8 ( 46.6)     +2262.7
bl1_filtered_mix             5298.4   +305.3 ( 26.3)    +300.6 ( 26.4)      +136.9
bl1_oracle_regime_grid       5094.1   +100.9 ( 12.0)     +98.3 ( 11.9)       +99.1
ce_oracle                    4894.5    -98.6 (  7.3)     -99.6 (  7.3)       +18.3
ce_oracle_grid               4902.1    -91.0 (  7.1)     -92.2 (  7.1)       +53.1
ce_filtered                  4895.9    -97.2 (  7.4)     -98.5 (  7.4)       +29.9
ce_filtered_grid             4903.9    -89.2 (  7.3)     -90.8 (  7.2)       +68.1

Barger-Lorig truncated at zero (share of steps): bl1_oracle_regime: 0.009, bl1_oracle_calm: 0.072, bl1_filtered_mix: 0.005, bl1_oracle_regime_grid: 0.009

--- Stage 3 agents (validation-selected checkpoints), aggregated over training seeds: [('ens_filt_full', 1), ('ens_filt_mean_only', 1), ('ens_oracle', 1), ('ens_raw_proxy', 1)]
arm                        n_seeds d_full mean  d_nf mean sd(seeds) world se CVaR95-TWAP
ens_filt_full                    1       -75.5      -76.8       nan     11.3      +106.0
ens_filt_mean_only               1       -60.9      -63.3       nan     10.9      +148.4
ens_oracle                       1       -73.9      -74.9       nan     10.6       +86.6
ens_raw_proxy                    1       -67.9      -68.9       nan      6.8       +66.5

--- head-to-head, noise-free cost, paired over worlds (arm seed-averaged minus benchmark; negative = agent cheaper)
arm                                ce_filtered_grid           ce_oracle_grid                ce_oracle         heuristic_belief
ens_filt_full                       +14.0 (t +2.94)          +15.4 (t +3.16)          +22.8 (t +5.02)          -88.4 (t -5.89)
ens_filt_mean_only                  +27.5 (t +5.94)          +28.9 (t +5.91)          +36.3 (t +7.58)          -75.0 (t -5.74)
ens_oracle                          +15.9 (t +3.85)          +17.3 (t +3.98)          +24.8 (t +6.13)          -86.5 (t -6.16)
ens_raw_proxy                       +21.9 (t +7.97)          +23.3 (t +8.40)          +30.7 (t +9.72)          -80.6 (t -6.67)

CAVEATS: world-sampling inference only for benchmarks; DDQN agents carry extra training-seed variance (sd(seeds)).
Barger-Lorig variants are misspecified for a regime-switching world and use simulator (uncalibrated) parameters.
ce_* = certainty-equivalent plan with exact mean forecast (not BL). ce_filtered assumes E[alpha;regime=j] ~ p_j*alpha_hat. Reference, not optimum.
Compare agents with benchmarks on the SAME metric: the noise-free column is the Stage 3 metric.
```
