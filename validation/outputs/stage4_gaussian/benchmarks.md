```
worlds: 200 (seeds 1000000..1000199), obs=gaussian.  diffs vs TWAP in $, negative = cheaper.

policy                       meanIS       d_full(se)    d_noisefree(se) CVaR95-TWAP
twap                         4993.1     +0.0 (  0.0)      +0.0 (  0.0)        +0.0
heuristic_belief             5007.3    +14.2 ( 13.3)      +8.3 ( 13.0)      +345.4
heuristic_oracle             5002.9     +9.8 ( 13.0)      +4.0 ( 12.8)      +340.0
bl1_oracle_regime            5333.4   +340.3 ( 28.6)    +336.9 ( 28.8)      +110.3
bl1_oracle_calm              5434.2   +441.0 ( 47.8)    +427.8 ( 46.6)     +2262.7
bl1_filtered_mix             5313.4   +320.3 ( 28.5)    +316.5 ( 28.7)       +92.8
bl1_oracle_regime_grid       5094.1   +100.9 ( 12.0)     +98.3 ( 11.9)       +99.1
ce_oracle                    4894.5    -98.6 (  7.3)     -99.6 (  7.3)       +18.3
ce_oracle_grid               4902.1    -91.0 (  7.1)     -92.2 (  7.1)       +53.1
ce_filtered                  4895.5    -97.7 (  7.3)     -98.7 (  7.3)       +20.0
ce_filtered_grid             4904.3    -88.8 (  7.1)     -90.2 (  7.1)       +56.6

Barger-Lorig truncated at zero (share of steps): bl1_oracle_regime: 0.009, bl1_oracle_calm: 0.072, bl1_filtered_mix: 0.005, bl1_oracle_regime_grid: 0.009

--- Stage 3 agents (validation-selected checkpoints), aggregated over training seeds: [('ens_filt_full', 1), ('ens_filt_full_bonus', 1), ('ens_filt_mean_bonus', 1), ('ens_filt_mean_only', 1), ('ens_filt_no_regime', 1), ('ens_filt_single', 1), ('ens_oracle', 1), ('ens_raw_proxy', 1)]
arm                        n_seeds d_full mean  d_nf mean sd(seeds) world se CVaR95-TWAP
ens_filt_full                    1       -78.8      -80.6       nan      7.3       +99.2
ens_filt_full_bonus              1       -88.2      -88.9       nan      8.9       +82.3
ens_filt_mean_bonus              1       -72.4      -73.3       nan      9.3       +99.5
ens_filt_mean_only               1       -76.8      -77.4       nan     10.2       +84.6
ens_filt_no_regime               1       -48.5      -48.9       nan     10.3       +55.5
ens_filt_single                  1       -67.4      -68.5       nan     11.5       +95.2
ens_oracle                       1       -84.2      -85.5       nan     10.5       +69.3
ens_raw_proxy                    1       -41.9      -43.6       nan     11.1      +195.6

--- head-to-head, noise-free cost, paired over worlds (arm seed-averaged minus benchmark; negative = agent cheaper)
arm                                ce_filtered_grid           ce_oracle_grid                ce_oracle         heuristic_belief
ens_filt_full                        +9.6 (t +4.41)          +11.6 (t +5.03)          +19.1 (t +6.52)          -88.9 (t -7.83)
ens_filt_full_bonus                  +1.2 (t +0.46)           +3.2 (t +1.23)          +10.7 (t +4.32)          -97.2 (t -7.56)
ens_filt_mean_bonus                 +16.9 (t +4.86)          +18.9 (t +5.26)          +26.4 (t +7.27)          -81.5 (t -6.81)
ens_filt_mean_only                  +12.8 (t +3.04)          +14.8 (t +3.47)          +22.2 (t +5.71)          -85.7 (t -6.53)
ens_filt_no_regime                  +41.3 (t +9.09)          +43.3 (t +9.46)         +50.8 (t +11.02)          -57.1 (t -3.87)
ens_filt_single                     +21.7 (t +4.13)          +23.7 (t +4.51)          +31.2 (t +6.33)          -76.8 (t -5.39)
ens_oracle                           +4.7 (t +1.15)           +6.7 (t +1.64)          +14.1 (t +3.93)          -93.8 (t -6.76)
ens_raw_proxy                       +46.6 (t +9.06)          +48.6 (t +9.34)          +56.0 (t +9.90)          -51.9 (t -3.88)

CAVEATS: world-sampling inference only for benchmarks; DDQN agents carry extra training-seed variance (sd(seeds)).
Barger-Lorig variants are misspecified for a regime-switching world and use simulator (uncalibrated) parameters.
ce_* = certainty-equivalent plan with exact mean forecast (not BL). ce_filtered assumes E[alpha;regime=j] ~ p_j*alpha_hat. Reference, not optimum.
Compare agents with benchmarks on the SAME metric: the noise-free column is the Stage 3 metric.
```
