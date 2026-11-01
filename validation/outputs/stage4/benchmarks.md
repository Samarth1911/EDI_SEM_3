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

--- Stage 3 agents (validation-selected checkpoints), aggregated over training seeds: [('filt_full_abl', 5), ('filt_full_bonus_abl_b1', 5), ('filt_mean_bonus_abl_b1', 5), ('filt_mean_only_abl', 5), ('filt_no_regime_abl', 5), ('filt_single_abl', 5), ('oracle_abl', 5), ('raw_proxy_abl', 5)]
arm                        n_seeds d_full mean  d_nf mean sd(seeds) world se CVaR95-TWAP
filt_full_abl                    5       -64.1      -65.5       7.9      7.6       +99.8
filt_full_bonus_abl_b1           5       -76.6      -78.2       4.6      8.9      +111.4
filt_mean_bonus_abl_b1           5       -60.2      -61.7      13.1      9.7      +194.2
filt_mean_only_abl               5       -66.7      -68.1      11.6     10.0      +127.1
filt_no_regime_abl               5       -43.1      -43.9      26.7      9.0       +88.1
filt_single_abl                  5       -49.8      -51.2      11.3      9.9      +133.7
oracle_abl                       5       -71.7      -73.1      10.5     10.0      +109.5
raw_proxy_abl                    5        -2.4       -3.8      27.3     11.0      +138.4

--- head-to-head, noise-free cost, paired over worlds (arm seed-averaged minus benchmark; negative = agent cheaper)
arm                                ce_filtered_grid           ce_oracle_grid                ce_oracle         heuristic_belief
filt_full_abl                      +24.7 (t +12.27)         +26.7 (t +12.98)         +34.2 (t +15.16)          -73.7 (t -6.22)
filt_full_bonus_abl_b1              +12.0 (t +4.52)          +14.0 (t +5.24)          +21.5 (t +7.79)          -86.4 (t -7.18)
filt_mean_bonus_abl_b1              +28.5 (t +7.51)          +30.5 (t +7.85)          +38.0 (t +9.39)          -69.9 (t -6.03)
filt_mean_only_abl                  +22.1 (t +5.91)          +24.1 (t +6.39)          +31.5 (t +8.81)          -76.4 (t -6.07)
filt_no_regime_abl                 +46.2 (t +14.33)         +48.2 (t +14.79)         +55.7 (t +16.22)          -52.2 (t -3.86)
filt_single_abl                    +39.0 (t +11.06)         +41.0 (t +11.57)         +48.4 (t +13.92)          -59.5 (t -4.61)
oracle_abl                          +17.0 (t +4.87)          +19.0 (t +5.41)          +26.5 (t +8.18)          -81.4 (t -6.28)
raw_proxy_abl                      +86.4 (t +15.93)         +88.4 (t +16.20)         +95.9 (t +16.72)          -12.0 (t -0.84)

CAVEATS: world-sampling inference only for benchmarks; DDQN agents carry extra training-seed variance (sd(seeds)).
Barger-Lorig variants are misspecified for a regime-switching world and use simulator (uncalibrated) parameters.
ce_* = certainty-equivalent plan with exact mean forecast (not BL). ce_filtered assumes E[alpha;regime=j] ~ p_j*alpha_hat. Reference, not optimum.
Compare agents with benchmarks on the SAME metric: the noise-free column is the Stage 3 metric.
```
