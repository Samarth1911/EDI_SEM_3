```
episodes=30000  eval_episodes=200  tags=['_abl', '_abl_b1']  (diff vs TWAP in $, negative = better)

=== per arm vs TWAP -- PRIMARY: validation-selected checkpoint, scored on test worlds ===
arm                             n     mean sd(seeds)     se 95% CI half world se  verdict
oracle                          5    -67.7       7.3    3.3         9.0     10.3  BETTER than TWAP
raw_proxy                       5     -9.7      26.2   11.7        32.6     12.0  n.s.
filt_full                       5    -61.0       8.6    3.8        10.6      8.4  BETTER than TWAP
filt_mean_only                  5    -63.4      12.1    5.4        15.0     10.6  BETTER than TWAP
filt_no_regime                  5    -49.0      20.1    9.0        25.0      9.7  BETTER than TWAP
filt_single                     5    -50.1      10.2    4.6        12.7     10.9  BETTER than TWAP
filt_full_bonus                 5    -73.3       7.5    3.4         9.3      9.3  BETTER than TWAP
filt_mean_bonus                 5    -56.2      17.7    7.9        21.9     10.7  BETTER than TWAP

=== per arm vs TWAP -- SECONDARY: final checkpoint ===
arm                             n     mean sd(seeds)     se 95% CI half world se  verdict
oracle                          5     +4.5      16.7    7.5        20.8     14.1  n.s.
raw_proxy                       5    +72.4      35.9   16.1        44.6     15.0  WORSE
filt_full                       5    +70.4     182.9   81.8       227.0     14.3  n.s.
filt_mean_only                  5    -10.7      28.2   12.6        35.0     12.9  n.s.
filt_no_regime                  5    +58.9      22.6   10.1        28.1     16.6  WORSE
filt_single                     5    +12.0      18.6    8.3        23.0     13.1  n.s.
filt_full_bonus                 5     +0.6      53.8   24.1        66.8     17.9  n.s.
filt_mean_bonus                 5   +137.0     309.2  138.3       383.9     59.1  n.s.

world se = paired se over the shared test worlds (common to all seeds; not reduced by more seeds).

=== paired contrasts A - B (negative = A cheaper), by training seed -- valsel ===
contrast                                                            n     mean     se      t 95% CI half  verdict
(a) filter vs raw proxy         : filt_full - raw_proxy             5    -51.3   15.2  -3.37        42.2  A better
(a) value of perfect info       : oracle - filt_full                5     -6.7    6.3  -1.06        17.5  n.s.
(b) uncertainty features        : filt_full - filt_mean_only        5     +2.5    4.4  +0.56        12.2  n.s.
(b) bonus, mean-only            : filt_mean_bonus - filt_mean_only  5     +7.2    7.0  +1.03        19.5  n.s.
(b) bonus, with std features    : filt_full_bonus - filt_full       5    -12.3    5.7  -2.16        15.8  n.s.
(c) regime probs masked         : filt_full - filt_no_regime        5    -12.0    9.9  -1.22        27.3  n.s.
(c) K=1 filter instead of IMM   : filt_full - filt_single           5    -10.9    4.7  -2.31        13.0  n.s.

=== paired contrasts A - B (negative = A cheaper), by training seed -- last ===
contrast                                                            n     mean     se      t 95% CI half  verdict
(a) filter vs raw proxy         : filt_full - raw_proxy             5     -2.1   91.6  -0.02       254.2  n.s.
(a) value of perfect info       : oracle - filt_full                5    -65.9   77.9  -0.85       216.3  n.s.
(b) uncertainty features        : filt_full - filt_mean_only        5    +81.1   81.4  +1.00       225.8  n.s.
(b) bonus, mean-only            : filt_mean_bonus - filt_mean_only  5   +147.7  139.6  +1.06       387.6  n.s.
(b) bonus, with std features    : filt_full_bonus - filt_full       5    -69.8   88.7  -0.79       246.4  n.s.
(c) regime probs masked         : filt_full - filt_no_regime        5    +11.5   81.8  +0.14       227.0  n.s.
(c) K=1 filter instead of IMM   : filt_full - filt_single           5    +58.4   79.2  +0.74       219.9  n.s.

NOTES: 7 contrasts x 2 metrics -> expect ~1 false 'better' at the 95% level by chance; treat single hits cautiously.
Test worlds are shared by all arms/seeds (paired comparisons are fair; absolute claims carry the 'world se').
Configs were chosen on these test worlds earlier: confirm headline numbers with score_fresh.py.
```
