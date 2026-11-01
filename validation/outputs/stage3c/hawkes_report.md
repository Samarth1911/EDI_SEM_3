```
episodes=30000  eval_episodes=200  tags=['_hawkes']  (diff vs TWAP in $, negative = better)

=== per arm vs TWAP -- PRIMARY: validation-selected checkpoint, scored on test worlds ===
arm                             n     mean sd(seeds)     se 95% CI half world se  verdict
oracle                          5    -60.9      15.9    7.1        19.8      9.8  BETTER than TWAP
raw_proxy                       5    -31.5      33.0   14.8        41.0      9.9  n.s.
filt_full                       5    -56.1       6.1    2.7         7.5     12.9  BETTER than TWAP
filt_mean_only                  5    -39.1      14.8    6.6        18.3     11.5  BETTER than TWAP

=== per arm vs TWAP -- SECONDARY: final checkpoint ===
arm                             n     mean sd(seeds)     se 95% CI half world se  verdict
oracle                          5     +8.8      41.3   18.5        51.2     14.6  n.s.
raw_proxy                       5    +63.9      72.2   32.3        89.6     14.4  n.s.
filt_full                       5    +17.4      16.9    7.6        21.0     15.8  n.s.
filt_mean_only                  5    +19.8      81.4   36.4       101.0     17.0  n.s.

world se = paired se over the shared test worlds (common to all seeds; not reduced by more seeds).

=== paired contrasts A - B (negative = A cheaper), by training seed -- valsel ===
contrast                                                            n     mean     se      t 95% CI half  verdict
(a) filter vs raw proxy         : filt_full - raw_proxy             5    -24.6   13.5  -1.82        37.5  n.s.
(a) value of perfect info       : oracle - filt_full                5     -4.7    9.3  -0.51        25.8  n.s.
(b) uncertainty features        : filt_full - filt_mean_only        5    -17.0    8.8  -1.92        24.5  n.s.
(b) bonus, mean-only            : filt_mean_bonus - filt_mean_only  (missing arm)
(b) bonus, with std features    : filt_full_bonus - filt_full       (missing arm)
(c) regime probs masked         : filt_full - filt_no_regime        (missing arm)
(c) K=1 filter instead of IMM   : filt_full - filt_single           (missing arm)

=== paired contrasts A - B (negative = A cheaper), by training seed -- last ===
contrast                                                            n     mean     se      t 95% CI half  verdict
(a) filter vs raw proxy         : filt_full - raw_proxy             5    -46.5   37.6  -1.24       104.3  n.s.
(a) value of perfect info       : oracle - filt_full                5     -8.6   14.7  -0.58        40.7  n.s.
(b) uncertainty features        : filt_full - filt_mean_only        5     -2.4   42.7  -0.06       118.5  n.s.
(b) bonus, mean-only            : filt_mean_bonus - filt_mean_only  (missing arm)
(b) bonus, with std features    : filt_full_bonus - filt_full       (missing arm)
(c) regime probs masked         : filt_full - filt_no_regime        (missing arm)
(c) K=1 filter instead of IMM   : filt_full - filt_single           (missing arm)

NOTES: 7 contrasts x 2 metrics -> expect ~1 false 'better' at the 95% level by chance; treat single hits cautiously.
Test worlds are shared by all arms/seeds (paired comparisons are fair; absolute claims carry the 'world se').
Configs were chosen on these test worlds earlier: confirm headline numbers with score_fresh.py.
```
