```
episodes=30000  eval_episodes=200  tags=['_abl']  (diff vs TWAP in $, negative = better)

=== per arm vs TWAP -- PRIMARY: validation-selected checkpoint, scored on test worlds ===
arm                             n     mean sd(seeds)     se 95% CI half world se  verdict
oracle                          5    -67.7       7.3    3.3         9.0     10.3  BETTER than TWAP
raw_proxy                       5     -9.7      26.2   11.7        32.6     12.0  n.s.
filt_full                       1    -59.7       nan    nan         nan      6.5  n<3

=== per arm vs TWAP -- SECONDARY: final checkpoint ===
arm                             n     mean sd(seeds)     se 95% CI half world se  verdict
oracle                          5     +4.5      16.7    7.5        20.8     14.1  n.s.
raw_proxy                       5    +72.4      35.9   16.1        44.6     15.0  WORSE
filt_full                       1     -9.9       nan    nan         nan     11.9  n<3

world se = paired se over the shared test worlds (common to all seeds; not reduced by more seeds).

=== paired contrasts A - B (negative = A cheaper), by training seed -- valsel ===
contrast                                                            n     mean     se      t 95% CI half  verdict
(a) filter vs raw proxy         : filt_full - raw_proxy             1  n<3: no test
(a) value of perfect info       : oracle - filt_full                1  n<3: no test
(b) uncertainty features        : filt_full - filt_mean_only        (missing arm)
(b) bonus, mean-only            : filt_mean_bonus - filt_mean_only  (missing arm)
(b) bonus, with std features    : filt_full_bonus - filt_full       (missing arm)
(c) regime probs masked         : filt_full - filt_no_regime        (missing arm)
(c) K=1 filter instead of IMM   : filt_full - filt_single           (missing arm)

=== paired contrasts A - B (negative = A cheaper), by training seed -- last ===
contrast                                                            n     mean     se      t 95% CI half  verdict
(a) filter vs raw proxy         : filt_full - raw_proxy             1  n<3: no test
(a) value of perfect info       : oracle - filt_full                1  n<3: no test
(b) uncertainty features        : filt_full - filt_mean_only        (missing arm)
(b) bonus, mean-only            : filt_mean_bonus - filt_mean_only  (missing arm)
(b) bonus, with std features    : filt_full_bonus - filt_full       (missing arm)
(c) regime probs masked         : filt_full - filt_no_regime        (missing arm)
(c) K=1 filter instead of IMM   : filt_full - filt_single           (missing arm)

NOTES: 7 contrasts x 2 metrics -> expect ~1 false 'better' at the 95% level by chance; treat single hits cautiously.
Test worlds are shared by all arms/seeds (paired comparisons are fair; absolute claims carry the 'world se').
Configs were chosen on these test worlds earlier: confirm headline numbers with score_fresh.py.
```
