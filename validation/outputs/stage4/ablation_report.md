```
handoff check: all 74 ensemble values match STAGE3_REVISIT_HANDOFF.md (b)3 and (b)5

STAGE 4 ABLATION REPORT -- noise-free cost difference vs TWAP in $, negative = cheaper; contrast X - Y < 0 means X cheaper.
worlds: test 200, fresh 500; single agents: validation-selected checkpoints on the test worlds (tags ['_abl', '_abl_b1'], 30000 episodes).

=== arms ===
arm                 single agents (test): mean sd n   ENSEMBLE test (se)   ENSEMBLE fresh (se)   members mean test/fresh
oracle                             -67.7    7.3   5        -82.2 (10.8)        -89.2 ( 6.4)       -67.7 /   -76.9
raw_proxy                           -9.7   26.2   5        -45.9 (11.4)        -50.6 ( 7.0)        -9.7 /   -11.3
filt_full                          -61.0    8.6   5        -76.9 ( 7.6)        -85.3 ( 4.5)       -61.0 /   -70.0
filt_mean_only                     -63.4   12.1   5        -73.2 (10.6)        -82.4 ( 6.2)       -63.4 /   -73.0
filt_no_regime                     -49.0   20.1   5        -57.9 (10.5)        -54.6 ( 6.5)       -49.0 /   -48.5
filt_single                        -50.1   10.2   5        -69.3 (11.7)        -73.0 ( 7.1)       -50.1 /   -54.9
filt_full_bonus                    -73.3    7.5   5        -87.9 ( 9.3)        -93.7 ( 5.5)       -73.3 /   -82.0
filt_mean_bonus                    -56.2   17.7   5        -67.7 ( 9.8)        -79.0 ( 5.7)       -56.2 /   -68.1

=== (a) Does the information help? (filter vs raw proxy; perfect info vs filter) ===

  filt_full - raw_proxy  [filtered belief vs raw proxy]   VERDICT: BORDERLINE
    single agents, paired by training seed (test worlds, n=5):   -51.3 (se 15.2, t -3.37, Holm p 0.196)
    ensemble test   WORLD-level   -31.0 (se 5.7, Holm p 0.000)   SEED-level Welch   -51.3 (t -4.15, Holm p 0.066)
    ensemble fresh  WORLD-level   -34.7 (se 3.4, Holm p 0.000)   SEED-level Welch   -58.7 (t -4.89, Holm p 0.032)

  oracle - filt_full  [oracle input vs filtered input]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):    -6.7 (se 6.3, t -1.06, Holm p 1.000)
    ensemble test   WORLD-level    -5.3 (se 4.3, Holm p 0.527)   SEED-level Welch    -6.7 (t -1.33, Holm p 0.886)
    ensemble fresh  WORLD-level    -4.0 (se 2.6, Holm p 0.255)   SEED-level Welch    -6.9 (t -1.22, Holm p 0.771)

=== (b) Do uncertainty features and the exploration bonus help? ===

  filt_full - filt_mean_only  [uncertainty features (no bonus)]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):    +2.5 (se 4.4, t +0.56, Holm p 1.000)
    ensemble test   WORLD-level    -3.7 (se 4.1, Holm p 0.527)   SEED-level Welch    +2.5 (t +0.37, Holm p 0.947)
    ensemble fresh  WORLD-level    -2.9 (se 2.5, Holm p 0.255)   SEED-level Welch    +3.0 (t +0.50, Holm p 1.000)

  filt_full_bonus - filt_full  [exploration bonus (with std features)]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):   -12.3 (se 5.7, t -2.16, Holm p 0.491)
    ensemble test   WORLD-level   -11.0 (se 2.9, Holm p 0.001)   SEED-level Welch   -12.3 (t -2.42, Holm p 0.256)
    ensemble fresh  WORLD-level    -8.5 (se 1.7, Holm p 0.000)   SEED-level Welch   -12.0 (t -2.60, Holm p 0.221)

  filt_mean_bonus - filt_mean_only  [exploration bonus (mean only)]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):    +7.2 (se 7.0, t +1.03, Holm p 1.000)
    ensemble test   WORLD-level    +5.4 (se 2.3, Holm p 0.078)   SEED-level Welch    +7.2 (t +0.76, Holm p 0.947)
    ensemble fresh  WORLD-level    +3.4 (se 1.4, Holm p 0.044)   SEED-level Welch    +4.9 (t +0.71, Holm p 1.000)

=== (c) Does the regime structure help? (regime probabilities; IMM vs K=1 filter) ===

  filt_full - filt_no_regime  [regime layer]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):   -12.0 (se 9.9, t -1.22, Holm p 1.000)
    ensemble test   WORLD-level   -19.0 (se 5.6, Holm p 0.004)   SEED-level Welch   -12.0 (t -1.22, Holm p 0.886)
    ensemble fresh  WORLD-level   -30.6 (se 3.3, Holm p 0.000)   SEED-level Welch   -21.4 (t -1.80, Holm p 0.526)

  filt_full - filt_single  [IMM filter vs single-regime filter]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):   -10.9 (se 4.7, t -2.31, Holm p 0.491)
    ensemble test   WORLD-level    -7.6 (se 5.6, Holm p 0.527)   SEED-level Welch   -10.9 (t -1.82, Holm p 0.537)
    ensemble fresh  WORLD-level   -12.3 (se 3.4, Holm p 0.001)   SEED-level Welch   -15.0 (t -2.50, Holm p 0.221)

=== summary (ensemble SEED-level Holm p, test / fresh) ===
  filt_full - raw_proxy                  -51.3 /   -58.7   p 0.066 / 0.032   BORDERLINE
  filt_full - filt_mean_only              +2.5 /    +3.0   p 0.947 / 1.000   NOT ESTABLISHED
  filt_full_bonus - filt_full            -12.3 /   -12.0   p 0.256 / 0.221   NOT ESTABLISHED
  filt_mean_bonus - filt_mean_only        +7.2 /    +4.9   p 0.947 / 1.000   NOT ESTABLISHED
  filt_full - filt_no_regime             -12.0 /   -21.4   p 0.886 / 0.526   NOT ESTABLISHED
  filt_full - filt_single                -10.9 /   -15.0   p 0.537 / 0.221   NOT ESTABLISHED
  oracle - filt_full                      -6.7 /    -6.9   p 0.886 / 0.771   NOT ESTABLISHED

HOW TO READ
- WORLD-level conditions on ONE ensemble of 5 agents per arm: its se is world sampling only. A world-level effect is a statement about these trained agents, not about the arm. SEED-level includes training variability but has n = 5 per arm.
- Holm is over the 7 contrasts within each (level, block). Test and fresh worlds are not independent evidence: they re-score the same agents. The single-agent column uses the test worlds only and its Holm p is new here (no earlier number to compare).
- The ensemble idea arose after test and fresh results were seen and is not pre-registered: ensemble contrasts are exploratory. The 7 contrasts themselves were listed in advance.
- The VERDICT is a rule of this script (see its docstring), not a result of the earlier handoffs.
- Oracle vs filtered compares arms whose inputs differ; the oracle arm has no posterior std, so read it next to filt_full only.
- Test worlds were also used earlier to choose configurations; fresh worlds are the cleaner check for absolute numbers.
```
