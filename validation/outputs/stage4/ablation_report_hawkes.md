```
handoff check: all 16 values match STAGE3_REVISIT_HANDOFF.md item 8

STAGE 4 ABLATION REPORT (spread_hawkes) -- noise-free cost difference vs TWAP in $, negative = cheaper; contrast X - Y < 0 means X cheaper.
worlds: test 200, fresh 500; single agents: validation-selected checkpoints on the test worlds (tags ['_hawkes'], 30000 episodes).

=== arms ===
arm                 single agents (test): mean sd n   ENSEMBLE test (se)   ENSEMBLE fresh (se)   members mean test/fresh
oracle                             -60.9   15.9   5        -72.4 (10.9)        -81.0 ( 6.4)       -60.9 /   -66.9
raw_proxy                          -31.5   33.0   5        -70.8 ( 7.1)        -74.0 ( 4.3)       -31.5 /   -36.0
filt_full                          -56.1    6.1   5        -77.8 (11.5)        -83.4 ( 6.9)       -56.1 /   -64.5
filt_mean_only                     -39.1   14.8   5        -61.8 (10.8)        -70.2 ( 6.6)       -39.1 /   -48.0

=== (a) Does the information help? (filter vs raw proxy; perfect info vs filter) ===

  filt_full - raw_proxy  [filtered belief vs raw proxy]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):   -24.6 (se 13.5, t -1.82, Holm p 0.381)
    ensemble test   WORLD-level    -7.0 (se 5.4, Holm p 0.199)   SEED-level Welch   -24.6 (t -1.64, Holm p 0.344)
    ensemble fresh  WORLD-level    -9.4 (se 3.3, Holm p 0.008)   SEED-level Welch   -28.5 (t -2.11, Holm p 0.227)

  oracle - filt_full  [oracle input vs filtered input]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):    -4.7 (se 9.3, t -0.51, Holm p 0.637)
    ensemble test   WORLD-level    +5.5 (se 3.0, Holm p 0.140)   SEED-level Welch    -4.7 (t -0.62, Holm p 0.561)
    ensemble fresh  WORLD-level    +2.4 (se 1.8, Holm p 0.186)   SEED-level Welch    -2.4 (t -0.31, Holm p 0.771)

=== (b) Do uncertainty features and the exploration bonus help? ===

  filt_full - filt_mean_only  [uncertainty features (no bonus)]   VERDICT: NOT ESTABLISHED
    single agents, paired by training seed (test worlds, n=5):   -17.0 (se 8.8, t -1.92, Holm p 0.381)
    ensemble test   WORLD-level   -16.1 (se 3.1, Holm p 0.000)   SEED-level Welch   -17.0 (t -2.38, Holm p 0.180)
    ensemble fresh  WORLD-level   -13.2 (se 2.2, Holm p 0.000)   SEED-level Welch   -16.5 (t -2.32, Holm p 0.227)

=== summary (ensemble SEED-level Holm p, test / fresh) ===
  filt_full - raw_proxy                  -24.6 /   -28.5   p 0.344 / 0.227   NOT ESTABLISHED
  filt_full - filt_mean_only             -17.0 /   -16.5   p 0.180 / 0.227   NOT ESTABLISHED
  oracle - filt_full                      -4.7 /    -2.4   p 0.561 / 0.771   NOT ESTABLISHED

HOW TO READ
- WORLD-level conditions on ONE ensemble of 5 agents per arm: its se is world sampling only. A world-level effect is a statement about these trained agents, not about the arm. SEED-level includes training variability but has n = 5 per arm.
- Holm is over the contrasts available (7 Gaussian, 3 Hawkes) within each (level, block). Test and fresh worlds are not independent evidence: they re-score the same agents. The single-agent column uses the test worlds only and its Holm p is new here (no earlier number to compare).
- The ensemble idea arose after test and fresh results were seen and is not pre-registered: ensemble contrasts are exploratory. The 7 contrasts themselves were listed in advance.
- The VERDICT is a rule of this script (see its docstring), not a result of the earlier handoffs.
- Oracle vs filtered compares arms whose inputs differ; the oracle arm has no posterior std, so read it next to filt_full only.
- Test worlds were also used earlier to choose configurations; fresh worlds are the cleaner check for absolute numbers.
- HAWKES: 4 arms only (oracle, raw_proxy, filt_full, filt_mean_only), so 3 of the 7 contrasts exist and Holm is over 3 (less strict). Seeds 3-4 were added after the 3-seed results were seen. The filter is overconfident on this arm. Handoff check covers the arm values only.
```
