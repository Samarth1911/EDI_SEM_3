# STAGE 1 -- HANDOFF SYNOPSIS (Person 1: World Generation)

_Status: Stage 1 code complete and validated (all HARD checks pass). Interface below is the as-built proposal for Stage 0; it is NOT yet ratified by the team. Parameters are illustrative, not calibrated to data._

## Files that make up Stage 1

| File | Role |
|---|---|
| `src/regime_simulator.py` | `RegimeSwitchingSimulator(MarketSimulator)`: 2-state Markov regime layer on the CIR simulator |
| `src/hawkes_observation.py` | v1 `HawkesObservationModel` + `GaussianObservation`; selector `make_observation_model("gaussian"\|"hawkes")` |
| `src/spread_hawkes.py` | v2 `SpreadHawkesObservationModel` (recommended "realistic" arm); selector `make_observation_model_v2("gaussian"\|"hawkes"\|"spread_hawkes")` |
| `validation/validate_stage1.py` | regime + v1 Hawkes validation (regenerates the auto handoff) |
| `validation/validate_spread_hawkes.py` | v2 validation |
| `validation/outputs/` | figures 04, 05, 06, 08; `stage1_validation_summary.json`, `spread_hawkes_validation_summary.json` |

`simulator.py`, `observation_model.py`, `particle_filter.py`, `heuristic_policy.py` are untouched. `validation/stage1_extras.py` is superseded by `spread_hawkes.py` and can be ignored.

Run (from `EDI/validation/`): `python validate_stage1.py` then `python validate_spread_hawkes.py`.

---

## (a) What was built

1. **Regime-switching simulator.** Hamilton-style 2-state Markov chain (0 = calm, 1 = stressed), per-step transition matrix `P = [[0.96, 0.04], [0.08, 0.92]]`. CIR kappa/alpha run conditional on the active regime via multipliers on the base `SimParams` (long-run mean, mean-reversion speed, vol-of-vol, price sigma). Stressed default: alpha long-run x2.5, kappa x1.6, lambda x2, price sigma x1.6. Presets: `RegimeParams.default()`, `.subtle()` (alpha x1.15, kappa x1.08), `.single()` (identical to the original simulator).
2. **v1 Hawkes observation model.** Univariate exponential-kernel Hawkes via Ogata thinning; regime-dependent baseline and branching ratio (mu0 = 50, mu_mult = [1.0, 1.5], branching = [0.5, 0.75], beta = 25). Gaussian model kept behind the same interface as a selectable alternative.
3. **v2 spread Hawkes observation model (recommended).** Spread is an integer tick state. Opening events self-excite; closing rate is state-dependent. Follows Fosset-Bouchaud-Benzaquen (arXiv:1912.00359) Sec. 4 with the Sec. 4.5 stabiliser. Defaults (verified in `spread_hawkes.py`): mu0 = 50, mu_mult = (1.0, 1.5), branching = (0.4, 0.8), beta = 10, lam_close = 60 (closing rate = 60*(S-1)), tick_value = 0.0015 $/tick, alpha_coupling = 0, tau = 0.02.

## (b) What the validation shows

### Regime simulator and v1 Hawkes (`validate_stage1.py`)

**HARD**
- Single-regime config vs original `MarketSimulator`: max |diff| = 0.0 over S, S_exec, kappa, alpha, v, q, IS across 3 cases (bit-for-bit).
- Markov chain (400,000 steps): max |P_hat - P| = 0.0008; pi_hat = [0.663, 0.337] (theory [0.667, 0.333]); mean sojourn = [24.9, 12.6] steps (theory [25.0, 12.5]).
- CIR mean reversion per regime: worst relative error 1.3% (tolerance 15%).
- v1 Hawkes thinning: calm count z = +1.60, KS p = 0.203; stressed z = +1.11, KS p = 0.568.
- **Branching ratio:** theory calm 0.50, stressed 0.75 (both < 1; >= 1 rejected at construction). Moment estimate: 0.509 / 0.754.

**INFO -- regime distinctness (default preset)**
- Stressed time fraction 0.348. Realised mean alpha ratio stressed/calm = **1.49** vs long-run-mean ratio 2.50 (stressed spells are short relative to mean-reversion time).
- Price-diffusion std ratio 1.58 (theory 1.60).
- Separability of TRUE alpha by regime: AUC 0.902, KS 0.623, Cohen's d 1.95. One-step Gaussian spread proxy AUC 0.901.
- Stressed spells: mean length 13.0 steps; alpha reaches the calm/stressed midpoint in 81% of spells (median 2 steps).
- `subtle` preset: alpha AUC 0.867, Cohen's d 1.63.

**INFO -- v1 Hawkes vs Gaussian (calm-only, 6,000 steps)**
- Spread-residual ACF1: Gaussian +0.017 vs Hawkes +0.089. Skew +0.01 vs +0.32. P(resid > 3 assumed-noise-std) 0.0008 vs 0.0202. Residual std ratio 1.36. Count Fano 1.34.

### v2 spread Hawkes (`validate_spread_hawkes.py`, all HARD pass)

| Check | Calm | Stressed |
|---|---|---|
| Stationary mean ticks: theory / estimate | 2.389 / 2.401 (z = +0.85) | 7.250 / 7.299 (z = +0.68) |
| Opening events per step: theory / estimate | 1.667 / 1.682 | 7.500 / 7.541 |
| Closing events per step (should equal opening) | 1.682 | 7.540 |
| Time-rescaling KS p (opening / closing) | 0.029 / 0.126 | 0.147 / 0.503 |
| Std of ticks; opening-count Fano | 1.23; 1.10 | 2.93; 1.44 |

Exact stationary mean: `E[S] = 1 + (mu/(1-a)) / lam_close`.

Calm-only proxies (6,000 steps):

| Model | resid std / assumed noise | ACF1 | skew | excess kurt | P(>3 sd) |
|---|---|---|---|---|---|
| Gaussian | 0.99 | +0.017 | +0.01 | -0.03 | 0.0008 |
| Hawkes v1 | 1.36 | +0.089 | +0.32 | +0.18 | 0.0202 |
| Spread Hawkes v2 | 1.41 | +0.180 | +0.32 | +0.21 | 0.0248 |

Mean spread difference vs Gaussian: -0.000033 $ (moment matching holds).

**Switching episode (realised mean spread ticks by TRUE regime):** calm 2.86 (stationary theory 2.39), stressed 4.67 (theory 7.25). Regimes only partly express, and calm is contaminated by post-stress carry-over (likely leftover excitation; not tested).

**Figures.** `04_regime_paths.png` shows clearly distinct calm/stressed periods. `08_spread_hawkes.png` shows the stressed spread distribution shifted right but overlapping heavily with calm, so the spread alone is only a modest regime signal. Persistence is short: residual ACF is 0.180 at lag 1, 0.065 at lag 2, about 0 by lag 4.

## (c) Data interface produced (DRAFT -- pending Stage 0)

**Simulator -> Observation Model / Filter / Decision**
- `RegimeSwitchingSimulator(params, regime_params, seed)`; `.run_stepwise(shock_step, shock_jump_kappa_mult, shock_jump_alpha_mult, seed, forced_regime_path)` uses the same pull-based generator protocol as `MarketSimulator` (`state = next(gen); state = gen.send(v_t)`; `StopIteration.value = episode`).
- Per-step `state` dict: `{t, S, kappa, alpha, q, N, shock_step, regime}`. `kappa`, `alpha`, `regime` are TRUE/hidden values. Belief-based policies must not read them (oracle only).
- `RegimeEpisode` = existing `Episode` fields + `regime: np.ndarray (N+1,)` int. Timing: `regime[t]` is the regime whose parameters generated `x_t` from `x_{t-1}`; `regime[0]` only fixes the initial state.
- `force_regime_window(path, start, duration, regime=1)` supports controlled tests and dashboard shock injection.

**Observation Model -> Particle Filter**
- All models: `observe_step(kappa_t, alpha_t, regime=0, rng=None) -> dict` with keys `spread, volume, realvol` (the only keys the current filter reads by name).
- v1 Hawkes adds `n_events, hawkes_excitation`.
- v2 spread Hawkes adds `spread_ticks, n_open, n_close, n_events (= n_open), hawkes_excitation`.
- `observe_episode(episode, seed) -> pandas.DataFrame` with columns `t, spread, volume, realvol, [extras,] regime, true_kappa, true_alpha`.
- **Hawkes models are STATEFUL:** call `observe_step` in time order after `reset(seed)`.

## (d) Assumptions, simplifications, open questions

1. **Stage 0 not done.** The interface above is as-built, offered as input to Stage 0. Flag mismatches rather than silently adapting.
2. **Uncalibrated parameters.** Regime multipliers, transition matrix, Hawkes parameters, tick_value and proxy loadings are illustrative, not fitted to data.
3. **Timing convention conflict.** v2 is causal: row t covers ((t-1)*tau, t*tau] with baseline set by `regime[t]`. v1 Hawkes still has a within-step look-ahead (row t counts events in [t*tau, (t+1)*tau) conditioned on x_t). Stage 0 must ratify ONE convention; one of the two models may then need a small edit.
4. **Short episodes vs regime persistence.** With N = 50 steps starting calm, P(no stressed step) is about 0.96^50 = 0.13, so ~13% of default training episodes never see stress unless `init_probs` is set to the stationary distribution (option exists) or the horizon is lengthened.
5. **Regimes only partly express.** Realised stressed/calm alpha ratio is 1.49x, not 2.5x, and realised spread ticks are 4.67 (stressed) vs 7.25 theory, with calm inflated to 2.86 vs 2.39. Person 2's detector faces gradual, contaminated transitions, and regime-specific stationary means are biased in both directions.
6. **Short spread memory (v2).** Closing memory is about 1/lam_close, roughly 0.8 steps. lam_close is the persistence knob; changing it requires rescaling tick_value to keep moment matching.
7. **Regime enters v2 only through mu and the branching ratio.** `alpha_coupling = 0`, so the opening baseline does not depend on the true alpha. Spread loads on alpha through the proxy equation, not the point process.
8. **Filter misspecification (intended ablation).** The PF likelihood is Gaussian with known means; Hawkes proxies are non-Gaussian, autocorrelated, fatter-tailed (P(>3 sd) about 0.02-0.025 vs 0.0008) and mean-shifted in stress. Expect an overconfident particle filter on these arms. Stage 2 should decide whether to add a count-based or tick-based likelihood.
9. **Model scope.** Univariate Hawkes only. Bacry et al. (2013) is a mutually-exciting setup, and the Fosset et al. price-feedback / Q-Hawkes terms are not implemented. No dependence of events on our own trading. Described here as "following" those papers, not implementing them; primary sources have not been checked line by line.
10. **Existing jump shock retained.** `shock_step` (within-regime jump) is distinct from a regime switch; the old filter validation used the former.
