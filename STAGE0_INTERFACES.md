# STAGE 0 -- SHARED INTERFACES (EDI project)

**Status: DRAFT for team ratification.** Code: `src/interfaces.py`. Conformance test: `validation/validate_interfaces.py` (20/20 checks pass against the real Stage 1 + existing code). **Amended after Stages 2-4:** Section 9 (Stage 2) and Section 10 (Stages 3-4: `PolicyInput.obs`, the implemented live API, the opt-in particle cloud). The 20/20 result predates those amendments; rerun `validate_interfaces.py` after the `PolicyInput` change.

**What this document is built from:** the existing validated code (`simulator.py`, `observation_model.py`, `particle_filter.py`, `heuristic_policy.py`, `frontend/engine.js`) and the Stage 1 code/handoff. **What it is NOT built from:** the team brief for Stages 2-4 was cut off mid-Stage 2 in the material I received, so the needs of the IMM-PF (beyond the outline), the decision layer (Double DQN is only mentioned in `heuristic_policy.py`) and the backend/frontend stage are **inferred, not read**. Persons 2-4: check Section 7 against your actual brief and flag misfits rather than adapting silently.

---

## 1. Global conventions

| Item | Convention |
|---|---|
| Steps | Decision steps `t = 0..N-1`. State arrays (`S, kappa, alpha, regime, q`) have `N+1` entries; observations, beliefs, actions, step logs have `N`. Default `N = 50`, `tau = T/N`. |
| Information set | `y_t` is measured over `((t-1)tau, t tau]`, depends only on `x_t` and `regime[t]`. The decision `v_t` uses `I_t = {y_0..y_t}` and is executed over `[t tau, (t+1)tau)`. **[DECISION 1]** |
| Belief | Filtering posterior `p(x_t, s_t | y_0..y_t)`, taken from the **reweighted, pre-resample** particles (as in the validated bootstrap PF). |
| Regime timing | `regime[t]` is the regime whose parameters generated `x_t` from `x_{t-1}` (`x_{t+1} = f(x_t, s_{t+1})`, `s_{t+1} ~ P[s_t,:]`). `P` is **per step**. |
| Oracle barrier | `kappa`, `alpha`, `regime` are truth. Only the evaluation code and oracle benchmarks may read them. The harness passes `truth=None` to any policy with `uses_oracle_truth = False`. The `regime` argument of `observe_step` is a generator-side input and must never reach a filter. |
| Seeds | `seed_streams(seed)` gives `sim=seed, obs=seed+10000, filter=seed+20000, policy=seed+30000`; the regime chain derives its own stream from `[seed, REGIME_RNG_TAG]`. Same `seed` gives identical market paths across policies (checked: alpha path identical across policies in all 3 observation modes). |
| Units | As in `simulator.py`. kappa/alpha are about 1e-5 to 1e-3, so **never compare them in absolute terms without the reference scale** (`ObsParams.kappa_ref, alpha_ref`). |
| Versioning | Every serialised object carries `schema_version` (currently `0.1-draft`). |

## 2. Contract C1 -- Simulator -> Observation model / oracle

`TruthStep(t, S, kappa, alpha, regime, q, N, shock_step)` mirrors the dict yielded by `run_stepwise` (`TruthStep.from_state`). Single-regime simulators emit `regime = 0` (the original `MarketSimulator` has no `regime` key; `from_state` defaults it).
`WorldConfig` (JSON-serialisable: sim params, regime specs and `P`, observation kind and params) is what an experiment logs so any episode can be regenerated.
Protocol: pull-based generator, unchanged (`state = next(gen); state = gen.send(v_t); StopIteration.value = Episode`).

## 3. Contract C2 -- Observation model -> Particle filter

`Observation(t, spread, volume, realvol, extras)`. Core channels are the only ones the current filter reads. `extras` may hold `spread_ticks, n_open, n_close, n_events, hawkes_excitation` (a count- or tick-based likelihood in Stage 2 may use them).
All three models (`gaussian`, `hawkes`, `spread_hawkes`) satisfy `ObservationModel`: `reset(seed)`, `observe_step(kappa_t, alpha_t, regime=0, rng=None) -> dict`. Hawkes models are stateful: call in time order after `reset`.

**Two non-conformances found (neither is a code change I made):**
1. The original `observation_model.ObservationModel.observe_step` has **no `regime` argument**. Use `hawkes_observation.GaussianObservation` (same maths, adds the argument) or `make_observation_model_v2("gaussian")`.
2. The **v1 Hawkes model has a within-step look-ahead**: row t counts events in `[t tau, (t+1)tau)` yet the decision at t sees it. This violates Decision 1 (its own docstring flags this). `spread_hawkes` already uses the causal window.

## 4. Contract C3 -- Particle filter -> Decision layer

`BeliefState(t, regime_probs[K], kappa_mean, kappa_std, alpha_mean, alpha_std, ess, per_regime[RegimeBelief], n_particles, loglik_increment)`.
* Pooled fields are moments of the regime **mixture**; they are what the existing heuristic reads (`alpha_mean`, `alpha_std`).
* A single-regime filter fills `K=1`, `regime_probs=[1.0]` (adapter: `belief_from_bootstrap_pf`, wraps the untouched PF).
* `to_vector()` gives a fixed-order feature vector `[alpha_mean, alpha_std, kappa_mean, kappa_std, p_0..p_{K-1}]`, length `4+K`, raw units (scaling is the policy's job).
* `Estimator` protocol: `reset(seed)`, `update(obs) -> BeliefState`. The current PF returns `None` from `step()` and stores `history`; the adapter reads `history[-1]`, so the PF needs no change.
* `validate_belief` rejects non-finite values, negative std, probabilities not summing to 1, wrong K.
* **[DECISION 3]** For K>1, define `ess` as the ESS of the **combined** weighted particle set (sum over regimes of weights), and report per-regime ESS in `per_regime`. Please confirm this is what Stage 2 will compute.

## 5. Contract C4 -- Decision layer -> Evaluation

* `Policy` protocol: `reset(seed)`, `act(PolicyInput) -> shares to sell`. `PolicyInput = (t, N, q, belief | None, truth | None, obs | None)`; the trailing optional `obs` (the flat observation vector of a learned agent) was added in Stage 4, see Section 10.2.
* `StepLog` per step (q_before, v, S, S_exec, obs, belief, and truth fields marked **evaluation only**). `EpisodeLog` holds policy, seed, `WorldConfig`, steps, IS, q0, S0, `q_final`, shock step.
* `validate_episode_log` is mandatory before evaluation: steps indexed 0..N-1, `0 <= v <= q`, `q_before[0] = q0`, **IS recomputed** from steps equals logged IS, and **inventory is liquidated**. The last check exists because `IS = S0*q0 - sum(S_exec*v)` silently **understates cost if shares are left unsold**.
* `paired_metrics(logs_by_policy, baseline="twap")`: **one** implementation of mean IS, std, CVaR95 and paired difference vs TWAP (same seeds, same order, else it raises). Regime-conditional metrics and belief-quality metrics (alpha RMSE, coverage, regime-detection accuracy/Brier) should be added to this same module by Person 4 so all policies are scored identically.
* Cost definitions: Python `Episode.IS` counts `S0*q0` against executed proceeds. The JS dashboard shows cumulative cost **on shares sold so far**. These agree only at the end of an episode with full liquidation.

## 6. Contract C5 -- Backend / frontend live-step API

**Finding:** the current dashboard is **fully client-side** (`engine.js` is a JS port of the Python reference; `serve.py` only serves static files). There was no API when this document was first written (the backend now exists, see Section 10.3; the frontend rebuild is Stage 5). The JS port covers **single-regime Gaussian only** (constants hard-coded, parity-checked against Python numbers). Regime switching, Hawkes observations and an IMM-PF do not exist in JS.

**[DECISION 4]** Choose one:
* **A (recommended): Python backend** exposing the JSON below and reusing the Python reference, so there is no parity drift. Cost: needs a running server; `engine.js`/`parity_check.js` become legacy.
* **B: port regime + Hawkes + IMM-PF to JS.** Keeps the no-server demo, but is a large amount of work and each Python change needs a re-parity check.

Endpoints (**implemented in Stage 4C**: `backend/`, FastAPI; Decision 4 option A was taken; see Section 10.3): `POST /api/episode` (body: `seed`, `obs_kind`, regime preset) and `POST /api/episode/{id}/step` (body per `STEP_REQUEST_SCHEMA`: optional `inject_shock` x2-x5, optional `force_regime {regime, duration}`). The step response is `STEP_RESPONSE_SCHEMA` (see `STEP_RESPONSE_EXAMPLE` in `interfaces.py`): `schema_version, episode_id, seed, t, N, done, config, truth, observation, belief, policies{name:{q,v,S,S_exec,cum_cost}}, events[]`. `validate_step_response` checks structure. The `truth` block is for display of the oracle overlay only.

## 7. Where the contracts might not fit later stages (please check)

1. **IMM-PF (Person 2).** The filter needs the regime model (`P`, per-regime CIR parameters, observation parameters). Stage 1 exposes these via `RegimeParams.effective_cir(sim_params, r)` and `RegimeParams.P`. **[DECISION 2]** Does the filter get the **true** values (as the current PF gets true CIR parameters) or deliberately misspecified ones? I assumed true, matching the existing PF, but it changes how flattering the results are. **[Stage 2 update: implemented as assumed -- the IMM-PF takes a `RegimeParams` and by default receives the TRUE one. Misspecified-model sensitivity runs (wrong `P`, wrong stress levels) are in `validation/validate_imm_pf.py`. Still awaiting team ratification.]**
2. **Likelihood.** Contract lets the filter read `extras`. If Stage 2 adds a count/tick-based likelihood it will use them; otherwise it reads only the three core channels and will be overconfident on Hawkes data (Stage 1 measured residual std 1.36x-1.41x the assumed noise). **[Stage 2 update: the IMM-PF reads only the three core channels with a Gaussian likelihood, exactly like the validated bootstrap PF; `extras` are ignored. A single knob `IMMParams.obs_noise_inflation` (default 1.0) widens the assumed noise and is swept in validation. No count/tick-based likelihood was built.]**
3. **Decision layer (Person 3).** `to_vector()` is belief-only. The full RL state (inventory fraction, time left, ...) is Stage 3's to define; I did not guess it. Action space is a scalar `v_t` in shares. **[Stage 3 update: the RL state, action grid and reward are defined in `STAGE3_HANDOFF.md`. The agent's action is an index into a grid of multipliers of the remaining-TWAP rate, turned into shares by the environment; the trained agent needs its flat observation vector, which `PolicyInput` did not carry: see Section 10.2.]**
4. **Regime info in policies.** Whether policies may use `regime_probs` is allowed by the contract; using true `regime` is oracle-only.

## 8. Decisions the team must ratify

| # | Decision | My proposal |
|---|---|---|
| 1 | Timing/information convention | Causal window `((t-1)tau, t tau]`; decision at t uses `y_0..y_t`. Keep `spread_hawkes` as the reference. Fix or retire v1 Hawkes (Person 1: a one-line window shift plus re-validation, or drop v1 from the ablation). |
| 2 | Does the filter know the true regime model? | Yes by default (as today); add a misspecified-model run as a sensitivity test. **Stage 2: implemented this way; misspecified-model results are in `STAGE2_HANDOFF.md` (b).** |
| 3 | K>1 ESS definition | Combined weighted-set ESS, per-regime ESS in `per_regime`. **Stage 2: implemented exactly this way (`ess = 1/sum_j sum_i (mu_j w_ji)^2`, range [1, K*n_particles]); bounds are a HARD check.** |
| 4 | Backend architecture | A: Python backend. **Stage 4: implemented (FastAPI, `backend/`); awaiting ratification.** |
| 5 | Episode horizon vs regime persistence | Default `N=50` with per-step `P`: about 13% of episodes starting calm never see stress (0.96^50). Use stationary `init_probs` for training episodes or lengthen N. **Stages 2 and 3 use stationary initial regime probabilities for the world and the filter prior.** |
| 6 | `PolicyInput.obs` (added in Stage 4) | Keep it: an optional trailing field holding the flat observation vector, so a learned agent can be hosted by the harness and the live API. Reported as verified by `check_agent_policies.py` (X1); awaiting ratification. |

## 9. Stage 2 addendum (Person 2) -- how the IMM-PF uses the contracts

_Status: written against the contracts above and validated by a full run of `validation/validate_imm_pf.py` (10/10 HARD checks; numbers in `STAGE2_HANDOFF.md` (b)). Decisions 2 and 3 are still awaiting team ratification._

**Conforms without change.** `IMMParticleFilter` implements `Estimator` (`reset(seed)`, `update(obs: Observation) -> BeliefState`), reads only `Observation.core_dict()`, never receives `regime` or truth, and returns a `BeliefState` with `K = 2`, `regime_probs`, `per_regime` (mean/std/prob/ESS per regime), pooled mixture moments, combined `ess`, `n_particles = K * n_per_bank` and `loglik_increment`. `validate_belief(b, K=2)` is applied to every belief in validation. A real K=2 belief also removes the one mismatch `validate_interfaces.py` had to whitelist (K=1 bootstrap belief vs `config.n_regimes = 2`); `validate_imm_pf.py` checks that a `StepResponse` built from an IMM belief passes `validate_step_response` with no exceptions.

**Semantics Persons 3 and 4 must know.**
1. `regime_probs` at step `t` is the FILTERED probability `p(s_t | y_0..y_t)` for the regime that generated `x_t` (Stage 1 timing). It is not a forecast. The one-step-ahead forecast `p(s_{t+1} | y_0..y_t)` is exposed as `IMMParticleFilter.next_regime_probs` (equals `P^T mu`), in case the decision layer wants it.
2. `alpha_std` / `kappa_std` are mixture moments, so they include the between-regime spread and can widen sharply when the regime is uncertain. Per-regime moments are in `per_regime`.
3. `to_vector()` for K=2 has length 6: `[alpha_mean, alpha_std, kappa_mean, kappa_std, p_calm, p_stressed]`.
4. The regime prior before the first observation is `RegimeParams.init_probs`. Training episodes should use the stationary distribution (Decision 5); the validation script does.
5. Update cost is reported in the validation output (ms per step); it matters for the live dashboard.

**Where the C2 contract limits Stage 2 (request to Person 1 / Stage 0 owners).**
1. `Observation` carries no mid-price return. Stage 1 measured a price-diffusion std ratio of about 1.58x in stress, so the price path itself is a strong regime signal that no filter can currently use. Adding an optional return channel to C2 would need a change in the observation models and in `interfaces.py`; it has NOT been done and is only a suggestion.
2. Hawkes extras (`spread_ticks`, `n_open`, ...) are ignored by the filter, so on the Hawkes arms the Gaussian likelihood is misspecified by construction. Expect an overconfident filter there; validation reports coverage per arm.
3. The v1 Hawkes arm still violates Decision 1 (look-ahead). Validation runs `gaussian` and `spread_hawkes` by default; v1 only with `--kinds=...,hawkes`.

**Shared metrics.** Regime-detection accuracy, Brier/log-loss/AUC, detection delay and pooled tracking/coverage are implemented filter-agnostically in `src/belief_metrics.py` (arrays in, dicts out) so Person 4 can score every estimator with the same code, as Section 5 asks. `interfaces.paired_metrics` was not modified.

## 10. Stage 3 and Stage 4 addendum (Persons 3 and 4) -- amendments to the contracts

_Status: based on the output of `check_agent_policies.py`, `check_particles.py`, `check_live_session.py`, `smoke_api.py` and `check_guards.py` (all print ALL CHECKS PASSED) and on the Stage 3 files. The source files that were changed (`interfaces.py`, `backend/*.py`) were not seen by the author of this addendum: verify the field lists below against the code before ratifying._

**10.1 Stage 3 against C3 / C4.**
* The environment reads `BeliefState.per_regime` (per-regime alpha/kappa mean and std), `regime_probs`, and, for the no-regime arm, the pooled moments. It requires the filter's K to equal `EnvConfig.n_regimes` (2 for the IMM-PF, 1 for the single-regime filter). It does not use `ess` or `loglik_increment`.
* `BeliefMDPEnv.get_episode()` is NOT the C4 schema. `src/stage3_adapter.py` converts a finished episode into `interfaces.EpisodeLog` (`env_episode_to_log`, `run_logs`). `StepLog.belief` is then a partial dict (`t, regime_probs, alpha_mean, alpha_std, kappa_mean, kappa_std`; no `per_regime`, `ess`, `loglik_increment`), `obs` is `None`, and the agent's action index is not logged. Stage 3 exports these logs for Stage 4: `validation/outputs/stage3c/logs/logs_valsel.pkl` (single agents + TWAP) and `logs_ensemble_valsel.pkl` (`ens_<arm>` + TWAP); see `STAGE3_HANDOFF.md`.

**10.2 `PolicyInput.obs` (resolves the open Stage 3 request).** `PolicyInput` gains `obs: Optional[np.ndarray] = None`. Reason: the trained agents' input includes the spread proxy and the normalised inventory and time, which `PolicyInput` did not carry, so they could not be hosted by the `Policy` protocol. With `obs` they are hosted as ordinary policies. Reported checks (`check_agent_policies.py`): for the `oracle`, `raw_proxy`, `filt_full`, `filt_no_regime` and `filt_full_bonus` arms the live observation vector equals the environment's exactly (max |d| 0.0; dimensions 7, 5, 13, 7, 13), the live trade path and implementation shortfall equal the environment's exactly, and the information barrier holds (the oracle arm gets truth, the raw-proxy arm gets neither truth nor belief, filtered arms get belief but not truth). Compatibility: a trailing optional field breaks only positional construction of `PolicyInput`.

**10.3 C5 implemented (Stage 4C).** FastAPI backend in `backend/` (`api_server.py`, `live_session.py`).
* Endpoints seen in the checks: `POST /api/episode` (body fields seen: `seed`, `obs_kind`, `policies` = subset of the hosted policies, `particle_sample`; unknown fields, an unknown policy or a bad `obs_kind` -> 422), `POST /api/episode/{id}/step` (body per `STEP_REQUEST_SCHEMA`: optional `inject_shock` x2-x5, optional `force_regime {regime, duration}`; out-of-range values and a second shock -> 422, and a rejected request does not advance the episode), `/api/policies` (lists the defaults) and `/api/episode/{id}/result` (409 before the episode ends, 404 for an unknown episode). Complete request schemas are in `backend/api_server.py` and the Stage 4 handoff.
* Failure handling: stepping a finished episode -> 409; a policy runtime failure -> 409 and a contract violation -> 500, both with the state rolled back so a retry reproduces the unfailed episode exactly; the oldest sessions are evicted.
* Hosted policies: `twap`, `heuristic_belief`, `ce_filtered`, `ce_filtered_grid`, `ce_oracle`, `ce_oracle_grid` (the last two get truth, the others do not), plus the trained agents (single agents and ensembles of them). Ensembles and the bonus arms are Gaussian-only (rejected on `spread_hawkes`), `filt_single` is not hosted, and a single agent does step on `spread_hawkes`. Agents trained on spread-Hawkes observations exist for four arms (`oracle`, `raw_proxy`, `filt_full`, `filt_mean_only`; 3 seeds, tag `_hawkes`, see `STAGE3_HANDOFF.md`); which checkpoint the backend loads for a Hawkes episode is defined in the backend code, which the author of this addendum has not seen. A Gaussian-trained agent run on Hawkes observations would be out of distribution, and the Stage 2 filter is overconfident on that arm.
* Guards (circuit breaker, stress band on the multiplier) are a backend feature outside the required C5 keys; they apply only to non-TWAP policies and leave unguarded policies unchanged (`check_guards.py`, G3).
* Verified: every step response passes `validate_step_response`; final `cum_cost` equals IS (max gap about 1e-10); live equals batch for all hosted benchmark policies on both observation arms (max |dIS| and |dv| 0.0); live noise-free cost equals the saved Stage 3 scores for the `oracle` and `filt_full_bonus` ensembles at 2000 particles (max gap 3e-12).
* The live checks run the filter at 500 particles per regime, while the agents were trained and scored on beliefs from 2000. A live run therefore sees slightly different beliefs from the evaluated ones; say so on any dashboard, or host at 2000.

**10.4 Optional `particles` block in the step response (opt-in).** `POST /api/episode` accepts `particle_sample` in 0..2000 (default 0; above the filter size or negative -> 422). When above 0 the step response gains `particles = {t, n_per_regime, regimes: [{regime, prob, n_total, alpha[], kappa[], w[]}]}`. Per the patch description: the cloud is the filter's reweighted PRE-resample posterior at step t, the sample is uniform without replacement from a separate RNG stream seeded by the episode seed and t (so a retry returns the same sample and no filter random numbers are drawn), and `w` is the particle weight times `n_total` (1.0 is an average weight). Reported checks (`check_particles.py`, default observation arm only, 500 particles per regime): beliefs, trade paths and IS are identical with and without sampling (shock at step 5); with the sample equal to the whole cloud the weighted mean and std of alpha and kappa equal the belief's per-regime values (relative error 6.6e-14 and 2.9e-10) and `prob` equals `regime_probs`; responses still pass `validate_step_response` (so the contract tolerates the extra key) and are JSON-safe; the key is absent by default. Size: about 29 KB per step at 200 particles per regime versus 1.7 KB without.

**10.5 Open items.**
1. The `particles` check ran on the default observation arm only; the `spread_hawkes` arm is untested for it.
2. `validate_interfaces.py` has not been reported as rerun after the `PolicyInput` change.
3. Decisions 2, 3, 4 and 6 (Section 8) still need team ratification.
4. Decision 1 (v1 Hawkes look-ahead) is unchanged: v1 is excluded from the ablations.

---

# HANDOFF SYNOPSIS (Stage 0)

**(a) What was built.** `src/interfaces.py`: dataclasses and protocols for the five module boundaries (truth step, observation, belief, policy/episode log, live-step API), plus validators, a paired-metrics function, JSON schemas for the step API, and adapters that wrap existing code without modifying it. `validation/validate_interfaces.py`: runs real episodes through the contracts. This document.

**(b) What the checks show.** `validate_interfaces.py` runs 6 seeds x 2 policies (TWAP, belief-heuristic) x 3 observation models (gaussian, hawkes, spread_hawkes) on `RegimeSwitchingSimulator` with the bootstrap PF (1000 particles), and passes 20/20 checks: truth steps well-formed, core channels finite and positive, beliefs valid (K=1), episode logs valid (indices, `0<=v<=q`, IS recomputed, inventory liquidated), alpha paths identical across policies (common random numbers), StepResponse structure and JSON serialisation valid, and a synthetic K=2 belief passes while a bad one is rejected. Existing files (`simulator.py`, `observation_model.py`, `particle_filter.py`, `heuristic_policy.py`) were not modified. **These checks show the contracts fit today's code; they say nothing about the IMM-PF or any learned policy, which do not exist yet.** Incidentally the paired belief-vs-TWAP IS difference on 6 episodes was +80 to +86 (s.e. about 48-51, i.e. not significant and n is tiny). It is a smoke-test number, not a result, but it hints the single-regime-tuned heuristic may not beat TWAP in the regime world, which Stage 3 should test properly.

**(c) Interface.** See sections 2-6 and `src/interfaces.py`. Main entry points: `TruthStep.from_state`, `Observation.from_dict`, `belief_from_bootstrap_pf`, `BeliefState`, `PolicyInput`, `episode_log_from_episode`, `validate_episode_log`, `paired_metrics`, `step_response`, `validate_step_response`, `seed_streams`.

**(d) Open items / caveats.** (1) Draft only: nothing is agreed until the team ratifies Section 8. (2) Stages 2-4 requirements are inferred; the brief I saw was truncated. (3) v1 Hawkes violates the proposed timing convention; the original `ObservationModel` lacks the `regime` argument. (4) The API in section 6 is proposed and unimplemented; the dashboard has no backend today and its JS engine only supports single-regime Gaussian. (5) All Stage 1 parameters remain uncalibrated. (6) The conformance test uses a small particle count and 6 seeds; it checks structure, not statistical performance.

**(d) Assumptions, simplifications, open questions.** (1) The original brief for Stages 2-4 was cut off in the material this document was built from, so the needs of the later stages were inferred; Sections 9 and 10 record where they actually differed (the IMM-PF needed the regime model, the learned agent needed an observation vector, the backend added an opt-in particle block). (2) `validate_interfaces.py` shows the contracts fit the code as of Stage 0 only; later conformance is evidenced by each stage's own check scripts, not by a rerun of it. (3) Decisions 1-6 are proposals that have been built against but not formally ratified by the team. (4) The Stage 0 contracts do not carry a mid-price return (Stage 2 suggestion) and the Hawkes extras are not read by the filter; both limit what Stage 2 and Stage 3 can show on the Hawkes arms. (5) `schema_version` is still `0.1-draft`; the additions in Section 10 should bump it once ratified.