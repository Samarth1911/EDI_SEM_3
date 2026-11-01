# EDI -- Optimal trade execution under latent liquidity

Research project extending Macri & Lillo (2025, *Applied Mathematical Finance*): optimal execution when market liquidity is latent, estimated by sequential Monte Carlo, and acted on by a reinforcement-learning agent.

> Status (1 Nov 2026): project reported complete after a team check; Stage 2 full-mode validation and the Stage 4C hosting check were re-run and pass. Per-stage detail is in the handoff documents; current status per stage is in `STATUS_SUMMARY.md`. The consolidated project synopsis (Person 4's final deliverable) may still replace this README.

## Pipeline

```
Simulator (Almgren-Chriss + CIR impact, Markov regimes)      Stage 1  src/simulator.py, src/regime_simulator.py
   -> Observation model (Gaussian | Hawkes | spread-Hawkes)  Stage 1  src/observation_model.py, hawkes_observation.py, spread_hawkes.py
   -> Filter (bootstrap PF | IMM-PF)                         Stage 2  src/particle_filter.py, src/imm_particle_filter.py
   -> Decision layer (belief-MDP, Double DQN, ensembles)     Stage 3  src/belief_mdp_env.py, env_factory.py, ddqn.py, ddqn_train{,2,3,4}.py,
                                                                      stage3c_bonus.py, belief_cache.py, stage3_adapter.py
   -> Evaluation, benchmarks, SHAP, agent hosting            Stage 4  src/barger_lorig.py, validation/run_stage4_*.py, backend/agent_policies.py
   -> Dashboard                                              Stage 5  frontend/
```
All module boundaries are defined in `src/interfaces.py` and documented in `STAGE0_INTERFACES.md`.

## Layout

| Path | What |
|---|---|
| `src/` | model code (see pipeline), `interfaces.py` shared contracts, `belief_metrics.py` filter-agnostic belief scoring, Stage 3 environment / agent / training loops / cached-belief environment / Stage 0 log adapter, `barger_lorig.py` (Stage 4 benchmark) |
| `validation/` | validation and experiment scripts. Outputs: Stage 1-2 in `validation/outputs/`; Stage 3 in `validation/outputs/stage3c/` (`results_base*.jsonl`, `ckpt/`, `logs/`, `ens_nf_*.npz`, reports) and `validation/outputs/belief_cache/`; Stage 4 in `validation/outputs/stage4*/` |
| `backend/` | Stage 4C live backend: `api_server.py` (FastAPI step API), `live_session.py` (`LiveEpisode`), `agent_policies.py` (agents and ensembles behind the `Policy` protocol), `guards.py`, `apply_particle_patch.py` |
| `validation/check_agent_policies.py` | Stage 4C check script (lives in `validation/`) |
| `frontend/` | Stage 5: `index.html` (offline demo), `dashboard.html` (live backend), `style.css`, `engine.js` + `app.js` (offline demo), `model.js` (data layer), `model.test.js`, `parity_check.js`, `serve.py` (see Stage 5 note below) |
| `STAGE0_INTERFACES.md` ... `STAGE4_HANDOFF.md`, `STAGE3_REVISIT_HANDOFF.md` | per-stage handoff documents, each ending in a HANDOFF SYNOPSIS. `STAGE3_HANDOFF.md` and `STAGE3_REVISIT_HANDOFF.md` are FINAL |
| `summary_stage3.md` | Person 3's running summary of Stage 3 |
| `STATUS_SUMMARY.md` | what is done, what is provisional, what remains |

## Setup and running

```
pip install -r requirements.txt          # numpy, scipy, matplotlib, pandas
python validation/validate_filter.py     # milestone-1 bootstrap filter validation
python validation/validate_interfaces.py # Stage 0 conformance
python validation/validate_stage1.py     # Stage 1: regimes + Hawkes (add --quick for a smoke test)
python validation/validate_spread_hawkes.py
python validation/validate_imm_pf.py     # Stage 2: IMM-PF (add --quick for a smoke test first)
```
Scripts add `src/` to the path themselves; run them from the project root unless noted. Modules in `src/` use flat imports, so run their `__main__` demos (e.g. `stage3c_bonus.py`, `stage3_adapter.py`) from inside `src/`.

**Generated files are not in the repository.** The belief cache, checkpoints (`ckpt/`) and training logs are git-ignored and must be regenerated with the Stage 3 commands below (the cache can be many GB).

### Stage 3 (run from `validation/`; PowerShell shown, bash is identical)

```
# 1. (filtered arms only, optional but ~19x faster) build the belief cache with the REAL filter; resumable
python build_belief_cache.py --filter imm    --what val test train --train-seed 0
python build_belief_cache.py --filter single --what val test train --train-seed 0     # repeat per training seed

# 2. train + evaluate arms (5 training seeds, 30000 episodes, 200 test worlds)
python run_stage3c_multiseed.py --arms oracle,raw_proxy --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --tag _abl
python run_stage3c_multiseed.py --arms filt_full,filt_mean_only,filt_no_regime,filt_single --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --use-cache --tag _abl
python run_stage3c_multiseed.py --arms filt_full_bonus,filt_mean_bonus --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --use-cache --bonus-beta 1.0 --tag _abl_b1

# 3. reports
python compare_arms.py --tags _abl _abl_b1                      # per-arm vs TWAP + paired contrasts (a), (b), (c)
python score_fresh.py --tags _abl _abl_b1 --n-fresh 500 --use-cache
python export_agent_logs.py --tags _abl _abl_b1 --which valsel --n 200

# 4. revisit: ensembles (mean Q of an arm's 5 val-selected agents) and ablation contrasts
python score_ensemble.py ...            # add --obs-kind spread_hawkes for the Hawkes arms
python ens_contrasts.py ...             # 7 pre-listed contrasts, Holm-adjusted
python export_ensemble_logs.py ...      # EpisodeLog pickle of the ensembles
```
`--tag` keys the results file (not lr / sync / episodes), so **use a new tag whenever a hyperparameter changes**; finished (arm, seed) pairs are skipped on rerun. Training seeds are limited to 0-6. Exact arguments for the `...` scripts and for the Hawkes arms: `STAGE3_REVISIT_HANDOFF.md`, `STAGE3_HANDOFF.md`.

### Running the live dashboard (two terminals, venv active)

```
cd backend ; pip install fastapi uvicorn pydantic ; uvicorn api_server:app --port 8000   # backend, port 8000
python frontend/serve.py                                                                # static server on port 5500 (set PORT to change), from the project root
```
Open `http://localhost:5500/dashboard.html`, press New episode (`serve.py` also opens the offline demo at `http://localhost:5500/`). `api_server.py` allows CORS for local development only (tighten before any deployment).

### Stage 4 (run from `validation/`)

```
python run_stage4_benchmarks.py --n 200 --agent-logs outputs\stage3c\logs\logs_ensemble_valsel.pkl --outdir outputs\stage4_gaussian
python run_stage4_benchmarks.py --n 200 --obs-kind spread_hawkes --agent-logs outputs\stage3c\logs\logs_ensemble_valsel_spread_hawkes.pkl --outdir outputs\stage4_ens_hawkes
python run_stage4_shap.py [--obs-kind spread_hawkes] [--n N]
python test_bl_consistency.py ; python test_ce_misspec.py
python check_agent_policies.py          # Stage 4C hosting check; expect ALL CHECKS PASSED
```
Always pass `--outdir` per run: files from an earlier run were overwritten once. Gaussian and Hawkes agent logs must be scored with the matching `--obs-kind`.

## Stage 2 in short (details: `STAGE2_HANDOFF.md`)

`IMMParticleFilter` runs one particle bank per regime, updates regime probabilities from each bank's marginal likelihood, and mixes banks through the transition matrix. It implements the `Estimator` protocol and returns a K = 2 `BeliefState`. Full-mode validation (200 episodes per arm): 10/10 HARD checks pass. Gaussian arm: alpha RMSE 2.5e-6 (bootstrap 1.8e-5), 90% coverage 0.94, regime accuracy 0.984. Spread-Hawkes arm: alpha RMSE 1.0e-5 (bootstrap 2.1e-5) and regime accuracy 0.949, but **the filter is overconfident (alpha 90% coverage 0.54)** because its Gaussian likelihood is biased there (mean proxy z up to 1.7 in stressed steps). Raising `obs_noise_inflation` to 2-3 lifts coverage to 0.79-0.88 in the sweep (INFO only; the default 1.0 is kept). In the forced-stress test the Hawkes filter raised a false alarm before the window in 18% of episodes (Gaussian: 0%).

## Stage 3 in short (details: `STAGE3_HANDOFF.md`, `STAGE3_REVISIT_HANDOFF.md`)

`BeliefMDPEnv` turns execution into a belief-MDP: the agent picks one of 7 multiples (0.25-2.0) of the remaining-TWAP rate each step; reward is per-step drift-only impact cost in bps with the cost TWAP would have paid subtracted. Eight arms: `oracle`, `raw_proxy`, `filt_full`, `filt_mean_only`, `filt_no_regime`, `filt_single`, plus `filt_full_bonus` / `filt_mean_bonus` (exploration bonus tied to the filter's posterior std). Agent: numpy Double DQN. 5 training seeds, 30000 episodes. Ablations (a), (b), (c) of the brief map onto these arms.

Paired difference vs TWAP in $ (negative = better). Single = mean of the 5 validation-selected agents (primary metric, test worlds); ensemble = mean Q of those 5 agents. Last column: paired gap of the ensemble to the certainty-equivalent rule on the same 7-point grid (`ce_filtered_grid`, which uses the TRUE state), first 200 fresh worlds.

| Arm | single, test | ensemble, test | ensemble, 500 fresh | ensemble minus CE grid (t) |
|---|---|---|---|---|
| `filt_full_bonus` | -73.3 | -87.9 | -93.7 | +1.2 (0.46) |
| `oracle` | -67.7 | -82.2 | -89.2 | +4.7 (1.15) |
| `filt_full` | -61.0 | -76.9 | -85.3 | +9.6 (4.41) |
| `filt_mean_only` | -63.4 | -73.2 | -82.4 | +12.8 (3.04) |
| `filt_mean_bonus` | -56.2 | -67.7 | -79.0 | +16.9 (4.86) |
| `filt_single` | -50.1 | -69.3 | -73.0 | +21.7 (4.13) |
| `filt_no_regime` | -49.0 | -57.9 | -54.6 | +41.3 (9.09) |
| `raw_proxy` | -9.7 | -45.9 | -50.6 | +46.6 (9.06) |

Main findings:
- The only ablation that survives multiple-comparison correction is **filtered belief vs raw proxy** (-51.3 $ test, -58.7 $ fresh, seed-level). Uncertainty features, the exploration bonus and the regime layer have no established effect.
- Final (last) checkpoints never beat TWAP; CVaR95 is worse than TWAP for every arm.
- A finer action grid gave a null result (default grid kept). Ensembling beats the average member in every arm. The best ensembles (`filt_full_bonus`, `oracle`) are statistically indistinguishable from CE on the same grid; the others trail it. Do not claim they beat CE.
- Spread-Hawkes (exploratory, 4 arms: oracle, raw_proxy, filt_full, filt_mean_only): single agents -31.5 to -60.9 $, ensembles -61.8 to -77.8 $ (test); all four ensembles trail CE by +14 to +28 $. Single agents re-scored on 500 fresh worlds (5-seed mean, validation-selected): oracle -66.9, filt_full -64.5, filt_mean_only -48.0, raw_proxy -36.0 $.

## Stage 4 in short (details: `STAGE4_HANDOFF.md`)

- **Benchmarks** (200 fresh worlds, TWAP mean IS 4993.1): CE beats TWAP by about 2% (continuous -99.6 $, grid -92.2 $) and is the strong reference. The heuristic is not distinguishable from TWAP and its mean is unstable across world sets. Barger-Lorig (first-order) loses to TWAP (+337 $); `test_bl_consistency.py` shows this comes from using the expansion outside its range, not a coding error. Describe it as a first-order approximation, not "the benchmark".
- **CE misspecification, first look:** errors in theta can remove up to about two thirds of CE's gain in some states; errors in lam barely matter. True-parameter CE is an optimistic reference.
- **SHAP (4B):** exact Shapley attribution of the filter's one-step update. Spread and volume dominate the alpha posterior; spread leads regime detection. This describes how the filter reacts, not what the agents use.
- **Benchmark outputs:** the Gaussian run (`obs=gaussian`, 200 fresh worlds, seeds 1000000..1000199) is in `benchmarks.md` / `benchmarks.json` and reproduces the numbers above (TWAP 4993.1; CE -99.6 $ continuous, -92.2 $ grid, noise-free; Barger-Lorig +336.9 $). Keep the Hawkes run in its own folder.
- **Agent hosting (4C):** `backend/agent_policies.py` hosts agents and ensembles behind the `Policy` protocol; `check_agent_policies.py` passes ALL CHECKS (single agents and ensembles on 5 Gaussian arms and 4 Hawkes arms: live observations, trade paths and costs equal the environment's exactly; live noise-free ensemble cost equals the saved score to <= 2.7e-12 for 4 checked ensembles; information barrier F1 holds; error cases E1-E7 pass). The stressed-regime band guard is optional, off unless requested, and exploratory: it clips the multiplier to [0.75, 1.25] when stressed (0 violations in 42 stressed steps); the circuit-breaker part did not work. `backend/guards.py` (Tier 1, wrappers around any policy, no retraining) describes it as: if a guarded policy's mark-to-market cost exceeds the hosted TWAP's by more than `breaker_usd`, switch it to the TWAP rate for the rest of the episode (latched). By the code's own caveat it caps further deterioration, not the loss already taken, and it reacts one step late. Both guards are off by default (`breaker_usd=None`, `constrain_in_stress=False`). The only "addendum" found is Section 10 of `STAGE0_INTERFACES.md`; it records that `check_guards.py` passes (guards apply only to non-TWAP policies, check G3) but does not say why the breaker "did not work". Since the guard checks pass, the phrase in `live_session.py` most likely means it did not help in the offline study rather than that the code is broken, but that reading is unconfirmed (TBD). `validation/check_guards.py` (checks U, G1-G5) tests the breaker mechanics on a deliberately aggressive synthetic policy (3x the TWAP rate), not on trained agents: it checks the latch, the switch to the TWAP rate, isolation of unguarded policies, rollback and error handling. Run on 1 Nov 2026: ALL CHECKS PASSED, 23 checks (log `validation/outputs/terminal_logs/15_guards.txt`; its docstring still says "not run by its author", which is outdated). In G1 the tripped policy's cost fell from 5139.15 to 5009.65 against TWAP at 5025.55. In G2 the aggressive policy is always clipped to the upper edge (multiplier 1.25), so the lower edge is covered only by the unit check U5. Whether the breaker helps trained agents is not tested by it.

## Honesty notes that apply project-wide

- All simulator, regime and Hawkes parameters are illustrative, not calibrated to market data.
- The filter and the CE rule use the simulator's true parameters by default; misspecified runs are sensitivity tests, not the headline.
- Stage 3: the primary metric (validation-selected checkpoint) was fixed after the oracle's final-checkpoint metric failed, so it is a post-hoc choice; both metrics are reported. Test worlds were reused to choose configurations, hence the fresh-world confirmation. With 3-5 seeds only differences of roughly 20-30 $ are detectable.
- The ensemble idea arose after results were seen; nothing was tuned, but it is not pre-registered and ensemble contrasts are exploratory. `filt_full_bonus` was singled out as best after seeing results. Ensembles cost 5x at inference.
- Hawkes seeds 3-4 were added after the 3-seed results were seen. Hawkes has no K = 1, no-regime or bonus arms, and no seed-level contrasts.
- CE is a reference, not the optimum (non-causal ceiling about -194 $). Ensemble tail risk (CVaR95) is worse than TWAP for every arm.
- Do not mix 200-world and 500-world numbers: the first 200 fresh worlds are 4-7 $ harder for every policy.
- The exploration bonus is a heuristic (beliefs do not depend on the agent's trades); only beta = 1.0 was run.
- Particle count in live hosting: agents were trained and scored at 2000 particles per regime (`LiveEpisode` default). `check_agent_policies.py` runs its live-vs-environment checks at 500 for speed and its saved-score checks at 2000. The dashboard sends no particle count; the episode started on 1 Nov 2026 showed no particle note after "Episode ... ready.", which means the backend ran at the training default (2000).
- Stage 0 contracts: per `STAGE0_INTERFACES.md` Section 10.5, Decisions 2, 3, 4 and 6 (filter knows the true regime model, K>1 ESS definition, Python/FastAPI backend, `PolicyInput.obs`) are implemented, checked and ratified (by the project owner, DEUS Group, 1 Nov 2026; individual approvers are not named); Decision 1 (v1 Hawkes look-ahead) is unchanged, so v1 is excluded from the ablations; Decision 5 (stationary initial regime probabilities) is what Stages 2 and 3 use.
- Anything marked provisional in a handoff document is provisional.

## Stage 5 / dashboard

Two pages, both front-ends for the same project:

- `frontend/index.html` (with `style.css`, `engine.js`, `app.js`): the milestone-1 stand-alone demo. It runs entirely in the browser: a single-regime Gaussian simulator, a 5000-particle bootstrap filter, and the belief-informed heuristic vs TWAP vs oracle (same rule with the true alpha). It shows the particle cloud for the hidden impact alpha, mid-price, inventory, trade rate and posterior std, with a Run / New episode control, speed selector and an injectable liquidity shock (x2-x5, the validated range). It uses the heuristic only, no RL agents; a header link goes to `dashboard.html`.
- `frontend/dashboard.html` (uses `model.js` for response validation and history): the Stage 4C live-backend version (Hawkes-aware). It calls the backend at `http://localhost:8000` from a page served on port 5500, so the backend must allow CORS, and it hosts the trained agents and ensembles through `backend/agent_policies.py`.

Controls of `dashboard.html`: seed (default 900000, the first test world), observation arm (gaussian / spread_hawkes; the agent list switches to that arm's defaults), comma-separated agent policies (always alongside TWAP, `ce_filtered` and `heuristic_belief`), Step / Run, speed, liquidity shock (x2-x5, one per episode) and a forced stressed regime. It plots the per-regime particle cloud (200 sampled particles) and per-policy costs. The page does not send a particle count, so the backend's default applies; the note shown after "Episode ... ready." reports any deviation from the training setting.

Frontend tests (not run by the author of this README): `node frontend/model.test.js` (data layer) and `node frontend/parity_check.js` (JS engine vs the Python experiment numbers).