# EDI -- Optimal trade execution under latent liquidity

Research project extending Macri & Lillo (2025, *Applied Mathematical Finance*): optimal execution when market liquidity is latent, estimated by sequential Monte Carlo, and acted on by a reinforcement-learning agent.

> **Draft README.** It covers the layout and Stages 0-3. The consolidated project synopsis (all four module synopses) is Person 4's final deliverable and will replace it. Current status per stage: `STATUS_SUMMARY.md`.

## Pipeline

```
Simulator (Almgren-Chriss + CIR impact, Markov regimes)      Stage 1  src/simulator.py, src/regime_simulator.py
   -> Observation model (Gaussian | Hawkes | spread-Hawkes)  Stage 1  src/observation_model.py, hawkes_observation.py, spread_hawkes.py
   -> Filter (bootstrap PF | IMM-PF)                         Stage 2  src/particle_filter.py, src/imm_particle_filter.py
   -> Decision layer (belief-MDP, Double DQN)                Stage 3  src/belief_mdp_env.py, env_factory.py, ddqn.py, ddqn_train{,2,3,4}.py,
                                                                      stage3c_bonus.py, belief_cache.py, stage3_adapter.py
   -> Evaluation + backend API                               Stage 4  (not in this repository yet)
   -> Dashboard                                              Stage 5  frontend/ (milestone-1 version, single-regime Gaussian only)
```
All module boundaries are defined in `src/interfaces.py` and documented in `STAGE0_INTERFACES.md` (draft awaiting team ratification).

## Layout

| Path | What |
|---|---|
| `src/` | model code (see pipeline above), `interfaces.py` shared contracts, `belief_metrics.py` filter-agnostic belief scoring, Stage 3 environment / agent / training loops / cached-belief environment / Stage 0 log adapter |
| `validation/` | validation and experiment scripts; Stage 1-2 outputs go to `validation/outputs/`, Stage 3 outputs to `validation/outputs/stage3c/` (`results_base*.jsonl`, `ckpt/`, `logs/`, report `.md` / `.json`) and `validation/outputs/belief_cache/` |
| `frontend/` | milestone-1 client-side dashboard (`serve.py` only serves static files) |
| `STAGE0_INTERFACES.md`, `STAGE1_HANDOFF.md`, `STAGE2_HANDOFF.md`, `STAGE3_HANDOFF.md` | per-stage handoff documents, each ending in a HANDOFF SYNOPSIS (`STAGE3_HANDOFF.md` is still the interim version) |
| `summary_stage3.md` | Person 3's running summary of Stage 3 (what is built, results so far, what is left) |
| `STATUS_SUMMARY.md` | what is done, what is provisional, what is not started |

## Setup and running

```
pip install -r requirements.txt          # numpy, scipy, matplotlib, pandas
python validation/validate_filter.py     # milestone-1 bootstrap filter validation
python validation/validate_interfaces.py # Stage 0 conformance (20 checks)
python validation/validate_stage1.py     # Stage 1: regimes + Hawkes (add --quick for a smoke test)
python validation/validate_spread_hawkes.py
python validation/validate_imm_pf.py     # Stage 2: IMM-PF (add --quick for a smoke test first)
```
Scripts add `src/` to the path themselves; run them from the project root. Modules in `src/` use flat imports, so run their `__main__` demos (e.g. `stage3c_bonus.py`, `stage3_adapter.py` self-tests) from inside `src/`.

### Stage 3 (run from `validation/`; PowerShell shown, bash is identical)

```
# 1. (filtered arms only, optional but ~19x faster) build the belief cache with the REAL filter; resumable
python build_belief_cache.py --filter imm    --what val test train --train-seed 0
python build_belief_cache.py --filter single --what val test train --train-seed 0     # repeat per training seed

# 2. train + evaluate arms (5 training seeds, 30000 episodes, 200 test worlds); saves agents to outputs/stage3c/ckpt/
python run_stage3c_multiseed.py --arms oracle,raw_proxy --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --tag _abl
python run_stage3c_multiseed.py --arms filt_full,filt_mean_only,filt_no_regime,filt_single --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --use-cache --tag _abl
python run_stage3c_multiseed.py --arms filt_full_bonus,filt_mean_bonus --seeds 5 --eval-episodes 200 --target-sync 2000 --n-val 300 --episodes 30000 --use-cache --bonus-beta 1.0 --tag _abl_b1

# 3. reports
python compare_arms.py --tags _abl _abl_b1                      # per-arm vs TWAP + paired contrasts (a), (b), (c)
python score_fresh.py --tags _abl _abl_b1 --n-fresh 500 --use-cache   # confirm on worlds never used for any choice
python export_agent_logs.py --tags _abl _abl_b1 --which valsel --n 200  # interfaces.EpisodeLog pickle for Person 4
```
`--tag` keys the results file (not lr / sync / episodes), so **use a new tag whenever a hyperparameter changes**; finished (arm, seed) pairs are skipped on rerun. Training-world seeds are `100000*(seed+1)+episode`, so `--seeds` is limited to 7 (training seeds 0-6).

## Stage 2 in short (details: `STAGE2_HANDOFF.md`)

`IMMParticleFilter` runs one particle bank per regime, updates regime probabilities from each bank's marginal likelihood, and mixes banks through the transition matrix. It implements the `Estimator` protocol and returns a K = 2 `BeliefState` (filtered regime probabilities, per-regime and pooled moments, combined ESS). It is validated against the untouched bootstrap filter on the same observations, with a regime-detection accuracy report. Full validation run: 10/10 HARD checks pass; on the Gaussian arm alpha RMSE is 2.5e-6 (bootstrap 1.8e-5), 90% coverage 0.94, regime accuracy 0.984. On the spread-Hawkes arm the filter is overconfident (alpha 90% coverage 0.54) because the Gaussian likelihood is biased there. Numbers and caveats: `STAGE2_HANDOFF.md` section (b), `validation/outputs/stage2_results.md`.

## Stage 3 in short (details: `summary_stage3.md`, `STAGE3_HANDOFF.md`)

`BeliefMDPEnv` turns trade execution into a belief-MDP: the agent picks one of 7 multiples (0.25-2.0) of the remaining-TWAP rate each step (1.0 = TWAP; the last step liquidates the rest); the reward is the per-step drift-only impact cost in bps of initial value with the cost TWAP would have paid in that step subtracted (an action-independent baseline). Observation arms: `oracle` (true alpha, kappa, regime), `raw_proxy`, and filtered arms fed by the Stage 2 filter (`filt_full`, `filt_mean_only` = no posterior std, `filt_no_regime` = regime probabilities hidden, `filt_single` = K = 1 filter), plus `filt_full_bonus` / `filt_mean_bonus` with an exploration bonus tied to the filter's posterior std. The agent is a numpy Double DQN (van Hasselt et al., 2016). Ablations (a), (b), (c) of the brief map onto these arms; `compare_arms.py` computes the paired contrasts.

Results so far (200 test worlds, paired difference vs TWAP in $, negative = better; 30000 episodes; **primary metric = validation-selected checkpoint scored on the test worlds**):

| Arm | training seeds | validation-selected | final checkpoint |
|---|---|---|---|
| `oracle` | 5 | -67.7 (95% CI half-width 9.0): better than TWAP | +4.5: not distinguishable from TWAP |
| `raw_proxy` | 5 | -9.7 (CI half-width 32.6): not distinguishable from TWAP | +72.4: worse than TWAP |
| `filt_full` | 1 | -59.7 (one seed, no test possible) | -9.9 |

**No ablation contrast has a result yet.** The remaining filtered and bonus arms are pending (see `STATUS_SUMMARY.md`), and `compare_arms.py` refuses contrasts with fewer than 3 seeds. The numbers above come from `compare_arms.py` output seen by the author of this README; the other Stage 3 checks listed in `summary_stage3.md` (environment, DQN, cached-environment and adapter self-tests) are Person 3's report.

## Honesty notes that apply project-wide

- All simulator, regime and Hawkes parameters are illustrative, not calibrated to market data.
- The filter is given the true regime model by default (Stage 0 Decision 2); misspecified runs are sensitivity tests, not the headline.
- Stage 3: the primary metric (validation-selected checkpoint) was fixed after the oracle's final-checkpoint metric failed, so it is a post-hoc choice; both metrics are reported for every arm. Test worlds were reused to choose configurations (target sync, episodes, EMA), so headline numbers should be confirmed on fresh worlds (`score_fresh.py`). With 3-5 training seeds, only differences of roughly 20-30 $ between arms are detectable. Gaussian observations only; the exploration bonus is a heuristic (the belief does not depend on the agent's trades, so there is no information to gain by acting).
- Anything marked provisional in a handoff document is provisional.