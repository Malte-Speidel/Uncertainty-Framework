# Uncertainty-Framework

Research framework for evolutionary multi-objective optimization (EMO) under noisy/uncertain objective evaluations, built on top of [pymoo](https://pymoo.org/). It provides noisy variants of standard EMO benchmark problems and a collection of algorithm adaptations that handle evaluation noise through confidence bounds, probabilistic/confidence-based dominance, co-evolution, or resampling.

## Requirements

- **Python ≥ 3.10** — the noise-sampling code (`_sample_noise` in `src/problems/`) uses `match`/`case` statements, which set the practical floor. Developed and tested against Python 3.13.
- Dependencies (`requirements.txt`): `pymoo`, `numpy`, `pandas`, `matplotlib`, `plotly`, `numba`, `pyarrow`, `fastparquet`.
- Not currently pinned in `requirements.txt` but imported directly: `scipy` (used by the probabilistic-dominance and MOCBA adaptations) and `streamlit` (used by the analysis dashboards in `src/analysis/`).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Noisy benchmark problems (`src/problems/`)

Each problem module defines a shared `_sample_noise(noise_type, noise_std, shape)` helper producing zero-mean noise of a given kind (`gaussian`, `poisson`, `uniform`, `rademacher` — each parameterized so its variance equals `noise_std**2`), and a family of classes that wrap the equivalent pymoo problem plus `LoggingMixin` (see below). Each class overrides `_evaluate` to add sampled noise on top of the clean objective values, and adds an `evaluate_noiseless(X)` method that returns the underlying clean `F` values (used by callbacks to track true convergence against the real Pareto front).

- `noisy_zdt.py` — `NZDT1`–`NZDT6`, noisy versions of ZDT1–ZDT6.
- `noisy_wfg.py` — `NWFG1`–`NWFG9`, noisy versions of WFG1–WFG9.
- `noisy_dtlz.py` — `NDTLZ1`, `NDTLZ2`, `NDTLZ3`, `NDTLZ7`, noisy versions of the corresponding DTLZ problems (`NDTLZ7` also recomputes its Pareto front analytically since noise-aware fronts don't come from pymoo directly).

## Algorithms (`src/algorithms/`)

All algorithms wrap the corresponding pymoo algorithm class and mix in `LoggingMixin`.

### Confidence-bound (UCB1) adaptations — `cb_adaptations/`

Common pattern across this family: every individual accumulates a running mean estimate (`f_sum`, `n_evals`) across generations, and the current rank-0 (non-dominated) front is re-evaluated once more each generation to shrink its uncertainty (`_initialize_advance`/`_advance` overrides). Before survival selection runs, a custom `*_Survival` class temporarily overwrites each individual's `F` with a lower-confidence-bound value `mean + c * sqrt(2*ln(T)/n_evals)`, where `T` is the total evaluation count among rank-0 individuals and `c` is a tunable exploration constant, then delegates to the wrapped algorithm's normal selection logic.

- `CB_NSGA2` (`cb_nsga2.py`) — wraps `NSGA2`; `CB_Survival` subclasses `RankAndCrowding`.
- `cb_nsga3` (`cb_nsga3.py`) — wraps `NSGA3`; builds Das-Dennis reference directions sized to the objective count via `_das_dennis_ref_dirs` (reused by the CBD/PD NSGA-III variants); its `CB_Survival` subclasses `ReferenceDirectionSurvival`.
- `CB_SMSEMOA` (`cb_sms_emoa.py`) — wraps `SMSEMOA`; `CB_Survival` subclasses `LeastHypervolumeContributionSurvival`.
- `CB_SPEA2` (`cb_spea2.py`) — wraps `SPEA2`; `CB_Survival` subclasses `SPEA2Survival` and keys off SPEA2's own `SPEA_R` rank attribute instead of `rank`.
- `cb_agemoea` (`age_moea.py`) — wraps `AGEMOEA`; `CB_AGEMOEASurvival` subclasses `AGEMOEASurvival`.

### Confidence-Based Dominance — `cbd_adaptations/`

`CBD_NSGA2`/`CBD_NSGA3` (`cbd_nsga2.py`, `cbd_nsga3.py`) replace Pareto dominance with a degree of confidence-interval dominance (Boonma & Suzuki 2009; Karshenas et al. 2015). `_cbd_domination_matrix` builds an N×N dominance matrix from each individual's running mean and a symmetric confidence interval of half-width `z·σ/√n`; one solution dominates another only once its CI-adjusted advantage clears a threshold `alpha` on every objective. `CBD_Survival`/`CBD_Survival_NSGA3` derive fronts from that matrix via `_fronts_from_dom_matrix` (shared with the PD adaptations, defined in `pd_nsga2.py`) instead of pymoo's standard non-dominated sorting, then hand off to the usual crowding (NSGA-II) or niching (NSGA-III) logic. Rank-0 survivors are re-evaluated every generation, as in the CB adaptations.

### Probabilistic Dominance — `pd_adaptations/`

`PD_NSGA2`/`PD_NSGA3` (`pd_nsga2.py`, `pd_nsga3.py`) use a softer notion of dominance: `_prob_domination_matrix` computes, for every pair of individuals and every objective, the probability (via the normal CDF over the standard error of the mean difference) that one is truly better than the other, and treats it as dominant once that probability exceeds `alpha` on all objectives simultaneously. `_fronts_from_dom_matrix` then peels fronts off the resulting matrix, analogous to standard non-dominated sorting.

### Co-evolution — `co_evolution/`

`DualEvolution` (`co_nsga2.py`) is a standalone driver — not a pymoo `Algorithm` subclass — that runs two independently configured algorithm instances against the same problem, splitting population size and generation/evaluation budget between them. After each joint generation it compares the hypervolume of both non-dominated fronts and swaps individuals from the weaker algorithm's population with the hypervolume-contribution leaders of the stronger one (`_swap_individuals_hv`), so weak performers get replaced by fitter migrants from the sibling run. Results from both sub-runs are merged into a single result dict in `_create_res`.

### Resampling adaptations — `resampling_adaptations/`

Rather than changing dominance or survival, these decide *how many times* to re-evaluate a noisy individual.

- `RTEA` (`rtea.py`) — a from-scratch `Algorithm` subclass implementing Fieldsend's Rolling Tide Evolutionary Algorithm. Maintains a non-dominated **archive** separate from the working population; each generation produces one offspring via SBX crossover plus `SeededGaussianMutation` (a `GaussianMutation` patched to draw its bounds-repair randomness from the algorithm's own seeded `random_state`, since pymoo's built-in version doesn't and breaks reproducibility), tests it against the archive (`_update_front`), and spends a small budget resampling the least-sampled archive member each generation (`_resample_archive`). Dominance dependencies between archive and population members are tracked in `self._dependents` so that demoting an archive member requeues its dependents for re-checking.
- `MOCBA_NSGA2` (`mocba_nsga2.py`) — NSGA-II combined with Multi-Objective Optimal Computing Budget Allocation: dominated individuals receive a resampling budget proportional to their probability of misclassification against their closest non-dominated competitor (`_closest_non_dom_by_risk`), rounded via largest-remainder rounding (`largest_remainder_round`) so the per-generation budget is hit exactly.
- `CBR_NSGA2` (`cbr_nsga2.py`) — NSGA-II combined with `CB_Resampling`'s threshold-based re-evaluation (see below).
- `SR_NSGA2` (`static_res_nsga2.py`) — a static-resampling baseline: every individual is evaluated a fixed `k` times and tracked via a running mean.

## Supporting modules

- `src/resampling/cb_resampling.py` — `CB_Resampling`, a stateless (`staticmethod`-only) helper used by `CBR_NSGA2` and related adaptations. `_prepare_infills` sets up per-individual running-mean bookkeeping; `resample()` chooses between a dynamically derived threshold (`_calc_threshold`, averaged from the population's per-individual suggested thresholds) or a fixed one, then `_threshold_resampling` re-evaluates individuals whose uncertainty indicator (based on a `_knn` Minkowski-distance neighbourhood estimate of `T`) exceeds that threshold.
- `src/callbacks/cbr_callback.py`, `src/callbacks/rtea_callback.py` — `CBR_Callback` and `RTEA_Callback`, pymoo `Callback` subclasses invoked once per generation to record convergence and re-evaluation metrics (running means, evaluation counts, clean/noiseless fronts, IGD+, GD+), with a `save_data` method to pickle the collected data. `RTEA_Callback` reads from the algorithm's `_archive` rather than `pop`, since RTEA keeps its Pareto approximation there.
- `config/logging_mixin.py` — `LoggingMixin`, a mixin giving every subclass its own named `logging.Logger` via `__init_subclass__`; used throughout the problem and algorithm classes.
- `config/logging_config.py` — `setup_logging()`, a one-shot `logging.config.dictConfig` setup that logs to console and to a file.
- `config/parameters.py`, `config/old_params/cb_fitness_parameters.py` — tuned hyper-parameter tables (resampling thresholds, MOCBA budgets, static-resampling `k`, CB/CBD/PD exploration constants) per problem and noise level, produced by prior parameter sweeps and consumed by the scripts in `experiments/`.
- `src/analysis/main_analysis.py`, `src/analysis/tau_sweep_analysis.py` — Streamlit + Matplotlib dashboards for exploring pickled experiment results (hypervolume, IGD+, GD+ comparisons across algorithms and sweep parameters).
- `experiments/` — driver scripts (`cbr_main_experiments.py`, `cbr_sweeps.py`) that wire problems, algorithms, parameters, and callbacks together into runnable experiment batches.
- `tests/` — lightweight sanity checks for the resampling adaptations, `CB_Resampling`, and RTEA, plus `import_tests.py`, a smoke test that every algorithm/problem module still imports cleanly.

## Usage

```bash
cd src
python main.py
```

Experiment batches can be run directly, e.g.:

```bash
python experiments/cbr_main_experiments.py
```
