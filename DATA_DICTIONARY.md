# Data dictionary

All public participant identifiers are release-specific pseudonyms of the form `P001`. No public mapping to the working identifiers is retained.

## `data/derived_features.csv`

One row represents a participant, therapy session, and walking task. The identifying columns are `patient_id`, `session_code`, `session`, `med`, `stim`, and `task`. The remaining motion-derived columns contain distribution summaries of short-window local-divergence profiles, bilateral asymmetry, coupling, trial counts, duration, and window counts. The five retained UPDRS-III columns are the total score and the leg-agility, postural-stability, and gait items used by the released validation analyses. Exact assessment dates, age, sex, and unused clinical items are excluded.

## `data/cohort_summary.csv`

Three aggregate-only rows report the cohort's male and female counts and the age distribution used in the descriptive statistics. Individual-level age and sex remain excluded. The `endpoint` column identifies the statistic; the remaining columns contain the applicable count and numeric summary fields.

## `data/gait_metrics.csv`

One row represents the same participant-session-task unit. Columns report cadence, walking speed, step and stride timing, step and stride length, variability, bilateral asymmetry, and event counts aggregated from the available motion-capture gait events.

## `data/parameter_sensitivity/*.csv`

These files use the same schema as `derived_features.csv`, once for the baseline and each one-factor parameter perturbation reported in the paper.

## `data/construct_validation/*.csv`

De-identified trial-level summaries used for conventional Rosenstein comparison and phase-randomized surrogate checks. They contain derived numeric summaries only, not marker trajectories or file paths.

## `results/`

Machine-readable tables underlying the paper and appendix. This directory includes descriptive and paired statistics, Bayesian summaries and diagnostics, clinical and standard-gait validation, leave-one-participant-out analyses, construct validation, parameter sensitivity, and posterior predictive checks.
