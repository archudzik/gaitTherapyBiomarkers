# Supplementary materials

This package accompanies "Interpretable Machine Learning Evaluates Differential Therapy Effects in Parkinsonian Gait."

## Contents

- `code/`: feature extraction, article statistics, Bayesian models, and validation analyses.
- `data/`: de-identified derived feature tables and an aggregate-only cohort summary. Raw marker trajectories and linked clinical records are not included.
- `results/`: machine-readable outputs underlying the camera-ready paper and appendix.
- `DATA_DICTIONARY.md`: column definitions and the scope of each released table.

## Environment

The analyses were run with Python 3.12.5. Install the recorded environment with:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run commands from this directory. The Bayesian scripts use a default seed of 2026. The recorded production run used four chains, 1,000 tuning draws, and 1,000 posterior draws. Hardware and runtime details are recorded in `results/compute_report.txt`.

## Reproduce analyses from released derived data

From the package root:

```bash
python code/compute_statistics.py
python code/reproduce_validation.py
python code/bayesian_hierarchical_models.py
```

The first two commands reproduce the deterministic article and validation CSV files. Individual-level age and sex are not released; `data/cohort_summary.csv` supplies only the aggregate demographics needed to reproduce the three corresponding descriptive rows. The Bayesian command refits all reported hierarchical models and recreates the model summary, runtime report, and trajectory figure under `reproduced/bayesian/`. Use `--help` on any script to inspect inputs, outputs, and optional parameters.

Posterior predictive checks can also be rerun from the released feature table:

```bash
python code/bayesian_posterior_predictive_checks.py
```

## Code map

| Script | Purpose | Public inputs are sufficient |
| --- | --- | --- |
| `compute_statistics.py` | Descriptive, paired, and factorial statistics reported in the paper | Yes |
| `bayesian_hierarchical_models.py` | Hierarchical endpoint models and predicted therapy trajectories | Yes |
| `reproduce_validation.py` | Clinical, standard-gait, and leave-one-patient-out validation tables | Yes |
| `bayesian_posterior_predictive_checks.py` | Posterior predictive mean and spread checks | Yes |
| `prepare_dataset.py` | C3D preprocessing and short-window feature extraction | No, raw C3D files are required |
| `clinical_and_gait_validation.py` | Gait-event extraction and the complete validation workflow | No, raw C3D files are required |
| `lle_parameter_robustness.py` | Feature extraction under eight parameter settings | No, raw C3D files are required |
| `lle_estimator_construct_validation.py` | Synthetic, surrogate, and conventional-LLE comparisons | Partly; real-trial analyses require raw C3D files |

The raw-data scripts are included to document the complete method. They cannot be rerun from this package because unrestricted release of the source motion-capture trajectories and linked clinical records is not authorized. The released derived tables are sufficient for the article statistics, Bayesian models, and reported follow-up validation tables.

## Release status

This is the camera-ready release tree. Author-written code is available under the MIT License. Author-produced de-identified derived data and documentation are available under CC BY 4.0. See `LICENSE-CODE`, `LICENSE-DATA`, and `LICENSE-CONTENT` for the exact scope and exclusions.

The final supplementary materials are archived as [GitHub release 1.0.0](https://github.com/archudzik/gaitTherapyBiomarkers/releases/tag/1.0.0). The archived release has the version-specific Zenodo DOI [10.5281/zenodo.23962355](https://doi.org/10.5281/zenodo.23962355); the DOI for the archive across versions is [10.5281/zenodo.23962354](https://doi.org/10.5281/zenodo.23962354). Raw motion-capture trajectories, linked clinical source records, private documents, local reference PDFs, and third-party materials are not part of this release and are not covered by its licenses.

## License map

| Material | License |
| --- | --- |
| `code/` | MIT License (`LICENSE-CODE`) |
| `data/` and machine-readable author-produced results | CC BY 4.0 (`LICENSE-DATA`) |
| Author-produced documentation | CC BY 4.0 (`LICENSE-CONTENT`) |
| Raw participant data and third-party materials | Not distributed; no rights granted |
