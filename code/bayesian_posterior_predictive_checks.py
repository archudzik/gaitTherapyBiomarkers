"""Posterior predictive checks for the Bayesian endpoint models.

This script refits the same 12 Bayesian hierarchical endpoint models used in
the paper and samples posterior predictive replicated outcomes.
It summarizes whether replicated outcomes reproduce the observed overall,
therapy-state, task, and patient-level mean/spread patterns.

Default outputs are written under `reproduced/posterior_predictive_checks/`.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from bayesian_hierarchical_models import (
    ENDPOINTS,
    MAIN_PREDICTORS,
    TASK_PREDICTORS,
    prepare_data,
    stable_model_seed,
)


@dataclass(frozen=True)
class PPCModelSpec:
    endpoint: str
    endpoint_type: str
    metric_variant: str
    source_column: str
    model_name: str
    predictors: list[str]


def model_specs() -> list[PPCModelSpec]:
    specs: list[PPCModelSpec] = []
    for endpoint, meta in ENDPOINTS.items():
        for metric_variant, source_column in [
            ("primary_self_match_only", meta["primary"]),
            ("theiler_excluded", meta["theiler"]),
        ]:
            specs.append(
                PPCModelSpec(
                    endpoint=endpoint,
                    endpoint_type=meta["type"],
                    metric_variant=metric_variant,
                    source_column=source_column,
                    model_name="main_effects",
                    predictors=MAIN_PREDICTORS,
                )
            )
        specs.append(
            PPCModelSpec(
                endpoint=endpoint,
                endpoint_type=meta["type"],
                metric_variant="primary_self_match_only",
                source_column=meta["primary"],
                model_name="task_interaction",
                predictors=TASK_PREDICTORS,
            )
        )
    return specs


def fit_and_sample_ppc(
    data: pd.DataFrame,
    spec: PPCModelSpec,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, az.InferenceData, float, float]:
    outcome = f"z_{spec.source_column}"
    model_df = data[["patient_id", "task", "session", outcome, *spec.predictors]].dropna().copy()
    patients = sorted(model_df["patient_id"].unique())
    patient_index = {patient: idx for idx, patient in enumerate(patients)}
    patient_codes = model_df["patient_id"].map(patient_index).to_numpy(dtype=int)
    x = model_df[spec.predictors].to_numpy(dtype=float)
    y = model_df[outcome].to_numpy(dtype=float)

    coords = {
        "obs": np.arange(len(model_df)),
        "patient": patients,
        "predictor": spec.predictors,
    }
    with pm.Model(coords=coords):
        patient_data = pm.Data("patient_idx", patient_codes, dims="obs")
        x_data = pm.Data("x", x, dims=("obs", "predictor"))
        alpha_group = pm.Normal("alpha_group", 0, 1)
        beta = pm.Normal("beta", 0, 0.5, dims="predictor")
        sigma = pm.HalfNormal("sigma", 1)
        sigma_patient = pm.HalfNormal("sigma_patient", 0.5)
        alpha_patient = pm.Normal("alpha_patient", alpha_group, sigma_patient, dims="patient")
        mu = alpha_patient[patient_data] + pm.math.dot(x_data, beta)
        pm.Normal("y", mu, sigma, observed=y, dims="obs")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
        )
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y"],
            random_seed=seed + 1,
            progressbar=False,
            return_inferencedata=False,
        )

    y_rep = np.asarray(ppc["y"], dtype=float).reshape(-1, len(model_df))
    mean = float(data[spec.source_column].mean())
    sd = float(data[spec.source_column].std(ddof=0))
    return model_df, y_rep, idata, mean, sd


def two_sided_ppc_pvalue(replicated: np.ndarray, observed: float) -> float:
    if len(replicated) == 0 or not np.isfinite(observed):
        return np.nan
    p_upper = float(np.mean(replicated >= observed))
    p_lower = float(np.mean(replicated <= observed))
    return min(1.0, 2.0 * min(p_upper, p_lower))


def summarize_stat(
    spec: PPCModelSpec,
    check_scope: str,
    group_label: str,
    stat_name: str,
    observed: float,
    replicated: np.ndarray,
    n_observed: int,
) -> dict[str, object]:
    return {
        "model": spec.model_name,
        "endpoint": spec.endpoint,
        "endpoint_type": spec.endpoint_type,
        "metric_variant": spec.metric_variant,
        "source_column": spec.source_column,
        "check_scope": check_scope,
        "group": group_label,
        "statistic": stat_name,
        "n_observed": n_observed,
        "observed": observed,
        "ppc_mean": float(np.mean(replicated)) if len(replicated) else np.nan,
        "ppc_ci_2.5": float(np.quantile(replicated, 0.025)) if len(replicated) else np.nan,
        "ppc_ci_97.5": float(np.quantile(replicated, 0.975)) if len(replicated) else np.nan,
        "ppc_tail_p_two_sided": two_sided_ppc_pvalue(replicated, observed),
        "observed_inside_95pct_ppc": bool(
            len(replicated)
            and np.quantile(replicated, 0.025) <= observed <= np.quantile(replicated, 0.975)
        ),
    }


def stat_rows_for_subset(
    spec: PPCModelSpec,
    model_df: pd.DataFrame,
    y_rep: np.ndarray,
    mask: np.ndarray,
    check_scope: str,
    group_label: str,
) -> list[dict[str, object]]:
    y = model_df.loc[mask, f"z_{spec.source_column}"].to_numpy(dtype=float)
    reps = y_rep[:, mask]
    if len(y) < 2:
        return []
    return [
        summarize_stat(
            spec,
            check_scope,
            group_label,
            "mean_z",
            float(np.mean(y)),
            reps.mean(axis=1),
            len(y),
        ),
        summarize_stat(
            spec,
            check_scope,
            group_label,
            "sd_z",
            float(np.std(y, ddof=0)),
            reps.std(axis=1, ddof=0),
            len(y),
        ),
    ]


def summarize_ppc(
    spec: PPCModelSpec,
    model_df: pd.DataFrame,
    y_rep: np.ndarray,
    idata: az.InferenceData,
    runtime_seconds: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_mask = np.ones(len(model_df), dtype=bool)
    rows.extend(stat_rows_for_subset(spec, model_df, y_rep, all_mask, "overall", "all"))

    for task, sub in model_df.groupby("task", dropna=False):
        mask = model_df.index.isin(sub.index)
        rows.extend(stat_rows_for_subset(spec, model_df, y_rep, mask, "task", str(task)))

    for session, sub in model_df.groupby("session", dropna=False):
        mask = model_df.index.isin(sub.index)
        rows.extend(
            stat_rows_for_subset(spec, model_df, y_rep, mask, "therapy_state", str(session))
        )

    for (task, session), sub in model_df.groupby(["task", "session"], dropna=False):
        mask = model_df.index.isin(sub.index)
        rows.extend(
            stat_rows_for_subset(
                spec,
                model_df,
                y_rep,
                mask,
                "task_x_therapy_state",
                f"{task}|{session}",
            )
        )

    patient_means = model_df.groupby("patient_id", dropna=False)[f"z_{spec.source_column}"].mean()
    patient_rep_means = []
    for _, sub in model_df.groupby("patient_id", dropna=False):
        mask = model_df.index.isin(sub.index)
        patient_rep_means.append(y_rep[:, mask].mean(axis=1))
    patient_rep_means = np.column_stack(patient_rep_means)
    rows.append(
        summarize_stat(
            spec,
            "patient_level",
            "patient_mean_spread",
            "sd_patient_means_z",
            float(patient_means.std(ddof=0)),
            patient_rep_means.std(axis=1, ddof=0),
            len(patient_means),
        )
    )

    summary = az.summary(idata, var_names=["beta"], round_to=None)
    diagnostic = {
        "model": spec.model_name,
        "endpoint": spec.endpoint,
        "metric_variant": spec.metric_variant,
        "source_column": spec.source_column,
        "n_rows": len(model_df),
        "n_patients": model_df["patient_id"].nunique(),
        "n_ppc_draws": y_rep.shape[0],
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "max_r_hat": float(pd.to_numeric(summary["r_hat"], errors="coerce").max()),
        "min_ess_bulk": float(pd.to_numeric(summary["ess_bulk"], errors="coerce").min()),
        "divergences": int(idata.sample_stats["diverging"].sum().values),
    }
    return rows, diagnostic


def write_report(results: pd.DataFrame, diagnostics: pd.DataFrame, output_dir: Path) -> None:
    finite = results.dropna(subset=["ppc_tail_p_two_sided"])
    flagged = finite[finite["ppc_tail_p_two_sided"] < 0.05].copy()
    outside = finite[~finite["observed_inside_95pct_ppc"]].copy()
    lines = [
        "# Bayesian Posterior Predictive Checks",
        "",
        "Posterior predictive replicated outcomes were generated for the 12 Bayesian endpoint models.",
        "Checks compare observed standardized endpoint means and spreads with replicated distributions overall, by task, by therapy state, by task x therapy state, and for patient-level mean spread.",
        "",
        "## Summary",
        f"- PPC rows: {len(results)}",
        f"- Models: {diagnostics[['model', 'endpoint', 'metric_variant']].drop_duplicates().shape[0]}",
        f"- Rows with two-sided PPC tail p < 0.05: {len(flagged)}",
        f"- Rows where observed statistic is outside the 95% replicated interval: {len(outside)}",
        f"- Max R-hat: {diagnostics['max_r_hat'].max():.2f}",
        f"- Min bulk ESS: {diagnostics['min_ess_bulk'].min():.0f}",
        f"- Total divergences: {int(diagnostics['divergences'].sum())}",
        "",
        "## Most Extreme PPC Rows",
    ]
    if flagged.empty:
        lines.append("none")
    else:
        for _, row in flagged.sort_values("ppc_tail_p_two_sided").head(12).iterrows():
            lines.append(
                f"- {row['model']} {row['endpoint']} {row['metric_variant']} "
                f"{row['check_scope']} {row['group']} {row['statistic']}: "
                f"observed={row['observed']:.3f}, PPC 95%=[{row['ppc_ci_2.5']:.3f}, "
                f"{row['ppc_ci_97.5']:.3f}], ppc_p={row['ppc_tail_p_two_sided']:.3f}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "Use PPCs as model-adequacy checks rather than new discovery tests. If no systematic misfit appears in the key endpoint/task/therapy summaries, report them as supporting the adequacy of the simple hierarchical Gaussian model for the observed summaries. Any isolated low-tail checks should be described conservatively.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/derived_features.csv"),
        help="Released participant-session-task feature table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduced/posterior_predictive_checks"),
        help="Directory for posterior-predictive tables and diagnostics.",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning draws per chain.")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args.dataset.resolve())

    all_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    specs = model_specs()
    if args.max_models is not None:
        specs = specs[: args.max_models]

    for index, spec in enumerate(specs, start=1):
        seed = stable_model_seed(args.seed, spec.endpoint, spec.metric_variant, spec.model_name)
        print(
            f"[{index:02d}/{len(specs):02d}] {spec.model_name} "
            f"{spec.endpoint} {spec.metric_variant}"
        )
        started = time.perf_counter()
        model_df, y_rep, idata, _, _ = fit_and_sample_ppc(
            data=data,
            spec=spec,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            seed=seed,
        )
        rows, diagnostics = summarize_ppc(
            spec=spec,
            model_df=model_df,
            y_rep=y_rep,
            idata=idata,
            runtime_seconds=time.perf_counter() - started,
            seed=seed,
        )
        all_rows.extend(rows)
        diagnostic_rows.append(diagnostics)

    results = pd.DataFrame(all_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    results.to_csv(output_dir / "posterior_predictive_check_results.csv", index=False)
    diagnostics.to_csv(output_dir / "posterior_predictive_check_diagnostics.csv", index=False)
    write_report(results, diagnostics, output_dir)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
