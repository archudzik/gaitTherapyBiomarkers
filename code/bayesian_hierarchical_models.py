"""Bayesian models for short-window LLE gait biomarkers.

This script fits a compact, hypothesis-driven Bayesian hierarchical analysis
for Parkinson's disease gait dynamics. It produces one CSV table
and one figure showing model-predicted marker trajectories across medication
and DBS stimulation states.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

THERAPY_STATES = [
    ("OFF/OFF", 0, 0),
    ("OFF/ON", 0, 1),
    ("ON/OFF", 1, 0),
    ("ON/ON", 1, 1),
]

TASKS = [("normal_walking", 0), ("fast_walking", 1)]

ENDPOINTS = {
    "overall_lle_asymmetry": {
        "type": "asymmetry",
        "primary": "abs_overall_asymmetry_mean",
        "theiler": "abs_overall_asymmetry_theiler_mean",
    },
    "heel_lle_asymmetry": {
        "type": "asymmetry",
        "primary": "abs_heel_asymmetry_mean",
        "theiler": "abs_heel_asymmetry_theiler_mean",
    },
    "overall_lle_coupling": {
        "type": "coupling",
        "primary": "left_right_mean_lle_correlation",
        "theiler": "left_right_mean_lle_correlation_theiler",
    },
    "toe_lle_coupling": {
        "type": "coupling",
        "primary": "LTOE_RTOE_lle_correlation",
        "theiler": "LTOE_RTOE_lle_correlation_theiler",
    },
}

ENDPOINT_LABELS = {
    "overall_lle_asymmetry": "Overall LLE asymmetry",
    "heel_lle_asymmetry": "Heel LLE asymmetry",
    "overall_lle_coupling": "Overall LLE coupling",
    "toe_lle_coupling": "Toe LLE coupling",
}

MAIN_PREDICTORS = ["MED", "STIM", "MEDxSTIM", "fast_task"]
TASK_PREDICTORS = [
    "MED",
    "STIM",
    "MEDxSTIM",
    "fast_task",
    "MEDxfast_task",
    "STIMxfast_task",
]


@dataclass
class ModelFit:
    endpoint: str
    endpoint_type: str
    metric_variant: str
    source_column: str
    model_name: str
    predictors: list[str]
    patients: list[str]
    samples: pd.DataFrame
    diagnostics: pd.DataFrame
    divergences: int
    n_rows: int
    outcome_mean: float
    outcome_sd: float


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def total_ram_gb() -> float | None:
    try:
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / (1024**3)
    except (AttributeError, OSError, ValueError):
        return None


def cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
    except (ModuleNotFoundError, OSError):
        return platform.processor() or platform.machine()


def system_report() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "processor": cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": total_ram_gb(),
    }


def stable_model_seed(base_seed: int, *parts: str) -> int:
    label = "::".join([str(base_seed), *parts])
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - values.mean()) / sd


def zscore_params(values: pd.Series) -> tuple[float, float]:
    values = values.astype(float)
    return float(values.mean()), float(values.std(ddof=0))


def prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "med", "stim", "task"}
    for meta in ENDPOINTS.values():
        required.add(meta["primary"])
        required.add(meta["theiler"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["MED"] = df["med"].astype(str).str.upper().eq("ON").astype(int)
    df["STIM"] = df["stim"].astype(str).str.upper().eq("ON").astype(int)
    df["MEDxSTIM"] = df["MED"] * df["STIM"]
    df["fast_task"] = df["task"].astype(str).str.lower().str.contains("fast").astype(int)
    df["MEDxfast_task"] = df["MED"] * df["fast_task"]
    df["STIMxfast_task"] = df["STIM"] * df["fast_task"]

    for meta in ENDPOINTS.values():
        for column in (meta["primary"], meta["theiler"]):
            df[f"z_{column}"] = zscore(df[column])
    return df


def fit_hierarchical_normal(
    data: pd.DataFrame,
    endpoint: str,
    endpoint_type: str,
    metric_variant: str,
    source_column: str,
    predictors: list[str],
    model_name: str,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> ModelFit:
    outcome = f"z_{source_column}"
    model_df = data[["patient_id", outcome, *predictors]].dropna().copy()
    patients = sorted(model_df["patient_id"].unique())
    patient_index = {patient: idx for idx, patient in enumerate(patients)}
    patient_codes = model_df["patient_id"].map(patient_index).to_numpy(dtype=int)
    x = model_df[predictors].to_numpy(dtype=float)
    y = model_df[outcome].to_numpy(dtype=float)

    coords = {
        "obs": np.arange(len(model_df)),
        "patient": patients,
        "predictor": predictors,
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

    posterior = idata.posterior
    beta_draws = (
        posterior["beta"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "predictor")
        .to_pandas()
    )
    beta_draws.columns = [f"beta_{name}" for name in predictors]
    samples = pd.DataFrame({"alpha_group": posterior["alpha_group"].values.reshape(-1)})
    samples = pd.concat([samples, beta_draws.reset_index(drop=True)], axis=1)
    samples["log_sigma"] = np.log(posterior["sigma"].values.reshape(-1))
    samples["log_sigma_patient"] = np.log(posterior["sigma_patient"].values.reshape(-1))

    diagnostics = az.summary(idata, var_names=["beta"], round_to=None)
    divergences = int(idata.sample_stats["diverging"].sum().values)

    return ModelFit(
        endpoint=endpoint,
        endpoint_type=endpoint_type,
        metric_variant=metric_variant,
        source_column=source_column,
        model_name=model_name,
        predictors=predictors,
        patients=patients,
        samples=samples,
        diagnostics=diagnostics,
        divergences=divergences,
        n_rows=len(model_df),
        outcome_mean=float(data[source_column].mean()),
        outcome_sd=float(data[source_column].std(ddof=0)),
    )


def summarize_coefficients(fit: ModelFit) -> list[dict[str, object]]:
    rows = []
    for predictor in fit.predictors:
        values = fit.samples[f"beta_{predictor}"]
        diag_name = f"beta[{predictor}]"
        diag = (
            fit.diagnostics.loc[diag_name]
            if diag_name in fit.diagnostics.index
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "section": "coefficient",
                "model": fit.model_name,
                "endpoint": fit.endpoint,
                "endpoint_type": fit.endpoint_type,
                "metric_variant": fit.metric_variant,
                "source_column": fit.source_column,
                "task": "",
                "therapy_condition": "",
                "term": predictor,
                "mean": values.mean(),
                "ci_2.5": values.quantile(0.025),
                "ci_97.5": values.quantile(0.975),
                "p_lt_0": (values < 0).mean(),
                "p_gt_0": (values > 0).mean(),
                "r_hat": diag.get("r_hat", np.nan),
                "ess_bulk": diag.get("ess_bulk", np.nan),
                "divergences": fit.divergences,
                "n_rows": fit.n_rows,
                "n_patients": len(fit.patients),
            }
        )
    return rows


def predicted_mean_draws(fit: ModelFit, med: int, stim: int, fast_task: int) -> pd.Series:
    med_stim = med * stim
    values = fit.samples["alpha_group"].copy()
    values += fit.samples.get("beta_MED", 0) * med
    values += fit.samples.get("beta_STIM", 0) * stim
    values += fit.samples.get("beta_MEDxSTIM", 0) * med_stim
    values += fit.samples.get("beta_fast_task", 0) * fast_task
    values += fit.samples.get("beta_MEDxfast_task", 0) * med * fast_task
    values += fit.samples.get("beta_STIMxfast_task", 0) * stim * fast_task
    return values


def summarize_marginal_means(fit: ModelFit) -> list[dict[str, object]]:
    rows = []
    for task, fast_task in TASKS:
        for therapy_condition, med, stim in THERAPY_STATES:
            values = predicted_mean_draws(fit, med, stim, fast_task)
            original_values = values * fit.outcome_sd + fit.outcome_mean
            rows.append(
                {
                    "section": "predicted_marginal_mean",
                    "model": fit.model_name,
                    "endpoint": fit.endpoint,
                    "endpoint_type": fit.endpoint_type,
                    "metric_variant": fit.metric_variant,
                    "source_column": fit.source_column,
                    "task": task,
                    "therapy_condition": therapy_condition,
                    "term": "",
                    "mean": original_values.mean(),
                    "ci_2.5": original_values.quantile(0.025),
                    "ci_97.5": original_values.quantile(0.975),
                    "p_lt_0": (values < 0).mean(),
                    "p_gt_0": (values > 0).mean(),
                    "r_hat": np.nan,
                    "ess_bulk": np.nan,
                    "divergences": fit.divergences,
                    "n_rows": fit.n_rows,
                    "n_patients": len(fit.patients),
                }
            )
    return rows


def plot_marker_trajectories(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary[
        (summary["section"] == "predicted_marginal_mean")
        & (summary["model"] == "task_interaction")
        & (summary["metric_variant"] == "primary_self_match_only")
    ].copy()
    endpoints = list(ENDPOINTS.keys())
    condition_order = [state[0] for state in THERAPY_STATES]
    task_style = {
        "normal_walking": {"color": "#245c73", "label": "Natural walking"},
        "fast_walking": {"color": "#9b2f2f", "label": "Fast walking"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True)
    axes = axes.flatten()
    for ax, endpoint in zip(axes, endpoints, strict=True):
        endpoint_df = plot_df[plot_df["endpoint"] == endpoint]
        x = np.arange(len(condition_order))
        for task, style in task_style.items():
            task_df = (
                endpoint_df[endpoint_df["task"] == task]
                .set_index("therapy_condition")
                .reindex(condition_order)
            )
            ax.errorbar(
                x,
                task_df["mean"],
                yerr=[
                    task_df["mean"] - task_df["ci_2.5"],
                    task_df["ci_97.5"] - task_df["mean"],
                ],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=style["color"],
                label=style["label"],
            )
        ax.axhline(0, color="0.85", linewidth=1)
        ax.set_title(ENDPOINT_LABELS[endpoint])
        ax.set_xticks(x)
        ax.set_xticklabels(condition_order)
        if ENDPOINTS[endpoint]["type"] == "asymmetry":
            ax.set_ylabel("LLE-derived slope asymmetry")
        else:
            ax.set_ylabel("Pearson correlation (r)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Model-predicted LLE-derived gait biomarkers by therapy state")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    np.random.seed(args.seed)
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_data(args.dataset)

    rows: list[dict[str, object]] = []
    task_fits: list[ModelFit] = []
    timing_rows: list[dict[str, object]] = []

    for endpoint, meta in ENDPOINTS.items():
        for metric_variant, source_column in [
            ("primary_self_match_only", meta["primary"]),
            ("theiler_excluded", meta["theiler"]),
        ]:
            fit_started_at = time.perf_counter()
            model_seed = stable_model_seed(args.seed, endpoint, metric_variant, "main_effects")
            main_fit = fit_hierarchical_normal(
                data=df,
                endpoint=endpoint,
                endpoint_type=meta["type"],
                metric_variant=metric_variant,
                source_column=source_column,
                predictors=MAIN_PREDICTORS,
                model_name="main_effects",
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                seed=model_seed,
            )
            timing_rows.append(
                {
                    "endpoint": endpoint,
                    "metric_variant": metric_variant,
                    "model": "main_effects",
                    "seed": model_seed,
                    "runtime_seconds": time.perf_counter() - fit_started_at,
                }
            )
            rows.extend(summarize_coefficients(main_fit))

        fit_started_at = time.perf_counter()
        model_seed = stable_model_seed(
            args.seed, endpoint, "primary_self_match_only", "task_interaction"
        )
        task_fit = fit_hierarchical_normal(
            data=df,
            endpoint=endpoint,
            endpoint_type=meta["type"],
            metric_variant="primary_self_match_only",
            source_column=meta["primary"],
            predictors=TASK_PREDICTORS,
            model_name="task_interaction",
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            seed=model_seed,
        )
        timing_rows.append(
            {
                "endpoint": endpoint,
                "metric_variant": "primary_self_match_only",
                "model": "task_interaction",
                "seed": model_seed,
                "runtime_seconds": time.perf_counter() - fit_started_at,
            }
        )
        task_fits.append(task_fit)
        rows.extend(summarize_coefficients(task_fit))
        rows.extend(summarize_marginal_means(task_fit))

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "bayesian_summary.csv", index=False)
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(output_dir / "runtime_by_model.csv", index=False)
    plot_marker_trajectories(summary, output_dir / "marker_trajectories.png")

    total_runtime = time.perf_counter() - started_at
    report = system_report()
    report["total_runtime_seconds"] = total_runtime
    report["draws"] = args.draws
    report["tune"] = args.tune
    report["chains"] = args.chains
    report["base_seed"] = args.seed
    report["n_models"] = len(timing_rows)

    lines = [
        "Bayesian hierarchical model compute report",
        "",
        f"Platform: {report['platform']}",
        f"Processor: {report['processor']}",
        f"Logical CPU cores: {report['cpu_logical_cores']}",
        f"RAM: {report['ram_gb']:.1f} GB" if report["ram_gb"] else "RAM: unavailable",
        f"Models fitted: {report['n_models']}",
        f"Base random seed: {args.seed}",
        f"Sampling: {args.chains} chains, {args.tune} tune + {args.draws} draws",
        f"Total runtime: {total_runtime:.1f} seconds ({total_runtime / 60:.1f} minutes)",
    ]
    (output_dir / "compute_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_dir / 'bayesian_summary.csv'}")
    print(f"Wrote {output_dir / 'runtime_by_model.csv'}")
    print(f"Wrote {output_dir / 'compute_report.txt'}")
    print(f"Wrote {output_dir / 'marker_trajectories.png'}")


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
        default=Path("reproduced/bayesian"),
        help="Directory for model summaries, diagnostics, and the trajectory figure.",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning draws per chain.")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
