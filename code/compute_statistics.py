"""Article-level statistics for the processed PD short-window LLE dataset.

The script produces one flat CSV file with the statistical results intended for
the Results section of the manuscript. Analyses are kept deliberately compact:
cohort/UPDRS description, LLE asymmetry and coupling summaries, paired session
contrasts, and within-subject 2 x 2 MED/STIM contrasts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONFIG = {
    "session_order": ("S01", "S02", "S03", "S04"),
    "session_labels": {
        "S01": "MED OFF / STIM OFF",
        "S02": "MED OFF / STIM ON",
        "S03": "MED ON / STIM OFF",
        "S04": "MED ON / STIM ON",
    },
    "paired_contrasts": {
        "OFF_OFF_vs_ON_ON": ("S01", "S04"),
        "OFF_OFF_vs_OFF_ON": ("S01", "S02"),
        "OFF_OFF_vs_ON_OFF": ("S01", "S03"),
        "OFF_ON_vs_ON_ON": ("S02", "S04"),
        "ON_OFF_vs_ON_ON": ("S03", "S04"),
        "OFF_ON_vs_ON_OFF": ("S02", "S03"),
    },
    "endpoints": {
        "overall_lle_asymmetry": {
            "type": "lle_asymmetry",
            "variants": {
                "primary_self_match_only": "abs_overall_asymmetry_mean",
                "theiler_excluded": "abs_overall_asymmetry_theiler_mean",
            },
        },
        "heel_lle_asymmetry": {
            "type": "lle_asymmetry",
            "variants": {
                "primary_self_match_only": "abs_heel_asymmetry_mean",
                "theiler_excluded": "abs_heel_asymmetry_theiler_mean",
            },
        },
        "toe_lle_asymmetry": {
            "type": "lle_asymmetry",
            "variants": {
                "primary_self_match_only": "abs_toe_asymmetry_mean",
                "theiler_excluded": "abs_toe_asymmetry_theiler_mean",
            },
        },
        "left_right_lle_coupling": {
            "type": "lle_coupling",
            "variants": {
                "primary_self_match_only": "left_right_mean_lle_correlation",
                "theiler_excluded": "left_right_mean_lle_correlation_theiler",
            },
        },
        "heel_lle_coupling": {
            "type": "lle_coupling",
            "variants": {
                "primary_self_match_only": "LHEE_RHEE_lle_correlation",
                "theiler_excluded": "LHEE_RHEE_lle_correlation_theiler",
            },
        },
        "toe_lle_coupling": {
            "type": "lle_coupling",
            "variants": {
                "primary_self_match_only": "LTOE_RTOE_lle_correlation",
                "theiler_excluded": "LTOE_RTOE_lle_correlation_theiler",
            },
        },
    },
}


RESULT_COLUMNS = [
    "section",
    "result_role",
    "result_order",
    "is_key_result",
    "metric_variant",
    "task",
    "endpoint",
    "endpoint_type",
    "session_code",
    "session_label",
    "contrast",
    "effect",
    "condition_a",
    "condition_b",
    "n",
    "n_pairs",
    "n_complete",
    "mean",
    "sd",
    "median",
    "iqr",
    "min",
    "max",
    "mean_a",
    "mean_b",
    "mean_diff_b_minus_a",
    "median_diff_b_minus_a",
    "mean_effect",
    "median_effect",
    "p_value",
    "p_fdr_section",
    "effect_size_dz",
    "notes",
]


KEY_RESULT_RULES = [
    {
        "result_role": "primary_endpoint",
        "result_order": 1,
        "section": "paired_session_contrast",
        "metric_variant": "primary_self_match_only",
        "task": "normal_walking",
        "endpoint": "overall_lle_asymmetry",
        "contrast": "OFF_OFF_vs_ON_ON",
    },
    {
        "result_role": "secondary_clinical_endpoint",
        "result_order": 2,
        "section": "paired_session_contrast",
        "metric_variant": "primary_self_match_only",
        "task": "normal_walking",
        "endpoint": "overall_lle_asymmetry",
        "contrast": "OFF_OFF_vs_ON_OFF",
    },
    {
        "result_role": "mechanistic_endpoint",
        "result_order": 3,
        "section": "factorial_med_stim_effect",
        "metric_variant": "theiler_excluded",
        "task": "normal_walking",
        "endpoint": "toe_lle_coupling",
        "effect": "STIM_main_effect",
    },
    {
        "result_role": "sensitivity_analysis",
        "result_order": 4,
        "section": "paired_session_contrast",
        "metric_variant": "theiler_excluded",
        "task": "normal_walking",
        "endpoint": "overall_lle_asymmetry",
        "contrast": "OFF_OFF_vs_ON_ON",
    },
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "session_code", "task", "med", "stim", "updrs3_total"}
    for meta in CONFIG["endpoints"].values():
        for column in meta["variants"].values():
            required.add(column)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    return df


def numeric_summary(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "median": float(clean.median()),
        "iqr": float(clean.quantile(0.75) - clean.quantile(0.25)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def wilcoxon_from_diff(diff: pd.Series) -> float:
    clean = diff.dropna().astype(float)
    clean = clean[np.abs(clean) > 1e-12]
    if len(clean) < 3:
        return np.nan
    try:
        return float(stats.wilcoxon(clean, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def cohen_dz(diff: pd.Series) -> float:
    clean = diff.dropna().astype(float)
    if len(clean) < 2:
        return np.nan
    sd = clean.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(clean.mean() / sd)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted[i] = running
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def base_row(section: str, **values: object) -> dict[str, object]:
    row = {col: np.nan for col in RESULT_COLUMNS}
    row["section"] = section
    row.update(values)
    return row


def load_cohort_summary(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Cohort summary not found: {path}")
    summary = pd.read_csv(path)
    required = {"endpoint", "n", "mean", "sd", "median", "iqr", "min", "max"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing cohort-summary columns: {missing}")
    allowed = {"sex_male_n", "sex_female_n", "age_years"}
    unexpected = sorted(set(summary["endpoint"].dropna()) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected cohort-summary endpoints: {unexpected}")
    return summary


def cohort_rows(
    df: pd.DataFrame, aggregate_summary: pd.DataFrame | None = None
) -> list[dict[str, object]]:
    patients = df.drop_duplicates("patient_id")
    rows = [
        base_row(
            "cohort_summary",
            endpoint="patients_n",
            n=int(patients["patient_id"].nunique()),
        )
    ]
    if "sex" in patients:
        rows.extend(
            [
                base_row(
                    "cohort_summary",
                    endpoint="sex_male_n",
                    n=int((patients["sex"] == "M").sum()),
                ),
                base_row(
                    "cohort_summary",
                    endpoint="sex_female_n",
                    n=int((patients["sex"] == "F").sum()),
                ),
            ]
        )
    if "age" in patients:
        age = numeric_summary(patients["age"])
        rows.append(base_row("cohort_summary", endpoint="age_years", **age))
    if aggregate_summary is not None:
        existing = {row["endpoint"] for row in rows}
        for _, item in aggregate_summary.iterrows():
            if item["endpoint"] in existing:
                continue
            values = {
                column: item[column]
                for column in ("n", "mean", "sd", "median", "iqr", "min", "max")
                if pd.notna(item[column])
            }
            rows.append(
                base_row(
                    "cohort_summary",
                    endpoint=item["endpoint"],
                    **values,
                )
            )

    availability = (
        df.groupby(["task", "session_code"]).agg(n=("patient_id", "nunique")).reset_index()
    )
    for _, item in availability.iterrows():
        rows.append(
            base_row(
                "data_availability",
                task=item["task"],
                session_code=item["session_code"],
                session_label=CONFIG["session_labels"].get(item["session_code"], ""),
                endpoint="patient_sessions_with_mocap",
                n=int(item["n"]),
            )
        )
    return rows


def updrs_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    unique_sessions = df.drop_duplicates(["patient_id", "session_code"])
    for session in CONFIG["session_order"]:
        sub = unique_sessions[unique_sessions["session_code"] == session]
        rows.append(
            base_row(
                "updrs3_summary",
                endpoint="updrs3_total",
                session_code=session,
                session_label=CONFIG["session_labels"][session],
                **numeric_summary(sub["updrs3_total"]),
            )
        )
    return rows


def endpoint_summary_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        for session in CONFIG["session_order"]:
            sub = task_df[task_df["session_code"] == session]
            for endpoint, meta in CONFIG["endpoints"].items():
                for variant, column in meta["variants"].items():
                    rows.append(
                        base_row(
                            "endpoint_session_summary",
                            metric_variant=variant,
                            task=task,
                            endpoint=endpoint,
                            endpoint_type=meta["type"],
                            session_code=session,
                            session_label=CONFIG["session_labels"][session],
                            **numeric_summary(sub[column]),
                        )
                    )
    return rows


def paired_contrast_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        for endpoint, meta in CONFIG["endpoints"].items():
            for variant, column in meta["variants"].items():
                pivot = task_df.pivot_table(
                    index="patient_id",
                    columns="session_code",
                    values=column,
                    aggfunc="mean",
                )
                for contrast, (session_a, session_b) in CONFIG["paired_contrasts"].items():
                    pair = pivot[[session_a, session_b]].dropna()
                    values_a = pair[session_a]
                    values_b = pair[session_b]
                    diff = values_b - values_a
                    rows.append(
                        base_row(
                            "paired_session_contrast",
                            metric_variant=variant,
                            task=task,
                            endpoint=endpoint,
                            endpoint_type=meta["type"],
                            contrast=contrast,
                            condition_a=CONFIG["session_labels"][session_a],
                            condition_b=CONFIG["session_labels"][session_b],
                            n_pairs=int(len(pair)),
                            mean_a=float(values_a.mean()) if len(pair) else np.nan,
                            mean_b=float(values_b.mean()) if len(pair) else np.nan,
                            mean_diff_b_minus_a=float(diff.mean()) if len(pair) else np.nan,
                            median_diff_b_minus_a=float(diff.median()) if len(pair) else np.nan,
                            p_value=wilcoxon_from_diff(diff),
                            effect_size_dz=cohen_dz(diff),
                        )
                    )
    return rows


def factorial_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        for endpoint, meta in CONFIG["endpoints"].items():
            for variant, column in meta["variants"].items():
                pivot = task_df.pivot_table(
                    index="patient_id",
                    columns="session_code",
                    values=column,
                    aggfunc="mean",
                )
                complete = pivot[list(CONFIG["session_order"])].dropna()
                if complete.empty:
                    continue
                effects = {
                    "MED_main_effect": ((complete["S03"] + complete["S04"]) / 2)
                    - ((complete["S01"] + complete["S02"]) / 2),
                    "STIM_main_effect": ((complete["S02"] + complete["S04"]) / 2)
                    - ((complete["S01"] + complete["S03"]) / 2),
                    "MED_x_STIM_interaction": (complete["S04"] - complete["S03"])
                    - (complete["S02"] - complete["S01"]),
                }
                for effect, values in effects.items():
                    rows.append(
                        base_row(
                            "factorial_med_stim_effect",
                            metric_variant=variant,
                            task=task,
                            endpoint=endpoint,
                            endpoint_type=meta["type"],
                            effect=effect,
                            n_complete=int(len(values)),
                            mean_effect=float(values.mean()),
                            median_effect=float(values.median()),
                            p_value=wilcoxon_from_diff(values),
                            effect_size_dz=cohen_dz(values),
                        )
                    )
    return rows


def add_section_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["p_fdr_section"] = np.nan
    group_cols = ["section", "task", "metric_variant"]
    for _, index in results.groupby(group_cols, dropna=False).groups.items():
        results.loc[index, "p_fdr_section"] = bh_fdr(results.loc[index, "p_value"])
    return results


def add_result_roles(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["result_role"] = ""
    results["result_order"] = np.nan
    results["is_key_result"] = False

    for rule in KEY_RESULT_RULES:
        mask = pd.Series(True, index=results.index)
        for column, value in rule.items():
            if column in {"result_role", "result_order"}:
                continue
            mask &= results[column].eq(value)
        results.loc[mask, "result_role"] = rule["result_role"]
        results.loc[mask, "result_order"] = rule["result_order"]
        results.loc[mask, "is_key_result"] = True

    return results


def build_article_results(
    df: pd.DataFrame, cohort_summary: pd.DataFrame | None = None
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(cohort_rows(df, cohort_summary))
    rows.extend(updrs_rows(df))
    rows.extend(endpoint_summary_rows(df))
    rows.extend(paired_contrast_rows(df))
    rows.extend(factorial_rows(df))
    results = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    results = results[RESULT_COLUMNS]
    results = add_section_fdr(results)
    results = add_result_roles(results)
    return results[RESULT_COLUMNS]


def run(dataset_path: Path, output_path: Path, cohort_summary_path: Path | None = None) -> None:
    df = load_dataset(dataset_path)
    cohort_summary = load_cohort_summary(cohort_summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = build_article_results(df, cohort_summary)
    results.to_csv(output_path, index=False)
    print("Done.")
    print(f"Input rows: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create article-level statistics CSV.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/derived_features.csv"),
        help="Released participant-session-task feature table.",
    )
    parser.add_argument(
        "--cohort-summary",
        type=Path,
        default=Path("data/cohort_summary.csv"),
        help="Aggregate cohort demographics released separately for privacy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproduced/article_results.csv"),
        help="Destination for the flat article-results table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.dataset.resolve(),
        args.output.resolve(),
        args.cohort_summary.resolve(),
    )


if __name__ == "__main__":
    main()
