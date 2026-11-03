"""LLE parameter robustness analysis.

The active paper uses one fixed short-window LLE configuration. This script
reruns the natural-walking analysis under a small set of nearby parameter
choices and reports whether the key therapy-response findings keep the same
direction and similar magnitude.

Default output unit:
    one row per parameter configuration and key result
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from compute_statistics import build_article_results
from prepare_dataset import MARKERS, DatasetProcessor, ProcessingConfig

BASE_EMBEDDING = {marker: {"d": 3, "tau": 25} for marker in MARKERS}

ROBUSTNESS_CONFIGS = (
    {
        "label": "baseline",
        "window_seconds": 1.0,
        "smoothing_window_length": 11,
        "smoothing_polyorder": 3,
        "embedding": BASE_EMBEDDING,
    },
    {
        "label": "window_0p8s",
        "window_seconds": 0.8,
        "smoothing_window_length": 11,
        "smoothing_polyorder": 3,
        "embedding": BASE_EMBEDDING,
    },
    {
        "label": "window_1p2s",
        "window_seconds": 1.2,
        "smoothing_window_length": 11,
        "smoothing_polyorder": 3,
        "embedding": BASE_EMBEDDING,
    },
    {
        "label": "embedding_d2_tau25",
        "window_seconds": 1.0,
        "smoothing_window_length": 11,
        "smoothing_polyorder": 3,
        "embedding": {marker: {"d": 2, "tau": 25} for marker in MARKERS},
    },
    {
        "label": "embedding_d3_tau20",
        "window_seconds": 1.0,
        "smoothing_window_length": 11,
        "smoothing_polyorder": 3,
        "embedding": {marker: {"d": 3, "tau": 20} for marker in MARKERS},
    },
    {
        "label": "embedding_d3_tau30",
        "window_seconds": 1.0,
        "smoothing_window_length": 11,
        "smoothing_polyorder": 3,
        "embedding": {marker: {"d": 3, "tau": 30} for marker in MARKERS},
    },
    {
        "label": "smooth_7",
        "window_seconds": 1.0,
        "smoothing_window_length": 7,
        "smoothing_polyorder": 3,
        "embedding": BASE_EMBEDDING,
    },
    {
        "label": "smooth_15",
        "window_seconds": 1.0,
        "smoothing_window_length": 15,
        "smoothing_polyorder": 3,
        "embedding": BASE_EMBEDDING,
    },
)


KEY_RESULTS = (
    {
        "result_name": "overall_asymmetry_combined_therapy",
        "section": "paired_session_contrast",
        "metric_variant": "primary_self_match_only",
        "endpoint": "overall_lle_asymmetry",
        "contrast": "OFF_OFF_vs_ON_ON",
        "effect": "",
        "expected_direction": -1,
    },
    {
        "result_name": "overall_asymmetry_medication_only",
        "section": "paired_session_contrast",
        "metric_variant": "primary_self_match_only",
        "endpoint": "overall_lle_asymmetry",
        "contrast": "OFF_OFF_vs_ON_OFF",
        "effect": "",
        "expected_direction": -1,
    },
    {
        "result_name": "toe_coupling_combined_therapy",
        "section": "paired_session_contrast",
        "metric_variant": "primary_self_match_only",
        "endpoint": "toe_lle_coupling",
        "contrast": "OFF_OFF_vs_ON_ON",
        "effect": "",
        "expected_direction": 1,
    },
    {
        "result_name": "toe_coupling_stim_factorial_theiler",
        "section": "factorial_med_stim_effect",
        "metric_variant": "theiler_excluded",
        "endpoint": "toe_lle_coupling",
        "contrast": "",
        "effect": "STIM_main_effect",
        "expected_direction": 1,
    },
)


@dataclass(frozen=True)
class RobustnessRun:
    """One natural-walking LLE parameter configuration."""

    label: str
    window_seconds: float
    smoothing_window_length: int
    smoothing_polyorder: int
    embedding: dict[str, dict[str, int]]

    @property
    def output_label(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.label)


class LLERobustnessAnalysis:
    """Run and summarize a compact LLE parameter robustness grid."""

    def __init__(
        self,
        data_dir: Path,
        updrs_path: Path,
        output_dir: Path,
        force: bool = False,
        max_configs: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.updrs_path = updrs_path
        self.output_dir = output_dir
        self.force = force
        self.max_configs = max_configs

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        configs = [RobustnessRun(**config) for config in ROBUSTNESS_CONFIGS]
        if self.max_configs is not None:
            configs = configs[: self.max_configs]

        rows = []
        config_rows = []
        for config in configs:
            print(f"Running robustness configuration: {config.label}")
            dataset_path = self._dataset_for_config(config)
            dataset = pd.read_csv(dataset_path)
            article_results = build_article_results(dataset)
            article_results.to_csv(
                self.output_dir / f"article_results_{config.output_label}.csv",
                index=False,
            )
            rows.extend(self._key_rows(config, article_results))
            config_rows.append(self._config_row(config, dataset))

        summary = pd.DataFrame(rows)
        configs_df = pd.DataFrame(config_rows)
        stability = self._stability_summary(summary)
        summary.to_csv(self.output_dir / "lle_parameter_robustness_results.csv", index=False)
        configs_df.to_csv(self.output_dir / "lle_parameter_robustness_configs.csv", index=False)
        stability.to_csv(self.output_dir / "lle_parameter_robustness_stability.csv", index=False)
        self._write_report(stability)

        print("Done.")
        print(f"Configurations: {len(configs)}")
        print(f"Output: {self.output_dir}")

    def _dataset_for_config(self, config: RobustnessRun) -> Path:
        dataset_dir = self.output_dir / "datasets" / config.output_label
        dataset_path = dataset_dir / "dataset.csv"
        if dataset_path.exists() and not self.force:
            return dataset_path
        processor = DatasetProcessor(
            data_dir=self.data_dir,
            updrs_path=self.updrs_path,
            output_dir=dataset_dir,
            tasks=["normal"],
            config=ProcessingConfig(
                window_seconds=config.window_seconds,
                step_seconds=0.1,
                theiler_seconds=0.1,
                smoothing_window_length=config.smoothing_window_length,
                smoothing_polyorder=config.smoothing_polyorder,
                embedding_params=config.embedding,
            ),
            quiet=True,
        )
        processor.run()
        return dataset_path

    def _key_rows(
        self, config: RobustnessRun, article_results: pd.DataFrame
    ) -> list[dict[str, object]]:
        rows = []
        for rule in KEY_RESULTS:
            match = article_results[
                article_results["section"].eq(rule["section"])
                & article_results["metric_variant"].eq(rule["metric_variant"])
                & article_results["task"].eq("normal_walking")
                & article_results["endpoint"].eq(rule["endpoint"])
            ].copy()
            if rule["contrast"]:
                match = match[match["contrast"].eq(rule["contrast"])]
            if rule["effect"]:
                match = match[match["effect"].eq(rule["effect"])]
            if match.empty:
                rows.append(self._missing_key_row(config, rule))
                continue
            row = match.iloc[0]
            estimate = row["mean_diff_b_minus_a"]
            if pd.isna(estimate):
                estimate = row["mean_effect"]
            rows.append(
                {
                    "config_label": config.label,
                    "result_name": rule["result_name"],
                    "section": rule["section"],
                    "metric_variant": rule["metric_variant"],
                    "endpoint": rule["endpoint"],
                    "contrast": rule["contrast"],
                    "effect": rule["effect"],
                    "expected_direction": rule["expected_direction"],
                    "n_pairs": row["n_pairs"],
                    "n_complete": row["n_complete"],
                    "estimate": estimate,
                    "p_value": row["p_value"],
                    "p_fdr_section": row["p_fdr_section"],
                    "effect_size_dz": row["effect_size_dz"],
                    "direction_matches_expected": direction_matches(
                        estimate,
                        rule["expected_direction"],
                    ),
                    "window_seconds": config.window_seconds,
                    "smoothing_window_length": config.smoothing_window_length,
                    "smoothing_polyorder": config.smoothing_polyorder,
                    "embedding_json": json.dumps(config.embedding, sort_keys=True),
                }
            )
        return rows

    def _missing_key_row(
        self,
        config: RobustnessRun,
        rule: dict[str, object],
    ) -> dict[str, object]:
        return {
            "config_label": config.label,
            "result_name": rule["result_name"],
            "section": rule["section"],
            "metric_variant": rule["metric_variant"],
            "endpoint": rule["endpoint"],
            "contrast": rule["contrast"],
            "effect": rule["effect"],
            "expected_direction": rule["expected_direction"],
            "n_pairs": np.nan,
            "n_complete": np.nan,
            "estimate": np.nan,
            "p_value": np.nan,
            "p_fdr_section": np.nan,
            "effect_size_dz": np.nan,
            "direction_matches_expected": False,
            "window_seconds": config.window_seconds,
            "smoothing_window_length": config.smoothing_window_length,
            "smoothing_polyorder": config.smoothing_polyorder,
            "embedding_json": json.dumps(config.embedding, sort_keys=True),
        }

    def _config_row(self, config: RobustnessRun, dataset: pd.DataFrame) -> dict[str, object]:
        return {
            "config_label": config.label,
            "window_seconds": config.window_seconds,
            "smoothing_window_length": config.smoothing_window_length,
            "smoothing_polyorder": config.smoothing_polyorder,
            "embedding_json": json.dumps(config.embedding, sort_keys=True),
            "dataset_rows": len(dataset),
            "patients": dataset["patient_id"].nunique(),
            "task": "normal_walking",
        }

    def _stability_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for result_name, sub in results.groupby("result_name", dropna=False):
            finite = sub.dropna(subset=["estimate"])
            rows.append(
                {
                    "result_name": result_name,
                    "n_configs": len(sub),
                    "n_finite": len(finite),
                    "expected_direction": finite["expected_direction"].iloc[0]
                    if len(finite)
                    else np.nan,
                    "direction_match_fraction": float(finite["direction_matches_expected"].mean())
                    if len(finite)
                    else np.nan,
                    "estimate_min": float(finite["estimate"].min()) if len(finite) else np.nan,
                    "estimate_max": float(finite["estimate"].max()) if len(finite) else np.nan,
                    "effect_size_dz_min": float(finite["effect_size_dz"].min())
                    if len(finite)
                    else np.nan,
                    "effect_size_dz_max": float(finite["effect_size_dz"].max())
                    if len(finite)
                    else np.nan,
                    "nominal_p_lt_0p05_fraction": float((finite["p_value"] < 0.05).mean())
                    if len(finite)
                    else np.nan,
                    "fdr_q_lt_0p10_fraction": float((finite["p_fdr_section"] < 0.10).mean())
                    if len(finite)
                    else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _write_report(self, stability: pd.DataFrame) -> None:
        lines = [
            "LLE parameter robustness summary",
            "",
        ]
        for _, row in stability.iterrows():
            lines.append(
                f"- {row['result_name']}: direction match "
                f"{row['direction_match_fraction']:.0%} across {int(row['n_finite'])} configs; "
                f"estimate range [{row['estimate_min']:.4f}, {row['estimate_max']:.4f}], "
                f"nominal p<0.05 in {row['nominal_p_lt_0p05_fraction']:.0%}"
            )
        (self.output_dir / "lle_parameter_robustness_summary.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def direction_matches(estimate: object, expected_direction: int) -> bool:
    try:
        value = float(estimate)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value) or abs(value) < 1e-12:
        return False
    return bool(np.sign(value) == np.sign(expected_direction))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact LLE parameter robustness analyses.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--updrs", type=Path, default=Path("data/updrs.csv"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/validation/lle_parameter_robustness"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = LLERobustnessAnalysis(
        data_dir=args.data_dir.resolve(),
        updrs_path=args.updrs.resolve(),
        output_dir=args.out_dir.resolve(),
        force=args.force,
        max_configs=args.max_configs,
    )
    analysis.run()


if __name__ == "__main__":
    main()
