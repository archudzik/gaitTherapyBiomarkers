"""Clinical and standard-gait validation for the PD LLE biomarker analysis.

The analyses include clinical-change correlations with UPDRS-III,
event-derived gait metrics from C3D events, and correlations between those
metrics and the LLE-derived biomarkers.

Default output unit:
    validation CSV tables under outputs/validation/clinical_and_gait
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
from prepare_dataset import CONFIG, TrialDiscoverer
from scipy import stats

SESSION_ORDER = tuple(CONFIG["sessions"].keys())
SESSION_LABELS = {
    code: f"MED {meta['med']} / STIM {meta['stim']}" for code, meta in CONFIG["sessions"].items()
}

VALIDATION_CONFIG = {
    "tasks": ("normal", "fast"),
    "paired_contrasts": {
        "OFF_OFF_vs_ON_ON": ("S01", "S04"),
        "OFF_OFF_vs_OFF_ON": ("S01", "S02"),
        "OFF_OFF_vs_ON_OFF": ("S01", "S03"),
    },
    "lle_endpoints": {
        "overall_lle_asymmetry_primary": "abs_overall_asymmetry_mean",
        "heel_lle_asymmetry_primary": "abs_heel_asymmetry_mean",
        "toe_lle_coupling_primary": "LTOE_RTOE_lle_correlation",
        "overall_lle_coupling_primary": "left_right_mean_lle_correlation",
        "overall_lle_asymmetry_theiler": "abs_overall_asymmetry_theiler_mean",
        "toe_lle_coupling_theiler": "LTOE_RTOE_lle_correlation_theiler",
    },
    "clinical_endpoints": {
        "updrs3_total": "updrs3_total",
        "updrs3_gait_item": "updrs3_30_gait",
        "updrs3_postural_stability": "updrs3_29_postural_stability",
        "updrs3_leg_agility_right": "updrs3_26_leg_agility_right_lower_limb",
        "updrs3_leg_agility_left": "updrs3_26_leg_agility_left_lower_limb",
    },
    "gait_metrics": {
        "cadence_steps_per_min": "cadence_steps_per_min",
        "walking_speed_mps": "walking_speed_mps",
        "step_time_mean_s": "step_time_mean_s",
        "step_time_cv": "step_time_cv",
        "stride_time_mean_s": "stride_time_mean_s",
        "stride_time_cv": "stride_time_cv",
        "stride_time_asymmetry_s": "stride_time_asymmetry_s",
        "step_length_mean_m": "step_length_mean_m",
        "stride_length_mean_m": "stride_length_mean_m",
        "stride_length_asymmetry_m": "stride_length_asymmetry_m",
    },
    "partial_controls": {
        "walking_speed_mps": "walking_speed_mps",
    },
    "focused_clinical_pairs": (
        ("overall_lle_asymmetry_primary", "updrs3_gait_item"),
        ("overall_lle_asymmetry_theiler", "updrs3_gait_item"),
        ("toe_lle_coupling_primary", "updrs3_total"),
        ("toe_lle_coupling_primary", "updrs3_postural_stability"),
        ("overall_lle_coupling_primary", "updrs3_total"),
        ("overall_lle_coupling_primary", "updrs3_postural_stability"),
    ),
    "key_lle_paired_effects": (
        ("normal_walking", "OFF_OFF_vs_ON_ON", "overall_lle_asymmetry_primary"),
        ("normal_walking", "OFF_OFF_vs_ON_OFF", "overall_lle_asymmetry_primary"),
        ("normal_walking", "OFF_OFF_vs_ON_ON", "toe_lle_coupling_primary"),
        ("normal_walking", "OFF_OFF_vs_ON_ON", "overall_lle_coupling_primary"),
        ("normal_walking", "OFF_OFF_vs_ON_ON", "overall_lle_asymmetry_theiler"),
        ("normal_walking", "OFF_OFF_vs_ON_ON", "toe_lle_coupling_theiler"),
    ),
}


RESULT_COLUMNS = [
    "analysis",
    "task",
    "contrast",
    "metric_a",
    "metric_b",
    "n",
    "estimate",
    "p_value",
    "p_fdr_task_contrast",
    "mean_delta_a",
    "mean_delta_b",
    "notes",
]


@dataclass(frozen=True)
class GaitEvent:
    """One parsed C3D gait event."""

    time_seconds: float
    context: str
    label: str

    @property
    def side(self) -> str:
        return self.context.strip().lower()

    @property
    def is_foot_strike(self) -> bool:
        return self.label.strip().lower() == "foot strike"


@dataclass(frozen=True)
class MarkerSample:
    """Marker position at a gait event."""

    time_seconds: float
    side: str
    frame_index: int
    ap_position_mm: float


class C3DGaitMetricExtractor:
    """Extract event-derived gait metrics from C3D walking trials."""

    HEEL_MARKER_BY_SIDE = {
        "left": "LHEE",
        "right": "RHEE",
    }
    PELVIS_MARKERS = ("LASI", "RASI", "LPSI", "RPSI", "CentreOfMass")

    def read_trial_metrics(self, trial) -> dict[str, object]:
        c3d = ezc3d.c3d(str(trial.path))
        labels = [label.strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
        points = np.asarray(c3d["data"]["points"], dtype=float)
        rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
        n_frames = int(points.shape[2])
        duration_seconds = n_frames / rate if rate else np.nan
        first_frame = int(c3d["header"]["points"].get("first_frame", 0))

        ap_axis = self._infer_ap_axis(points, labels)
        foot_strikes = [event for event in self._read_events(c3d) if event.is_foot_strike]
        event_samples = self._event_marker_samples(
            foot_strikes=foot_strikes,
            points=points,
            labels=labels,
            rate=rate,
            first_frame=first_frame,
            ap_axis=ap_axis,
        )
        metrics = self._summarize_event_samples(event_samples, points, labels, rate, ap_axis)
        metrics.update(
            {
                "trial_id": trial.trial_id,
                "patient_id": trial.patient_id,
                "session_code": trial.session_code,
                "session": CONFIG["sessions"][trial.session_code]["session"],
                "med": CONFIG["sessions"][trial.session_code]["med"],
                "stim": CONFIG["sessions"][trial.session_code]["stim"],
                "task": trial.task,
                "trial_path": str(trial.path),
                "sample_rate_hz": rate,
                "duration_seconds": duration_seconds,
                "first_frame": first_frame,
                "ap_axis": ap_axis,
                "n_foot_strikes": len(event_samples),
                "n_left_foot_strikes": sum(sample.side == "left" for sample in event_samples),
                "n_right_foot_strikes": sum(sample.side == "right" for sample in event_samples),
            }
        )
        return metrics

    def _read_events(self, c3d) -> list[GaitEvent]:
        event_params = c3d["parameters"].get("EVENT", {})
        used = int(np.asarray(event_params.get("USED", {}).get("value", [0]))[0])
        if used <= 0:
            return []
        contexts = list(event_params.get("CONTEXTS", {}).get("value", []))[:used]
        labels = list(event_params.get("LABELS", {}).get("value", []))[:used]
        times = np.asarray(event_params.get("TIMES", {}).get("value", []), dtype=float)
        if times.ndim != 2 or times.shape[0] < 2:
            return []
        seconds = times[0, :used] * 60.0 + times[1, :used]
        rows = [
            GaitEvent(float(time), str(context), str(label))
            for time, context, label in zip(seconds, contexts, labels, strict=False)
            if np.isfinite(time)
        ]
        return sorted(rows, key=lambda event: event.time_seconds)

    def _infer_ap_axis(self, points: np.ndarray, labels: list[str]) -> int:
        marker_ranges = []
        for marker in self.PELVIS_MARKERS:
            if marker not in labels:
                continue
            xyz = self._valid_marker_xyz(points, labels.index(marker))
            if xyz is None:
                continue
            marker_ranges.append(np.nanmax(xyz, axis=1) - np.nanmin(xyz, axis=1))
        if not marker_ranges:
            return 1
        ranges = np.nanmedian(np.vstack(marker_ranges), axis=0)
        horizontal_ranges = ranges[:2]
        return int(np.nanargmax(horizontal_ranges))

    def _event_marker_samples(
        self,
        foot_strikes: list[GaitEvent],
        points: np.ndarray,
        labels: list[str],
        rate: float,
        first_frame: int,
        ap_axis: int,
    ) -> list[MarkerSample]:
        samples = []
        for event in foot_strikes:
            if event.side not in self.HEEL_MARKER_BY_SIDE:
                continue
            marker = self.HEEL_MARKER_BY_SIDE[event.side]
            if marker not in labels:
                continue
            frame = int(round(event.time_seconds * rate)) - first_frame
            value = self._marker_value_near_frame(
                points=points,
                marker_index=labels.index(marker),
                frame=frame,
                axis=ap_axis,
            )
            if not np.isfinite(value):
                continue
            samples.append(
                MarkerSample(
                    time_seconds=event.time_seconds,
                    side=event.side,
                    frame_index=frame,
                    ap_position_mm=float(value),
                )
            )
        return sorted(samples, key=lambda sample: sample.time_seconds)

    def _valid_marker_xyz(self, points: np.ndarray, marker_index: int) -> np.ndarray | None:
        xyz = points[:3, marker_index, :].copy()
        residual = points[3, marker_index, :]
        valid = (
            (residual >= 0)
            & np.all(np.isfinite(xyz), axis=0)
            & ~np.all(np.isclose(xyz, 0.0), axis=0)
        )
        xyz[:, ~valid] = np.nan
        if np.isfinite(xyz).sum() == 0:
            return None
        return xyz

    def _marker_value_near_frame(
        self,
        points: np.ndarray,
        marker_index: int,
        frame: int,
        axis: int,
        search_radius: int = 5,
    ) -> float:
        n_frames = points.shape[2]
        for offset in range(search_radius + 1):
            for candidate in (frame - offset, frame + offset):
                if candidate < 0 or candidate >= n_frames:
                    continue
                xyz = points[:3, marker_index, candidate]
                residual = points[3, marker_index, candidate]
                if residual >= 0 and np.all(np.isfinite(xyz)) and not np.all(np.isclose(xyz, 0.0)):
                    return float(xyz[axis])
        return np.nan

    def _summarize_event_samples(
        self,
        samples: list[MarkerSample],
        points: np.ndarray,
        labels: list[str],
        rate: float,
        ap_axis: int,
    ) -> dict[str, float]:
        if len(samples) < 2:
            return self._empty_metrics()

        active_seconds = samples[-1].time_seconds - samples[0].time_seconds
        step_times = np.asarray(
            [
                later.time_seconds - earlier.time_seconds
                for earlier, later in zip(samples, samples[1:], strict=False)
                if later.side != earlier.side
            ],
            dtype=float,
        )
        step_lengths = np.asarray(
            [
                abs(later.ap_position_mm - earlier.ap_position_mm) / 1000.0
                for earlier, later in zip(samples, samples[1:], strict=False)
                if later.side != earlier.side
            ],
            dtype=float,
        )

        stride_times_by_side: dict[str, list[float]] = {"left": [], "right": []}
        stride_lengths_by_side: dict[str, list[float]] = {"left": [], "right": []}
        for side in ("left", "right"):
            side_samples = [sample for sample in samples if sample.side == side]
            stride_times_by_side[side] = [
                later.time_seconds - earlier.time_seconds
                for earlier, later in zip(side_samples, side_samples[1:], strict=False)
            ]
            stride_lengths_by_side[side] = [
                abs(later.ap_position_mm - earlier.ap_position_mm) / 1000.0
                for earlier, later in zip(side_samples, side_samples[1:], strict=False)
            ]

        stride_times = np.asarray(
            stride_times_by_side["left"] + stride_times_by_side["right"],
            dtype=float,
        )
        stride_lengths = np.asarray(
            stride_lengths_by_side["left"] + stride_lengths_by_side["right"],
            dtype=float,
        )
        walking_speed = self._walking_speed(points, labels, samples, rate, ap_axis)

        metrics = {
            "active_seconds": active_seconds,
            "cadence_steps_per_min": ((len(samples) - 1) / active_seconds * 60.0)
            if active_seconds > 0
            else np.nan,
            "walking_speed_mps": walking_speed,
            "step_time_mean_s": nanmean(step_times),
            "step_time_sd_s": nansd(step_times),
            "step_time_cv": coefficient_of_variation(step_times),
            "step_length_mean_m": nanmean(step_lengths),
            "step_length_sd_m": nansd(step_lengths),
            "step_length_cv": coefficient_of_variation(step_lengths),
            "stride_time_mean_s": nanmean(stride_times),
            "stride_time_sd_s": nansd(stride_times),
            "stride_time_cv": coefficient_of_variation(stride_times),
            "stride_length_mean_m": nanmean(stride_lengths),
            "stride_length_sd_m": nansd(stride_lengths),
            "stride_length_cv": coefficient_of_variation(stride_lengths),
        }
        for side in ("left", "right"):
            metrics[f"{side}_stride_time_mean_s"] = nanmean(stride_times_by_side[side])
            metrics[f"{side}_stride_length_mean_m"] = nanmean(stride_lengths_by_side[side])
        metrics["stride_time_asymmetry_s"] = abs(
            metrics["left_stride_time_mean_s"] - metrics["right_stride_time_mean_s"]
        )
        metrics["stride_length_asymmetry_m"] = abs(
            metrics["left_stride_length_mean_m"] - metrics["right_stride_length_mean_m"]
        )
        return metrics

    def _walking_speed(
        self,
        points: np.ndarray,
        labels: list[str],
        samples: list[MarkerSample],
        rate: float,
        ap_axis: int,
    ) -> float:
        if len(samples) < 2 or rate <= 0:
            return np.nan
        marker = next((name for name in self.PELVIS_MARKERS if name in labels), None)
        if marker is None:
            return np.nan
        start_frame = samples[0].frame_index
        end_frame = samples[-1].frame_index
        marker_index = labels.index(marker)
        start = self._marker_value_near_frame(points, marker_index, start_frame, ap_axis)
        end = self._marker_value_near_frame(points, marker_index, end_frame, ap_axis)
        elapsed = samples[-1].time_seconds - samples[0].time_seconds
        if not np.isfinite(start) or not np.isfinite(end) or elapsed <= 0:
            return np.nan
        return abs(end - start) / 1000.0 / elapsed

    def _empty_metrics(self) -> dict[str, float]:
        metric_names = set(VALIDATION_CONFIG["gait_metrics"].values()) | {
            "active_seconds",
            "step_time_sd_s",
            "step_length_sd_m",
            "step_length_cv",
            "stride_time_sd_s",
            "stride_length_sd_m",
            "stride_length_cv",
            "left_stride_time_mean_s",
            "right_stride_time_mean_s",
            "left_stride_length_mean_m",
            "right_stride_length_mean_m",
        }
        return {name: np.nan for name in sorted(metric_names)}


class ValidationAnalysis:
    """Build clinical and standard-gait validation tables."""

    def __init__(self, dataset_path: Path, data_dir: Path, output_dir: Path) -> None:
        self.dataset_path = dataset_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.gait_extractor = C3DGaitMetricExtractor()

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._load_dataset()
        gait_trial = self._build_trial_gait_metrics()
        gait_dataset = self._aggregate_gait_metrics(gait_trial)

        clinical = self._change_correlations(
            dataset,
            VALIDATION_CONFIG["lle_endpoints"],
            VALIDATION_CONFIG["clinical_endpoints"],
            analysis="lle_clinical_change_spearman",
        )
        gait_validation = self._change_correlations(
            dataset.merge(
                gait_dataset,
                on=["patient_id", "session_code", "session", "med", "stim", "task"],
                how="inner",
            ),
            VALIDATION_CONFIG["lle_endpoints"],
            VALIDATION_CONFIG["gait_metrics"],
            analysis="lle_gait_metric_change_spearman",
        )
        clinical_partial = self._partial_clinical_correlations(
            dataset.merge(
                gait_dataset,
                on=["patient_id", "session_code", "session", "med", "stim", "task"],
                how="inner",
            )
        )
        focused_clinical = self._focused_clinical_validation(
            clinical=clinical,
            clinical_partial=clinical_partial,
        )
        loo_effects = self._leave_one_patient_out_effects(dataset)
        loo_correlations = self._leave_one_patient_out_correlations(
            dataset.merge(
                gait_dataset,
                on=["patient_id", "session_code", "session", "med", "stim", "task"],
                how="inner",
            )
        )
        gait_stats = self._gait_metric_statistics(gait_dataset)
        model_diagnostics = self._model_diagnostics()

        gait_trial.to_csv(self.output_dir / "gait_event_metrics_trial.csv", index=False)
        gait_dataset.to_csv(self.output_dir / "gait_event_metrics.csv", index=False)
        clinical.to_csv(self.output_dir / "clinical_change_correlations.csv", index=False)
        clinical_partial.to_csv(
            self.output_dir / "clinical_change_partial_correlations.csv",
            index=False,
        )
        focused_clinical.to_csv(
            self.output_dir / "focused_clinical_validation.csv",
            index=False,
        )
        loo_effects.to_csv(self.output_dir / "leave_one_patient_out_effects.csv", index=False)
        loo_correlations.to_csv(
            self.output_dir / "leave_one_patient_out_clinical_correlations.csv",
            index=False,
        )
        gait_validation.to_csv(self.output_dir / "gait_metric_change_correlations.csv", index=False)
        gait_stats.to_csv(self.output_dir / "gait_metric_statistics.csv", index=False)
        model_diagnostics.to_csv(self.output_dir / "bayesian_model_diagnostics.csv", index=False)
        self._write_summary(
            clinical,
            clinical_partial,
            focused_clinical,
            loo_effects,
            loo_correlations,
            gait_validation,
            gait_stats,
            model_diagnostics,
        )

        print("Done.")
        print(f"Dataset rows: {len(dataset)}")
        print(f"Gait trial rows: {len(gait_trial)}")
        print(f"Gait patient-session-task rows: {len(gait_dataset)}")
        print(f"Output: {self.output_dir}")

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path)
        required = {"patient_id", "session_code", "session", "med", "stim", "task"}
        required |= set(VALIDATION_CONFIG["lle_endpoints"].values())
        required |= set(VALIDATION_CONFIG["clinical_endpoints"].values())
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required dataset columns: {missing}")
        return df

    def _build_trial_gait_metrics(self) -> pd.DataFrame:
        trials = TrialDiscoverer(self.data_dir, list(VALIDATION_CONFIG["tasks"])).discover()
        rows = []
        errors = []
        for index, trial in enumerate(trials, start=1):
            try:
                rows.append(self.gait_extractor.read_trial_metrics(trial))
            except Exception as exc:  # noqa: BLE001 - keep batch analysis going.
                errors.append(
                    {
                        "trial_id": trial.trial_id,
                        "task": trial.task,
                        "session_code": trial.session_code,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if index % 50 == 0:
                print(f"Processed gait events: {index}/{len(trials)}")
        if errors:
            pd.DataFrame(errors).to_csv(self.output_dir / "gait_event_errors.csv", index=False)
        return pd.DataFrame(rows)

    def _aggregate_gait_metrics(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        if trial_df.empty:
            return pd.DataFrame()
        numeric_cols = trial_df.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {"session", "sample_rate_hz", "duration_seconds"}
        keep_numeric = [col for col in numeric_cols if col not in excluded]
        group_cols = ["patient_id", "session_code", "session", "med", "stim", "task"]
        return (
            trial_df.groupby(group_cols, dropna=False)
            .agg(
                **{col: (col, "mean") for col in keep_numeric},
                n_trials_with_events=("trial_id", "count"),
                total_foot_strikes=("n_foot_strikes", "sum"),
            )
            .reset_index()
        )

    def _change_correlations(
        self,
        df: pd.DataFrame,
        metrics_a: dict[str, str],
        metrics_b: dict[str, str],
        analysis: str,
    ) -> pd.DataFrame:
        rows = []
        for task in sorted(df["task"].dropna().unique()):
            task_df = df[df["task"] == task]
            for contrast, (session_a, session_b) in VALIDATION_CONFIG["paired_contrasts"].items():
                value_columns = list(metrics_a.values()) + list(metrics_b.values())
                pivot = task_df.pivot_table(
                    index="patient_id",
                    columns="session_code",
                    values=value_columns,
                    aggfunc="mean",
                )
                for name_a, col_a in metrics_a.items():
                    for name_b, col_b in metrics_b.items():
                        if not self._has_contrast_columns(
                            pivot, col_a, col_b, session_a, session_b
                        ):
                            continue
                        delta_a = pivot[(col_a, session_b)] - pivot[(col_a, session_a)]
                        delta_b = pivot[(col_b, session_b)] - pivot[(col_b, session_a)]
                        result = spearman_result(delta_a, delta_b)
                        rows.append(
                            base_result_row(
                                analysis=analysis,
                                task=task,
                                contrast=contrast,
                                metric_a=name_a,
                                metric_b=name_b,
                                n=result["n"],
                                estimate=result["rho"],
                                p_value=result["p_value"],
                                mean_delta_a=float(delta_a.dropna().mean()),
                                mean_delta_b=float(delta_b.dropna().mean()),
                                notes=f"{SESSION_LABELS[session_b]} minus {SESSION_LABELS[session_a]}",
                            )
                        )
        return add_correlation_fdr(pd.DataFrame(rows, columns=RESULT_COLUMNS))

    def _partial_clinical_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        metrics_a = VALIDATION_CONFIG["lle_endpoints"]
        metrics_b = VALIDATION_CONFIG["clinical_endpoints"]
        controls = VALIDATION_CONFIG["partial_controls"]
        for task in sorted(df["task"].dropna().unique()):
            task_df = df[df["task"] == task]
            value_columns = (
                list(metrics_a.values()) + list(metrics_b.values()) + list(controls.values())
            )
            for contrast, (session_a, session_b) in VALIDATION_CONFIG["paired_contrasts"].items():
                pivot = task_df.pivot_table(
                    index="patient_id",
                    columns="session_code",
                    values=value_columns,
                    aggfunc="mean",
                )
                for name_a, col_a in metrics_a.items():
                    for name_b, col_b in metrics_b.items():
                        for control_name, control_col in controls.items():
                            if not self._has_partial_columns(
                                pivot,
                                col_a,
                                col_b,
                                control_col,
                                session_a,
                                session_b,
                            ):
                                continue
                            delta_a = pivot[(col_a, session_b)] - pivot[(col_a, session_a)]
                            delta_b = pivot[(col_b, session_b)] - pivot[(col_b, session_a)]
                            delta_control = (
                                pivot[(control_col, session_b)] - pivot[(control_col, session_a)]
                            )
                            result = partial_spearman_result(delta_a, delta_b, delta_control)
                            rows.append(
                                base_result_row(
                                    analysis="lle_clinical_change_partial_spearman",
                                    task=task,
                                    contrast=contrast,
                                    metric_a=name_a,
                                    metric_b=name_b,
                                    n=result["n"],
                                    estimate=result["rho"],
                                    p_value=result["p_value"],
                                    mean_delta_a=float(delta_a.dropna().mean()),
                                    mean_delta_b=float(delta_b.dropna().mean()),
                                    notes=(
                                        f"{SESSION_LABELS[session_b]} minus {SESSION_LABELS[session_a]}; "
                                        f"rank partial correlation controlling {control_name}"
                                    ),
                                )
                            )
        return add_correlation_fdr(pd.DataFrame(rows, columns=RESULT_COLUMNS))

    def _focused_clinical_validation(
        self,
        clinical: pd.DataFrame,
        clinical_partial: pd.DataFrame,
    ) -> pd.DataFrame:
        focused = []
        for source, df in (
            ("spearman", clinical),
            ("partial_spearman_speed_adjusted", clinical_partial),
        ):
            subset = df[
                (df["task"] == "normal_walking") & (df["contrast"] == "OFF_OFF_vs_ON_ON")
            ].copy()
            pair_mask = pd.Series(False, index=subset.index)
            for metric_a, metric_b in VALIDATION_CONFIG["focused_clinical_pairs"]:
                pair_mask |= subset["metric_a"].eq(metric_a) & subset["metric_b"].eq(metric_b)
            subset = subset[pair_mask].copy()
            subset["validation_set"] = source
            focused.append(subset)
        if not focused:
            return pd.DataFrame()
        result = pd.concat(focused, ignore_index=True)
        result["p_fdr_focused"] = np.nan
        for _, index in result.groupby("validation_set", dropna=False).groups.items():
            result.loc[index, "p_fdr_focused"] = bh_fdr(result.loc[index, "p_value"])
        leading = [
            "validation_set",
            "analysis",
            "task",
            "contrast",
            "metric_a",
            "metric_b",
            "n",
            "estimate",
            "p_value",
            "p_fdr_focused",
            "p_fdr_task_contrast",
            "mean_delta_a",
            "mean_delta_b",
            "notes",
        ]
        return result[leading]

    def _leave_one_patient_out_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        patients = sorted(df["patient_id"].dropna().unique())
        for task, contrast, metric_name in VALIDATION_CONFIG["key_lle_paired_effects"]:
            session_a, session_b = VALIDATION_CONFIG["paired_contrasts"][contrast]
            column = VALIDATION_CONFIG["lle_endpoints"][metric_name]
            task_df = df[df["task"] == task]
            full = paired_effect_result(task_df, column, session_a, session_b)
            loo_values = []
            for held_out in patients:
                sub = task_df[task_df["patient_id"] != held_out]
                result = paired_effect_result(sub, column, session_a, session_b)
                rows.append(
                    {
                        "analysis": "leave_one_patient_out_paired_effect",
                        "task": task,
                        "contrast": contrast,
                        "metric": metric_name,
                        "held_out_patient_id": held_out,
                        "n_pairs": result["n_pairs"],
                        "full_mean_diff_b_minus_a": full["mean_diff"],
                        "loo_mean_diff_b_minus_a": result["mean_diff"],
                        "full_effect_size_dz": full["effect_size_dz"],
                        "loo_effect_size_dz": result["effect_size_dz"],
                        "full_p_value": full["p_value"],
                        "loo_p_value": result["p_value"],
                        "same_direction_as_full": same_direction(
                            full["mean_diff"],
                            result["mean_diff"],
                        ),
                    }
                )
                loo_values.append(result["mean_diff"])
            finite = pd.Series(loo_values).dropna()
            rows.append(
                {
                    "analysis": "leave_one_patient_out_paired_effect_summary",
                    "task": task,
                    "contrast": contrast,
                    "metric": metric_name,
                    "held_out_patient_id": "",
                    "n_pairs": full["n_pairs"],
                    "full_mean_diff_b_minus_a": full["mean_diff"],
                    "loo_mean_diff_b_minus_a": np.nan,
                    "full_effect_size_dz": full["effect_size_dz"],
                    "loo_effect_size_dz": np.nan,
                    "full_p_value": full["p_value"],
                    "loo_p_value": np.nan,
                    "same_direction_as_full": float(
                        np.mean([same_direction(full["mean_diff"], value) for value in finite])
                    )
                    if len(finite)
                    else np.nan,
                    "loo_min_mean_diff_b_minus_a": float(finite.min()) if len(finite) else np.nan,
                    "loo_max_mean_diff_b_minus_a": float(finite.max()) if len(finite) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _leave_one_patient_out_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        patients = sorted(df["patient_id"].dropna().unique())
        task = "normal_walking"
        contrast = "OFF_OFF_vs_ON_ON"
        session_a, session_b = VALIDATION_CONFIG["paired_contrasts"][contrast]
        task_df = df[df["task"] == task]
        for metric_a, metric_b in VALIDATION_CONFIG["focused_clinical_pairs"]:
            col_a = VALIDATION_CONFIG["lle_endpoints"][metric_a]
            col_b = VALIDATION_CONFIG["clinical_endpoints"][metric_b]
            control_col = VALIDATION_CONFIG["partial_controls"]["walking_speed_mps"]
            full_delta = paired_deltas(task_df, [col_a, col_b, control_col], session_a, session_b)
            full = partial_spearman_result(
                full_delta[col_a], full_delta[col_b], full_delta[control_col]
            )
            loo_estimates = []
            for held_out in patients:
                sub = task_df[task_df["patient_id"] != held_out]
                delta = paired_deltas(sub, [col_a, col_b, control_col], session_a, session_b)
                result = partial_spearman_result(delta[col_a], delta[col_b], delta[control_col])
                rows.append(
                    {
                        "analysis": "leave_one_patient_out_partial_spearman",
                        "task": task,
                        "contrast": contrast,
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "held_out_patient_id": held_out,
                        "n": result["n"],
                        "full_estimate": full["rho"],
                        "loo_estimate": result["rho"],
                        "full_p_value": full["p_value"],
                        "loo_p_value": result["p_value"],
                        "same_direction_as_full": same_direction(full["rho"], result["rho"]),
                        "notes": "Rank partial correlation controlling walking_speed_mps",
                    }
                )
                loo_estimates.append(result["rho"])
            finite = pd.Series(loo_estimates).dropna()
            rows.append(
                {
                    "analysis": "leave_one_patient_out_partial_spearman_summary",
                    "task": task,
                    "contrast": contrast,
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "held_out_patient_id": "",
                    "n": full["n"],
                    "full_estimate": full["rho"],
                    "loo_estimate": np.nan,
                    "full_p_value": full["p_value"],
                    "loo_p_value": np.nan,
                    "same_direction_as_full": float(
                        np.mean([same_direction(full["rho"], value) for value in finite])
                    )
                    if len(finite)
                    else np.nan,
                    "loo_min_estimate": float(finite.min()) if len(finite) else np.nan,
                    "loo_max_estimate": float(finite.max()) if len(finite) else np.nan,
                    "notes": "Rank partial correlation controlling walking_speed_mps",
                }
            )
        return pd.DataFrame(rows)

    def _has_contrast_columns(
        self,
        pivot: pd.DataFrame,
        col_a: str,
        col_b: str,
        session_a: str,
        session_b: str,
    ) -> bool:
        return all(
            item in pivot.columns
            for item in (
                (col_a, session_a),
                (col_a, session_b),
                (col_b, session_a),
                (col_b, session_b),
            )
        )

    def _has_partial_columns(
        self,
        pivot: pd.DataFrame,
        col_a: str,
        col_b: str,
        control_col: str,
        session_a: str,
        session_b: str,
    ) -> bool:
        return all(
            item in pivot.columns
            for item in (
                (col_a, session_a),
                (col_a, session_b),
                (col_b, session_a),
                (col_b, session_b),
                (control_col, session_a),
                (control_col, session_b),
            )
        )

    def _gait_metric_statistics(self, gait_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for task in sorted(gait_df["task"].dropna().unique()):
            task_df = gait_df[gait_df["task"] == task]
            for metric_name, column in VALIDATION_CONFIG["gait_metrics"].items():
                for session in SESSION_ORDER:
                    values = task_df.loc[task_df["session_code"] == session, column]
                    summary = numeric_summary(values)
                    rows.append(
                        {
                            "section": "session_summary",
                            "task": task,
                            "metric": metric_name,
                            "session_code": session,
                            "contrast": "",
                            **summary,
                            "mean_a": np.nan,
                            "mean_b": np.nan,
                            "mean_diff_b_minus_a": np.nan,
                            "p_value": np.nan,
                            "effect_size_dz": np.nan,
                            "notes": SESSION_LABELS[session],
                        }
                    )
                pivot = task_df.pivot_table(
                    index="patient_id",
                    columns="session_code",
                    values=column,
                    aggfunc="mean",
                )
                for contrast, (session_a, session_b) in VALIDATION_CONFIG[
                    "paired_contrasts"
                ].items():
                    if session_a not in pivot.columns or session_b not in pivot.columns:
                        continue
                    pair = pivot[[session_a, session_b]].dropna()
                    diff = pair[session_b] - pair[session_a]
                    rows.append(
                        {
                            "section": "paired_contrast",
                            "task": task,
                            "metric": metric_name,
                            "session_code": "",
                            "contrast": contrast,
                            **numeric_summary(diff),
                            "mean_a": float(pair[session_a].mean()) if len(pair) else np.nan,
                            "mean_b": float(pair[session_b].mean()) if len(pair) else np.nan,
                            "mean_diff_b_minus_a": float(diff.mean()) if len(pair) else np.nan,
                            "p_value": wilcoxon_from_diff(diff),
                            "effect_size_dz": cohen_dz(diff),
                            "notes": f"{SESSION_LABELS[session_b]} minus {SESSION_LABELS[session_a]}",
                        }
                    )
        return pd.DataFrame(rows)

    def _model_diagnostics(self) -> pd.DataFrame:
        path = Path("outputs/bayesian_hierarchical_models/bayesian_summary.csv")
        if not path.exists():
            return pd.DataFrame(
                [
                    {
                        "metric": "bayesian_summary_available",
                        "value": 0,
                        "notes": str(path),
                    }
                ]
            )
        summary = pd.read_csv(path)
        coefficients = summary[summary["section"] == "coefficient"].copy()
        coefficients["r_hat"] = pd.to_numeric(coefficients["r_hat"], errors="coerce")
        coefficients["ess_bulk"] = pd.to_numeric(coefficients["ess_bulk"], errors="coerce")
        model_level = (
            coefficients.groupby(["model", "endpoint", "metric_variant"], dropna=False)
            .agg(
                max_r_hat=("r_hat", "max"),
                min_ess_bulk=("ess_bulk", "min"),
                divergences=("divergences", "first"),
                n_rows=("n_rows", "first"),
                n_patients=("n_patients", "first"),
            )
            .reset_index()
        )
        return model_level

    def _write_summary(
        self,
        clinical: pd.DataFrame,
        clinical_partial: pd.DataFrame,
        focused_clinical: pd.DataFrame,
        loo_effects: pd.DataFrame,
        loo_correlations: pd.DataFrame,
        gait_validation: pd.DataFrame,
        gait_stats: pd.DataFrame,
        model_diagnostics: pd.DataFrame,
    ) -> None:
        lines = [
            "Clinical and gait validation summary",
            "",
            "Clinical-change correlations with |rho| >= 0.35 and p < 0.15:",
            *format_selected_correlations(clinical),
            "",
            "Clinical-change partial correlations controlling walking-speed change:",
            *format_selected_correlations(clinical_partial),
            "",
            "Focused natural-walking OFF/OFF to ON/ON clinical validation:",
            *format_focused_validation(focused_clinical),
            "",
            "Leave-one-patient-out stability:",
            *format_loo_stability(loo_effects, loo_correlations),
            "",
            "Gait-metric change correlations with |rho| >= 0.35 and p < 0.15:",
            *format_selected_correlations(gait_validation),
            "",
            "Classical gait metric paired contrasts with p < 0.05:",
            *format_selected_gait_contrasts(gait_stats),
            "",
            "Bayesian model diagnostics:",
            *format_model_diagnostics(model_diagnostics),
        ]
        (self.output_dir / "validation_summary.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def reproduce_validation_tables(
    dataset_path: Path,
    gait_metrics_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Reproduce the validation tables that only require released data."""
    analysis = ValidationAnalysis(dataset_path, Path("."), output_dir)
    dataset = analysis._load_dataset()
    gait_dataset = pd.read_csv(gait_metrics_path)
    join_columns = ["patient_id", "session_code", "session", "med", "stim", "task"]
    merged = dataset.merge(gait_dataset, on=join_columns, how="inner")

    clinical = analysis._change_correlations(
        dataset,
        VALIDATION_CONFIG["lle_endpoints"],
        VALIDATION_CONFIG["clinical_endpoints"],
        analysis="lle_clinical_change_spearman",
    )
    clinical_partial = analysis._partial_clinical_correlations(merged)
    outputs = {
        "clinical_change_correlations.csv": clinical,
        "clinical_change_partial_correlations.csv": clinical_partial,
        "focused_clinical_validation.csv": analysis._focused_clinical_validation(
            clinical, clinical_partial
        ),
        "gait_metric_change_correlations.csv": analysis._change_correlations(
            merged,
            VALIDATION_CONFIG["lle_endpoints"],
            VALIDATION_CONFIG["gait_metrics"],
            analysis="lle_gait_metric_change_spearman",
        ),
        "gait_metric_statistics.csv": analysis._gait_metric_statistics(gait_dataset),
        "leave_one_patient_out_effects.csv": analysis._leave_one_patient_out_effects(dataset),
        "leave_one_patient_out_clinical_correlations.csv": (
            analysis._leave_one_patient_out_correlations(merged)
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    return outputs


def base_result_row(**values: object) -> dict[str, object]:
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(values)
    return row


def nanmean(values) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else np.nan


def nansd(values) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if len(array) > 1 else np.nan


def coefficient_of_variation(values) -> float:
    mean = nanmean(values)
    sd = nansd(values)
    if not np.isfinite(mean) or not np.isfinite(sd) or abs(mean) < 1e-12:
        return np.nan
    return float(sd / abs(mean))


def numeric_summary(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "iqr": np.nan}
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "median": float(clean.median()),
        "iqr": float(clean.quantile(0.75) - clean.quantile(0.25)),
    }


def paired_deltas(
    df: pd.DataFrame,
    columns: list[str],
    session_a: str,
    session_b: str,
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="patient_id",
        columns="session_code",
        values=columns,
        aggfunc="mean",
    )
    rows = {}
    for column in columns:
        if (column, session_a) not in pivot.columns or (
            column,
            session_b,
        ) not in pivot.columns:
            rows[column] = pd.Series(dtype=float)
            continue
        rows[column] = pivot[(column, session_b)] - pivot[(column, session_a)]
    return pd.DataFrame(rows)


def paired_effect_result(
    df: pd.DataFrame,
    column: str,
    session_a: str,
    session_b: str,
) -> dict[str, float | int]:
    delta = paired_deltas(df, [column], session_a, session_b)[column].dropna()
    return {
        "n_pairs": int(len(delta)),
        "mean_diff": float(delta.mean()) if len(delta) else np.nan,
        "p_value": wilcoxon_from_diff(delta),
        "effect_size_dz": cohen_dz(delta),
    }


def same_direction(reference: float, value: float) -> bool:
    if not np.isfinite(reference) or not np.isfinite(value):
        return False
    if abs(reference) < 1e-12 or abs(value) < 1e-12:
        return False
    return bool(np.sign(reference) == np.sign(value))


def spearman_result(values_a: pd.Series, values_b: pd.Series) -> dict[str, float | int]:
    pair = pd.concat([values_a, values_b], axis=1).dropna()
    if len(pair) < 5 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return {"n": int(len(pair)), "rho": np.nan, "p_value": np.nan}
    rho, p_value = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return {"n": int(len(pair)), "rho": float(rho), "p_value": float(p_value)}


def add_correlation_fdr(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    results = results.copy()
    results["p_fdr_task_contrast"] = np.nan
    for _, index in results.groupby(["analysis", "task", "contrast"], dropna=False).groups.items():
        results.loc[index, "p_fdr_task_contrast"] = bh_fdr(results.loc[index, "p_value"])
    return results[RESULT_COLUMNS]


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


def partial_spearman_result(
    values_a: pd.Series,
    values_b: pd.Series,
    control: pd.Series,
) -> dict[str, float | int]:
    pair = pd.concat([values_a, values_b, control], axis=1).dropna()
    if len(pair) < 6:
        return {"n": int(len(pair)), "rho": np.nan, "p_value": np.nan}
    ranked = pair.rank()
    if ranked.iloc[:, 0].nunique() < 2 or ranked.iloc[:, 1].nunique() < 2:
        return {"n": int(len(pair)), "rho": np.nan, "p_value": np.nan}
    x = np.column_stack([np.ones(len(ranked)), ranked.iloc[:, 2].to_numpy(dtype=float)])
    residual_a = residualize(ranked.iloc[:, 0].to_numpy(dtype=float), x)
    residual_b = residualize(ranked.iloc[:, 1].to_numpy(dtype=float), x)
    rho = float(np.corrcoef(residual_a, residual_b)[0, 1])
    p_value = partial_correlation_p_value(rho, n=len(pair), n_controls=1)
    return {"n": int(len(pair)), "rho": float(rho), "p_value": float(p_value)}


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def partial_correlation_p_value(rho: float, n: int, n_controls: int) -> float:
    df = n - n_controls - 2
    if df <= 0 or not np.isfinite(rho):
        return np.nan
    clipped = float(np.clip(rho, -0.999999, 0.999999))
    t_value = clipped * np.sqrt(df / (1.0 - clipped**2))
    return float(2.0 * stats.t.sf(abs(t_value), df=df))


def wilcoxon_from_diff(diff: pd.Series) -> float:
    clean = pd.to_numeric(diff, errors="coerce").dropna()
    clean = clean[np.abs(clean) > 1e-12]
    if len(clean) < 3:
        return np.nan
    try:
        return float(stats.wilcoxon(clean, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def cohen_dz(diff: pd.Series) -> float:
    clean = pd.to_numeric(diff, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    sd = clean.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(clean.mean() / sd)


def format_selected_correlations(results: pd.DataFrame, max_lines: int = 18) -> list[str]:
    if results.empty:
        return ["none"]
    selected = (
        results[(results["estimate"].abs() >= 0.35) & (results["p_value"] < 0.15)]
        .sort_values(["task", "contrast", "p_value"])
        .head(max_lines)
    )
    if selected.empty:
        return ["none"]
    lines = []
    for _, row in selected.iterrows():
        lines.append(
            f"- {row['task']} {row['contrast']}: {row['metric_a']} vs {row['metric_b']}, "
            f"n={int(row['n'])}, rho={row['estimate']:.2f}, "
            f"p={row['p_value']:.3f}, q={row['p_fdr_task_contrast']:.3f}"
        )
    return lines


def format_focused_validation(results: pd.DataFrame) -> list[str]:
    if results.empty:
        return ["none"]
    selected = results.sort_values(["validation_set", "p_value"])
    lines = []
    for _, row in selected.iterrows():
        lines.append(
            f"- {row['validation_set']}: {row['metric_a']} vs {row['metric_b']}, "
            f"n={int(row['n'])}, rho={row['estimate']:.2f}, "
            f"p={row['p_value']:.3f}, focused q={row['p_fdr_focused']:.3f}"
        )
    return lines


def format_loo_stability(loo_effects: pd.DataFrame, loo_correlations: pd.DataFrame) -> list[str]:
    lines = []
    if not loo_effects.empty:
        selected = loo_effects[
            loo_effects["analysis"] == "leave_one_patient_out_paired_effect_summary"
        ]
        for _, row in selected.iterrows():
            lines.append(
                f"- effect {row['task']} {row['contrast']} {row['metric']}: "
                f"full delta={row['full_mean_diff_b_minus_a']:.4f}, "
                f"LOO range=[{row['loo_min_mean_diff_b_minus_a']:.4f}, "
                f"{row['loo_max_mean_diff_b_minus_a']:.4f}], "
                f"same direction={row['same_direction_as_full']:.0%}"
            )
    if not loo_correlations.empty:
        selected = loo_correlations[
            loo_correlations["analysis"] == "leave_one_patient_out_partial_spearman_summary"
        ]
        for _, row in selected.iterrows():
            lines.append(
                f"- partial correlation {row['metric_a']} vs {row['metric_b']}: "
                f"full rho={row['full_estimate']:.2f}, "
                f"LOO range=[{row['loo_min_estimate']:.2f}, {row['loo_max_estimate']:.2f}], "
                f"same direction={row['same_direction_as_full']:.0%}"
            )
    return lines or ["none"]


def format_selected_gait_contrasts(results: pd.DataFrame, max_lines: int = 18) -> list[str]:
    if results.empty or "section" not in results:
        return ["none"]
    selected = (
        results[(results["section"] == "paired_contrast") & (results["p_value"] < 0.05)]
        .sort_values(["task", "p_value"])
        .head(max_lines)
    )
    if selected.empty:
        return ["none"]
    lines = []
    for _, row in selected.iterrows():
        lines.append(
            f"- {row['task']} {row['contrast']}: {row['metric']}, "
            f"n={int(row['n'])}, delta={row['mean_diff_b_minus_a']:.4f}, "
            f"p={row['p_value']:.4f}, dz={row['effect_size_dz']:.2f}"
        )
    return lines


def format_model_diagnostics(model_diagnostics: pd.DataFrame) -> list[str]:
    if model_diagnostics.empty:
        return ["none"]
    required = {"max_r_hat", "min_ess_bulk", "divergences"}
    if not required.issubset(model_diagnostics.columns):
        return ["not available"]
    max_r_hat = pd.to_numeric(model_diagnostics["max_r_hat"], errors="coerce").max()
    min_ess = pd.to_numeric(model_diagnostics["min_ess_bulk"], errors="coerce").min()
    divergences = pd.to_numeric(model_diagnostics["divergences"], errors="coerce").sum()
    n_models = len(model_diagnostics)
    return [
        f"- models={n_models}, max r_hat={max_r_hat:.2f}, "
        f"min bulk ESS={min_ess:.0f}, total divergences={int(divergences)}"
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clinical and gait validation analyses.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/pd_lle_dataset/dataset.csv"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/validation/clinical_and_gait"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = ValidationAnalysis(
        dataset_path=args.dataset.resolve(),
        data_dir=args.data_dir.resolve(),
        output_dir=args.out_dir.resolve(),
    )
    analysis.run()


if __name__ == "__main__":
    main()
