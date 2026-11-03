"""PD gait dataset processor.

This module converts raw C3D motion-capture files and UPDRS-III item scores into
one CSV suitable for downstream statistics and machine learning.

Default output unit:
    one row per subject x session x walking task
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CONFIG = {
    "tasks": {
        "normal": {
            "directory": "1_2_2_ChodPredkoscNaturalna",
            "label": "normal_walking",
        },
        "fast": {"directory": "1_2_3_ChodPredkoscSzybsza", "label": "fast_walking"},
    },
    "sessions": {
        "S01": {"session": 1, "med": "OFF", "stim": "OFF"},
        "S02": {"session": 2, "med": "OFF", "stim": "ON"},
        "S03": {"session": 3, "med": "ON", "stim": "OFF"},
        "S04": {"session": 4, "med": "ON", "stim": "ON"},
    },
    "therapy_condition_to_session": {
        "medOFFstimOFF": "S01",
        "medOFFstimON": "S02",
        "medONstimOFF": "S03",
        "medONstimON": "S04",
    },
    "markers": ("LHEE", "RHEE", "LTOE", "RTOE"),
    # Fixed embedding parameters used by the feature extractor. Use --params-file
    # to run the same processor with an independently calibrated parameter set.
    "embedding": {
        "LHEE": {"d": 3, "tau": 25},
        "RHEE": {"d": 3, "tau": 25},
        "LTOE": {"d": 3, "tau": 25},
        "RTOE": {"d": 3, "tau": 25},
    },
}

TASK_DIRS = {key: value["directory"] for key, value in CONFIG["tasks"].items()}
TASK_LABELS = {key: value["label"] for key, value in CONFIG["tasks"].items()}
SESSION_CONDITIONS = CONFIG["sessions"]
THERAPY_CONDITION_TO_SESSION = CONFIG["therapy_condition_to_session"]
MARKERS = CONFIG["markers"]
GLOBAL_EMBEDDING = CONFIG["embedding"]


@dataclass(frozen=True)
class ProcessingConfig:
    """Fixed parameters for the short-window LLE feature extraction."""

    window_seconds: float = 1.0
    step_seconds: float = 0.1
    theiler_seconds: float = 0.1
    smoothing_window_length: int = 11
    smoothing_polyorder: int = 3
    embedding_params: dict[str, dict[str, int]] | None = None

    @property
    def marker_params(self) -> dict[str, dict[str, int]]:
        return self.embedding_params or GLOBAL_EMBEDDING


@dataclass(frozen=True)
class TrialRecord:
    """Metadata for one raw C3D trial."""

    path: Path
    patient_id: str
    session_code: str
    exercise_code: str
    trial_code: str
    task_key: str

    @property
    def task(self) -> str:
        return TASK_LABELS[self.task_key]

    @property
    def trial_id(self) -> str:
        return f"{self.patient_id}_{self.session_code}_{self.exercise_code}_{self.trial_code}_{self.task_key}"


class C3DMarkerReader:
    """Read marker Z trajectories from C3D files."""

    def __init__(self, markers: tuple[str, ...] = MARKERS) -> None:
        self.markers = markers

    def read(self, path: Path) -> tuple[dict[str, np.ndarray], float, int]:
        c3d = ezc3d.c3d(str(path))
        labels = [label.strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
        rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
        points = np.asarray(c3d["data"]["points"], dtype=float)
        n_frames = points.shape[2]

        series: dict[str, np.ndarray] = {}
        for marker in self.markers:
            if marker not in labels:
                series[marker] = np.full(n_frames, np.nan)
                continue
            idx = labels.index(marker)
            xyz = points[:3, idx, :].T
            residual = points[3, idx, :]
            z = xyz[:, 2].copy()
            invalid = (~np.isfinite(z)) | (residual < 0) | np.all(np.isclose(xyz, 0.0), axis=1)
            z[invalid] = np.nan
            series[marker] = z
        return series, rate, n_frames


class ShortWindowLLE:
    """Compute short-window Lyapunov-style local divergence features."""

    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config

    @staticmethod
    def _fill_missing(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        good = np.isfinite(values)
        if good.sum() < 3:
            return values
        x = np.arange(len(values))
        filled = values.copy()
        filled[~good] = np.interp(x[~good], x[good], values[good])
        return filled

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        values = self._fill_missing(values)
        smoothing_window = int(self.config.smoothing_window_length)
        smoothing_polyorder = int(self.config.smoothing_polyorder)
        if smoothing_window % 2 == 0:
            smoothing_window -= 1
        if smoothing_window <= smoothing_polyorder + 1:
            return values
        if np.isfinite(values).sum() < smoothing_window:
            return values
        window_length = min(
            smoothing_window, len(values) if len(values) % 2 == 1 else len(values) - 1
        )
        if window_length <= smoothing_polyorder + 1:
            return values
        return savgol_filter(
            values,
            window_length=window_length,
            polyorder=smoothing_polyorder,
        )

    @staticmethod
    def _embed(values: np.ndarray, dimension: int, delay: int) -> np.ndarray | None:
        n_vectors = len(values) - (dimension - 1) * delay
        if n_vectors < 4:
            return None
        return np.column_stack(
            [values[offset : offset + n_vectors] for offset in range(0, dimension * delay, delay)]
        )

    def local_lle(
        self,
        values: np.ndarray,
        dimension: int,
        delay: int,
        theiler_samples: int = 0,
    ) -> float:
        """Estimate a short-window Lyapunov-style local trajectory divergence slope."""
        clean = self._smooth(values)
        if not np.all(np.isfinite(clean)):
            return np.nan
        embedded = self._embed(clean, dimension, delay)
        if embedded is None:
            return np.nan

        divergence: list[float] = []
        for index in range(len(embedded) - 1):
            distances = np.linalg.norm(embedded[index] - embedded, axis=1)
            if theiler_samples > 0:
                low = max(0, index - theiler_samples)
                high = min(len(embedded), index + theiler_samples + 1)
                distances[low:high] = np.inf
            else:
                distances[index] = np.inf
            neighbor = int(np.argmin(distances))
            if not np.isfinite(distances[neighbor]) or neighbor + 1 >= len(embedded):
                continue

            next_distance = np.linalg.norm(embedded[index + 1] - embedded[neighbor + 1])
            if next_distance > 0:
                divergence.append(float(np.log(next_distance)))

        if len(divergence) < 2:
            return np.nan
        x = np.arange(len(divergence))
        y = np.asarray(divergence, dtype=float)
        slope, _intercept = np.polyfit(x, y, 1)
        return float(slope)

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        if np.isfinite(array).sum() == 0:
            return np.nan
        return float(np.nanmean(array))

    def compute_windows(
        self,
        trial: TrialRecord,
        marker_series: dict[str, np.ndarray],
        rate: float,
        n_frames: int,
    ) -> pd.DataFrame:
        window_samples = max(1, int(round(self.config.window_seconds * rate)))
        step_samples = max(1, int(round(self.config.step_seconds * rate)))
        theiler_samples = max(1, int(round(self.config.theiler_seconds * rate)))
        rows: list[dict[str, object]] = []

        for end in range(window_samples, n_frames + 1, step_samples):
            start = end - window_samples
            marker_lle: dict[str, float] = {}
            marker_lle_theiler: dict[str, float] = {}
            for marker, params in self.config.marker_params.items():
                marker_lle[marker] = self.local_lle(
                    marker_series[marker][start:end],
                    dimension=params["d"],
                    delay=params["tau"],
                    theiler_samples=0,
                )
                marker_lle_theiler[marker] = self.local_lle(
                    marker_series[marker][start:end],
                    dimension=params["d"],
                    delay=params["tau"],
                    theiler_samples=theiler_samples,
                )

            metrics = self._window_metrics(marker_lle)
            metrics_theiler = self._window_metrics(marker_lle_theiler)

            row = {
                "trial_id": trial.trial_id,
                "patient_id": trial.patient_id,
                "session_code": trial.session_code,
                "exercise_code": trial.exercise_code,
                "trial_code": trial.trial_code,
                "task": trial.task,
                "sample_rate_hz": rate,
                "n_frames": n_frames,
                "duration_seconds": n_frames / rate if rate else np.nan,
                "window_start_frame": start,
                "window_end_frame": end - 1,
                "window_center_seconds": ((start + end - 1) / 2) / rate,
                "window_seconds": self.config.window_seconds,
                "step_seconds": self.config.step_seconds,
                "theiler_seconds": self.config.theiler_seconds,
                "smoothing_window_length": self.config.smoothing_window_length,
                "smoothing_polyorder": self.config.smoothing_polyorder,
            }
            row.update(metrics)
            row.update({f"{key}_theiler": value for key, value in metrics_theiler.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def _window_metrics(self, marker_lle: dict[str, float]) -> dict[str, float]:
        lhee = marker_lle["LHEE"]
        rhee = marker_lle["RHEE"]
        ltoe = marker_lle["LTOE"]
        rtoe = marker_lle["RTOE"]
        left_mean = self._safe_mean([lhee, ltoe])
        right_mean = self._safe_mean([rhee, rtoe])
        heel_asymmetry = lhee - rhee if np.isfinite(lhee) and np.isfinite(rhee) else np.nan
        toe_asymmetry = ltoe - rtoe if np.isfinite(ltoe) and np.isfinite(rtoe) else np.nan
        overall_asymmetry = (
            left_mean - right_mean if np.isfinite(left_mean) and np.isfinite(right_mean) else np.nan
        )
        return {
            "LHEE_lle": lhee,
            "RHEE_lle": rhee,
            "LTOE_lle": ltoe,
            "RTOE_lle": rtoe,
            "left_mean_lle": left_mean,
            "right_mean_lle": right_mean,
            "heel_asymmetry": heel_asymmetry,
            "toe_asymmetry": toe_asymmetry,
            "overall_asymmetry": overall_asymmetry,
            "abs_heel_asymmetry": abs(heel_asymmetry) if np.isfinite(heel_asymmetry) else np.nan,
            "abs_toe_asymmetry": abs(toe_asymmetry) if np.isfinite(toe_asymmetry) else np.nan,
            "abs_overall_asymmetry": (
                abs(overall_asymmetry) if np.isfinite(overall_asymmetry) else np.nan
            ),
        }


def positive_fraction(values: pd.Series) -> float:
    finite = values.dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite > 0))


def sign_switches_per_second(values: pd.Series, duration_seconds: float) -> float:
    finite = values.dropna().to_numpy(dtype=float)
    finite = finite[finite != 0]
    if len(finite) < 2 or duration_seconds <= 0:
        return np.nan
    signs = np.sign(finite)
    return float(np.sum(signs[1:] != signs[:-1]) / duration_seconds)


def lag1_autocorrelation(values: pd.Series) -> float:
    finite = values.dropna().to_numpy(dtype=float)
    if len(finite) < 3 or np.nanstd(finite) == 0:
        return np.nan
    return float(np.corrcoef(finite[:-1], finite[1:])[0, 1])


def paired_correlation(left: pd.Series, right: pd.Series) -> float:
    data = pd.concat([left, right], axis=1).dropna()
    if len(data) < 3 or data.iloc[:, 0].std() == 0 or data.iloc[:, 1].std() == 0:
        return np.nan
    return float(data.iloc[:, 0].corr(data.iloc[:, 1]))


def summarize_series(
    prefix: str, values: pd.Series, include_positive_fraction: bool = True
) -> dict[str, float]:
    finite = values.dropna().astype(float)
    if finite.empty:
        summary = {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p90_p10": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    else:
        p10 = float(finite.quantile(0.10))
        p25 = float(finite.quantile(0.25))
        p75 = float(finite.quantile(0.75))
        p90 = float(finite.quantile(0.90))
        summary = {
            f"{prefix}_mean": float(finite.mean()),
            f"{prefix}_median": float(finite.median()),
            f"{prefix}_sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            f"{prefix}_iqr": p75 - p25,
            f"{prefix}_p10": p10,
            f"{prefix}_p90": p90,
            f"{prefix}_p90_p10": p90 - p10,
            f"{prefix}_min": float(finite.min()),
            f"{prefix}_max": float(finite.max()),
        }
    if include_positive_fraction:
        summary[f"{prefix}_positive_fraction"] = positive_fraction(finite)
    return summary


class FeatureSummarizer:
    """Summarize window-level LLE rows into trial and dataset-level features."""

    BASE_COLUMNS = [
        "trial_id",
        "patient_id",
        "session_code",
        "exercise_code",
        "task",
        "trial_code",
        "sample_rate_hz",
        "n_frames",
        "duration_seconds",
    ]

    def summarize_trial(self, window_df: pd.DataFrame) -> dict[str, object]:
        first = window_df.iloc[0]
        row: dict[str, object] = {column: first[column] for column in self.BASE_COLUMNS}
        condition = SESSION_CONDITIONS.get(first["session_code"], {})
        row["session"] = condition.get("session", np.nan)
        row["med"] = condition.get("med", "")
        row["stim"] = condition.get("stim", "")
        row["n_windows"] = int(len(window_df))

        for marker in MARKERS:
            col = f"{marker}_lle"
            row.update(summarize_series(col, window_df[col]))
            row[f"{col}_sign_switches_per_second"] = sign_switches_per_second(
                window_df[col], float(first["duration_seconds"])
            )
            row[f"{col}_lag1_autocorrelation"] = lag1_autocorrelation(window_df[col])
            col_theiler = f"{marker}_lle_theiler"
            row.update(summarize_series(col_theiler, window_df[col_theiler]))
            row[f"{col_theiler}_sign_switches_per_second"] = sign_switches_per_second(
                window_df[col_theiler], float(first["duration_seconds"])
            )
            row[f"{col_theiler}_lag1_autocorrelation"] = lag1_autocorrelation(
                window_df[col_theiler]
            )

        signed_columns = [
            "left_mean_lle",
            "right_mean_lle",
            "heel_asymmetry",
            "toe_asymmetry",
            "overall_asymmetry",
            "left_mean_lle_theiler",
            "right_mean_lle_theiler",
            "heel_asymmetry_theiler",
            "toe_asymmetry_theiler",
            "overall_asymmetry_theiler",
        ]
        absolute_columns = [
            "abs_heel_asymmetry",
            "abs_toe_asymmetry",
            "abs_overall_asymmetry",
            "abs_heel_asymmetry_theiler",
            "abs_toe_asymmetry_theiler",
            "abs_overall_asymmetry_theiler",
        ]
        for col in signed_columns:
            row.update(summarize_series(col, window_df[col]))
        for col in absolute_columns:
            row.update(summarize_series(col, window_df[col], include_positive_fraction=False))

        row["LHEE_RHEE_lle_correlation"] = paired_correlation(
            window_df["LHEE_lle"], window_df["RHEE_lle"]
        )
        row["LTOE_RTOE_lle_correlation"] = paired_correlation(
            window_df["LTOE_lle"], window_df["RTOE_lle"]
        )
        row["left_right_mean_lle_correlation"] = paired_correlation(
            window_df["left_mean_lle"], window_df["right_mean_lle"]
        )
        row["LHEE_RHEE_lle_correlation_theiler"] = paired_correlation(
            window_df["LHEE_lle_theiler"], window_df["RHEE_lle_theiler"]
        )
        row["LTOE_RTOE_lle_correlation_theiler"] = paired_correlation(
            window_df["LTOE_lle_theiler"], window_df["RTOE_lle_theiler"]
        )
        row["left_right_mean_lle_correlation_theiler"] = paired_correlation(
            window_df["left_mean_lle_theiler"], window_df["right_mean_lle_theiler"]
        )
        row["left_dominance_fraction"] = positive_fraction(window_df["overall_asymmetry"])
        row["left_dominance_fraction_theiler"] = positive_fraction(
            window_df["overall_asymmetry_theiler"]
        )
        return row

    def aggregate_dataset_rows(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        if trial_df.empty:
            return pd.DataFrame()
        numeric_cols = trial_df.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {
            "session",
            "sample_rate_hz",
            "n_frames",
            "duration_seconds",
            "n_windows",
        }
        keep_numeric = [col for col in numeric_cols if col not in excluded]
        group_cols = ["patient_id", "session_code", "session", "med", "stim", "task"]
        dataset = (
            trial_df.groupby(group_cols, dropna=False)
            .agg(
                **{col: (col, "mean") for col in keep_numeric},
                n_trials=("trial_id", "count"),
                total_duration_seconds=("duration_seconds", "sum"),
                total_windows=("n_windows", "sum"),
            )
            .reset_index()
        )
        return dataset


class TrialDiscoverer:
    """Find PD C3D walking trials using the known dataset naming pattern."""

    PATTERN = re.compile(r"-(S\d{2})-(E\d{2})-(T\d{2})\.c3d$", re.IGNORECASE)

    def __init__(self, data_dir: Path, tasks: list[str]) -> None:
        self.data_dir = data_dir
        self.tasks = tasks

    def discover(self) -> list[TrialRecord]:
        trials: list[TrialRecord] = []
        for task_key in self.tasks:
            root = self.data_dir / TASK_DIRS[task_key]
            for path in sorted(root.rglob("*.c3d")):
                match = self.PATTERN.search(path.name)
                if match is None:
                    continue
                patient_id = next(
                    (part for part in path.parts if re.fullmatch(r"B\d+", part)), None
                )
                if patient_id is None:
                    continue
                trials.append(
                    TrialRecord(
                        path=path,
                        patient_id=patient_id,
                        session_code=match.group(1).upper(),
                        exercise_code=match.group(2).upper(),
                        trial_code=match.group(3).upper(),
                        task_key=task_key,
                    )
                )
        return trials


def load_embedding_params(path: Path | None) -> dict[str, dict[str, int]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    params: dict[str, dict[str, int]] = {}
    for marker in MARKERS:
        if marker not in loaded:
            raise ValueError(f"Missing marker in params file: {marker}")
        params[marker] = {
            "d": int(loaded[marker]["d"]),
            "tau": int(loaded[marker]["tau"]),
        }
    return params


def load_updrs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    required = {"patient_id", "therapy_condition"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"Missing UPDRS columns: {missing_required}")

    item_cols = [col for col in df.columns if col not in required]
    for col in item_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["patient_id"] = df["patient_id"].astype(str)
    df["session_code"] = df["therapy_condition"].map(THERAPY_CONDITION_TO_SESSION)
    if df["session_code"].isna().any():
        missing = sorted(df.loc[df["session_code"].isna(), "therapy_condition"].dropna().unique())
        raise ValueError(f"Unmapped UPDRS therapy_condition labels: {missing}")
    df["updrs3_total"] = df[item_cols].sum(axis=1, min_count=len(item_cols))
    return df[["patient_id", "session_code", "updrs3_total", *item_cols]]


class DatasetProcessor:
    """End-to-end raw C3D to analysis dataset processor."""

    def __init__(
        self,
        data_dir: Path,
        updrs_path: Path,
        output_dir: Path,
        tasks: list[str],
        config: ProcessingConfig,
        max_files: int | None = None,
        quiet: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.updrs_path = updrs_path
        self.output_dir = output_dir
        self.tasks = tasks
        self.config = config
        self.max_files = max_files
        self.quiet = quiet
        self.reader = C3DMarkerReader()
        self.extractor = ShortWindowLLE(config)
        self.summarizer = FeatureSummarizer()

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trials = TrialDiscoverer(self.data_dir, self.tasks).discover()
        if self.max_files is not None:
            trials = trials[: self.max_files]
        if not trials:
            raise SystemExit(f"No C3D trials found in {self.data_dir} for tasks={self.tasks}")

        trial_rows: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []

        for index, trial in enumerate(trials, start=1):
            if not self.quiet:
                print(f"[{index:03d}/{len(trials):03d}] {trial.trial_id}")
            try:
                marker_series, rate, n_frames = self.reader.read(trial.path)
                window_df = self.extractor.compute_windows(trial, marker_series, rate, n_frames)
                if window_df.empty:
                    raise ValueError("No LLE windows were computed")
                trial_rows.append(self.summarizer.summarize_trial(window_df))
            except Exception as exc:  # noqa: BLE001 - keep batch processing going.
                errors.append(
                    {
                        "trial_id": trial.trial_id,
                        "task": trial.task,
                        "session_code": trial.session_code,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        trial_df = pd.DataFrame(trial_rows)
        dataset = self.summarizer.aggregate_dataset_rows(trial_df)
        dataset = self._merge_clinical_metadata(dataset)

        dataset.to_csv(self.output_dir / "dataset.csv", index=False)

        print("\nDone.")
        print(f"Trials attempted: {len(trials)}")
        print(f"Trials succeeded: {len(trial_rows)}")
        print(f"Trials failed: {len(errors)}")
        print(f"Dataset rows: {len(dataset)}")
        print(f"Dataset: {self.output_dir / 'dataset.csv'}")
        if errors:
            print("Processing errors:")
            for error in errors[:10]:
                print(
                    f"  {error['trial_id']} {error['task']} "
                    f"{error['session_code']} {error['error_type']}: {error['error']}"
                )
            if len(errors) > 10:
                print(f"  ... {len(errors) - 10} more")

    def _merge_clinical_metadata(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            return dataset
        updrs = load_updrs(self.updrs_path)
        merged = dataset.merge(updrs, on=["patient_id", "session_code"], how="left")
        return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PD LLE dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--updrs", type=Path, default=Path("data/updrs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pd_lle_dataset"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_DIRS),
        default=["normal", "fast"],
        help="Walking tasks to include in dataset.csv.",
    )
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--step-seconds", type=float, default=0.1)
    parser.add_argument("--theiler-seconds", type=float, default=0.1)
    parser.add_argument("--smoothing-window-length", type=int, default=11)
    parser.add_argument("--smoothing-polyorder", type=int, default=3)
    parser.add_argument("--params-file", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProcessingConfig(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        theiler_seconds=args.theiler_seconds,
        smoothing_window_length=args.smoothing_window_length,
        smoothing_polyorder=args.smoothing_polyorder,
        embedding_params=load_embedding_params(args.params_file),
    )
    processor = DatasetProcessor(
        data_dir=args.data_dir.resolve(),
        updrs_path=args.updrs.resolve(),
        output_dir=args.output_dir.resolve(),
        tasks=args.tasks,
        config=config,
        max_files=args.max_files,
        quiet=args.quiet,
    )
    processor.run()


if __name__ == "__main__":
    main()
