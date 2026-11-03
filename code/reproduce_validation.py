"""Reproduce clinical and gait validation from the released derived tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from clinical_and_gait_validation import reproduce_validation_tables


def run(dataset_path: Path, gait_path: Path, output_dir: Path) -> None:
    outputs = reproduce_validation_tables(dataset_path, gait_path, output_dir)
    print(f"Wrote {len(outputs)} validation tables to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/derived_features.csv"),
        help="Released participant-session-task feature table.",
    )
    parser.add_argument(
        "--gait-metrics",
        type=Path,
        default=Path("data/gait_metrics.csv"),
        help="Released participant-session-task gait-metric table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduced/validation"),
        help="Directory for the seven reproduced validation tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.dataset.resolve(), args.gait_metrics.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
