"""Export and validate numeric result tables from completed experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from respay.analysis import analyze_results
from respay.paths import PROCESSED_DATA_ROOT, RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-predictions", type=Path, default=RESULTS_ROOT / "respay/residual/predictions.csv"
    )
    parser.add_argument(
        "--direct-predictions",
        type=Path,
        default=RESULTS_ROOT / "respay/direct_target/predictions.csv",
    )
    parser.add_argument(
        "--baseline-predictions", type=Path, default=RESULTS_ROOT / "baselines/predictions.csv"
    )
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "tables")
    args = parser.parse_args()
    outputs = analyze_results(
        args.main_predictions,
        args.direct_predictions,
        args.baseline_predictions,
        args.processed_root,
        args.output_dir,
    )
    print(f"Exported {len(outputs)} numeric CSV tables to {args.output_dir}")


if __name__ == "__main__":
    main()
