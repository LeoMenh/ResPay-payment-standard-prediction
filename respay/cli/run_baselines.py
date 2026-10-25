"""Run Current standard, Ridge, MLP, and XGB comparison experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from respay.baselines import run_baselines
from respay.dataset import DEFAULT_EXPERIMENT_TASKS
from respay.paths import RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "baselines")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_EXPERIMENT_TASKS,
        default=DEFAULT_EXPERIMENT_TASKS,
    )
    args = parser.parse_args()
    run_baselines(tuple(args.tasks), tuple(args.seeds), args.output_dir, args.processed_root)


if __name__ == "__main__":
    main()
