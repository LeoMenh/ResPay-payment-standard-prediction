"""Prepare the fixed DIP 2023-to-2024 and DRG 1.1-to-2.0 group splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from respay.data.dip import DEFAULT_RAW_DATA_PATH, export_dip_datasets
from respay.data.drg import DEFAULT_RAW_DATA_DIR, export_drg_datasets
from respay.dataset import DEFAULT_EXPERIMENT_TASKS
from respay.paths import PROCESSED_DATA_ROOT


def main() -> None:
    """Export the selected cohorts from explicit or repository-relative raw paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_EXPERIMENT_TASKS,
        default=DEFAULT_EXPERIMENT_TASKS,
    )
    parser.add_argument("--output-root", type=Path, default=PROCESSED_DATA_ROOT)
    parser.add_argument("--dip-raw-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--drg-raw-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    args = parser.parse_args()
    for task in args.tasks:
        try:
            if task == "dip_disease":
                outputs = export_dip_datasets(args.dip_raw_path, args.output_root / "dip_processed")
            else:
                outputs = export_drg_datasets(args.output_root / "drg_processed", args.drg_raw_dir)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        for name, path in outputs.items():
            print(f"{task} {name}: {path}")


if __name__ == "__main__":
    main()
