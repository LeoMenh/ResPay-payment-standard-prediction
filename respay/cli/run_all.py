"""Prepare cohorts, run selected experiments, and export numeric results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from respay.analysis import analyze_results
from respay.baselines import run_baselines
from respay.data.dip import DEFAULT_RAW_DATA_PATH, export_dip_datasets
from respay.data.drg import DEFAULT_RAW_DATA_DIR, export_drg_datasets
from respay.dataset import DEFAULT_EXPERIMENT_TASKS
from respay.paths import RESULTS_ROOT
from respay.trainer import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dip-raw-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--drg-raw-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    tasks = DEFAULT_EXPERIMENT_TASKS
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique seed values")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cpu":
        torch.use_deterministic_algorithms(True)

    output_root = args.output_root
    processed_root = output_root / "processed"
    task_exports = {
        "dip_disease": lambda: export_dip_datasets(
            args.dip_raw_path, processed_root / "dip_processed"
        ),
        "drg_identity_amount": lambda: export_drg_datasets(
            processed_root / "drg_processed", args.drg_raw_dir
        ),
    }
    for task in tasks:
        task_exports[task]()

    experiment_root = output_root / "respay"
    prediction_paths: dict[str, Path] = {}
    for variant, residual in (("residual", True), ("direct_target", False)):
        frames = []
        for task in tasks:
            for seed in args.seeds:
                prediction, _ = train_one(
                    task,
                    seed,
                    args.epochs,
                    torch.device(args.device),
                    experiment_root / variant / task / f"seed_{seed}",
                    use_residual_target=residual,
                    processed_root=processed_root,
                )
                frames.append(prediction)
        variant_dir = experiment_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = variant_dir / "predictions.csv"
        pd.concat(frames, ignore_index=True).to_csv(prediction_path, index=False)
        prediction_paths[variant] = prediction_path

    baseline_path = run_baselines(
        tuple(tasks), tuple(args.seeds), output_root / "baselines", processed_root
    )
    analyze_results(
        prediction_paths["residual"],
        prediction_paths["direct_target"],
        baseline_path,
        processed_root,
        output_root / "tables",
    )


if __name__ == "__main__":
    main()
