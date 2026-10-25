"""Train final ResPay experiments or the direct-target ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from respay.dataset import DEFAULT_EXPERIMENT_TASKS
from respay.paths import RESULTS_ROOT
from respay.trainer import train_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "respay")
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_EXPERIMENT_TASKS,
        default=DEFAULT_EXPERIMENT_TASKS,
    )
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--direct-target", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique seed values")
    torch.set_num_threads(args.threads)
    if args.device == "cpu":
        torch.use_deterministic_algorithms(True)

    outputs = []
    experiment = "direct_target" if args.direct_target else "residual"
    for task in args.tasks:
        for seed in args.seeds:
            predictions, _ = train_one(
                task,
                seed,
                args.epochs,
                torch.device(args.device),
                args.output_dir / experiment / task / f"seed_{seed}",
                use_residual_target=not args.direct_target,
                processed_root=args.processed_root,
            )
            outputs.append(predictions)
    output_dir = args.output_dir / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
