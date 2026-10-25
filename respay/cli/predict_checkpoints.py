"""Regenerate both model prediction files from trusted saved checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from respay.dataset import DEFAULT_EXPERIMENT_TASKS, load_dataset_bundle
from respay.paths import PROCESSED_DATA_ROOT, RESULTS_ROOT
from respay.trainer import GATE_THRESHOLDS, predict_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=RESULTS_ROOT / "respay")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "checkpoint_predictions")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique seed values")
    torch.set_num_threads(args.threads)
    bundles = {
        task: load_dataset_bundle(task, args.processed_root) for task in DEFAULT_EXPERIMENT_TASKS
    }
    hashes = {}
    for variant in ("residual", "direct_target"):
        frames = []
        for task, bundle in bundles.items():
            for seed in args.seeds:
                path = (
                    args.checkpoint_root
                    / variant
                    / task
                    / f"seed_{seed}"
                    / f"{task}_seed_{seed}_model.pt"
                )
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["config"]["seed"] != seed:
                    raise ValueError(f"Checkpoint seed does not match its location: {path}")
                if saved["use_residual_target"] != (variant == "residual"):
                    raise ValueError(
                        f"Checkpoint target variant does not match its location: {path}"
                    )
                frames.append(predict_checkpoint(path, bundle, torch.device(args.device)))
                hashes[str(path.relative_to(args.checkpoint_root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        folder = args.output_dir / variant
        folder.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
        print(f"Replayed {variant} predictions to {folder / 'predictions.csv'}", flush=True)
    report = {
        "checkpoint_sha256": hashes,
        "gate_thresholds": GATE_THRESHOLDS,
        "policy": "Current task gates override historical checkpoint gate metadata; no refitting",
        "seeds": args.seeds,
        "device": args.device,
        "torch_threads": args.threads,
    }
    (args.output_dir / "checkpoint_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
