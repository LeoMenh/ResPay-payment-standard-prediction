"""Training and group-level inference for ResPay and its direct-target ablation."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from respay.dataset import DatasetBundle, load_dataset_bundle
from respay.model import ResPay, TrainingConfig, regression_loss, set_seed

THRESHOLDS = {"dip_disease": 0.45, "drg_identity_amount": 0.25}
TASK_LABELS = {"dip_disease": "DIP", "drg_identity_amount": "DRG"}
GATE_THRESHOLDS = {"dip_disease": 0.0, "drg_identity_amount": 0.75}


def apply_gate(
    prediction: np.ndarray, prior: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Apply the inclusive score threshold after any disease-level aggregation."""
    return np.where(probability >= threshold, prediction, prior)


def enrich_identity_units(unit_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        source_df.groupby("identity_code", sort=True, dropna=False)
        .agg(
            disease_code=("disease_code", "first"),
            identity_type=("identity_type", "first"),
            identity_weight=("identity_weight", "first"),
            target_group_amount=("target_group_amount", "first"),
            current_group_amount=("current_group_amount", "first"),
        )
        .reset_index()
    )
    return unit_df.merge(metadata, on="identity_code", how="left", validate="one_to_one")


def aggregate_identity_by_disease(
    values: torch.Tensor, unit_df: pd.DataFrame, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    disease_codes = unit_df["disease_code"].astype(str).tolist()
    unique_codes = sorted(set(disease_codes))
    code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}
    disease_idx = torch.tensor(
        [code_to_idx[code] for code in disease_codes], dtype=torch.long, device=device
    )
    weights = torch.tensor(
        unit_df["identity_weight"].astype(np.float32).to_numpy(),
        dtype=values.dtype,
        device=device,
    )
    weighted_sum = torch.zeros(len(unique_codes), dtype=values.dtype, device=device)
    weight_sum = torch.zeros_like(weighted_sum)
    weighted_sum.index_add_(0, disease_idx, values * weights)
    weight_sum.index_add_(0, disease_idx, weights)
    return weighted_sum / weight_sum.clamp_min(1e-8), disease_idx, unique_codes


def disease_targets_from_units(
    unit_df: pd.DataFrame, target_col: str, unique_codes: list[str], device: torch.device
) -> torch.Tensor:
    target_map = unit_df.groupby("disease_code", dropna=False)[target_col].first().to_dict()
    return torch.tensor(
        [float(target_map[code]) for code in unique_codes], dtype=torch.float32, device=device
    )


def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    positives = float(labels.sum().item())
    negatives = float(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training split needs both LR and non-LR diseases")
    positive_weight = negatives / positives
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=logits.new_tensor(positive_weight)
    )
    return loss, positive_weight


def predict_groups(
    model: ResPay,
    bundle: DatasetBundle,
    device: torch.device,
    residual_mean: float,
    residual_std: float,
    use_residual_target: bool,
) -> pd.DataFrame:
    """Return standard-scale predictions, aggregating DRG logits before sigmoid."""
    task = bundle.config.name
    model.eval()
    with torch.no_grad():
        output = model(
            bundle.x_test_numeric.to(device),
            bundle.test_group_idx.to(device),
            len(bundle.y_test),
        )
        unit_prediction = output.prediction * residual_std + residual_mean
        if use_residual_target:
            unit_prediction = unit_prediction + bundle.y_test_prior.to(device)
        if task == "dip_disease":
            codes = bundle.test_disease_df[bundle.config.code_col].astype(str).tolist()
            prediction = unit_prediction
            probability = torch.sigmoid(output.lr_logit)
            true = bundle.y_test.to(device)
            prior = bundle.y_test_prior.to(device)
        elif task == "drg_identity_amount":
            units = enrich_identity_units(bundle.test_disease_df, bundle.test_df)
            prediction, _, codes = aggregate_identity_by_disease(unit_prediction, units, device)
            disease_logits, _, check_codes = aggregate_identity_by_disease(
                output.lr_logit, units, device
            )
            if check_codes != codes:
                raise RuntimeError("DRG test disease order changed")
            # Insurance shares apply to logits, then sigmoid produces the disease score.
            probability = torch.sigmoid(disease_logits)
            true = disease_targets_from_units(units, "target_group_amount", codes, device)
            prior = disease_targets_from_units(units, "current_group_amount", codes, device)
        else:
            raise ValueError(f"Unknown task: {task}")
        arrays = [
            tensor.detach().cpu().numpy().astype(float)
            for tensor in (prediction, probability, true, prior)
        ]
    frame = pd.DataFrame(
        {
            "code": codes,
            "true": arrays[2],
            "prior": arrays[3],
            "prediction": arrays[0],
            "lr_probability": arrays[1],
        }
    )
    frame["gated_prediction"] = apply_gate(
        frame["prediction"].to_numpy(),
        frame["prior"].to_numpy(),
        frame["lr_probability"].to_numpy(),
        threshold=GATE_THRESHOLDS[task],
    )
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    if frame["code"].duplicated().any() or not np.isfinite(numeric).all():
        raise RuntimeError(f"Invalid disease-level predictions for {task}")
    return frame


def predict_checkpoint(
    checkpoint_path: str | Path, bundle: DatasetBundle, device: torch.device
) -> pd.DataFrame:
    """Replay a trusted checkpoint using the current task-specific inference gate."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved["feature_cols"] != bundle.feature_cols:
        raise ValueError("Checkpoint feature names/order do not match the processed dataset")
    task = bundle.config.name
    if saved["lr_threshold"] != THRESHOLDS[task]:
        raise ValueError("Checkpoint LR label boundary does not match the task")
    model = ResPay(len(bundle.feature_cols), saved["config"]["hidden_dim"]).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    frame = predict_groups(
        model,
        bundle,
        device,
        saved["residual_mean"],
        saved["residual_std"],
        saved["use_residual_target"],
    )
    return frame.assign(task=TASK_LABELS[task], seed=saved["config"]["seed"])


def train_one(
    task: str,
    seed: int,
    epochs: int,
    device: torch.device,
    output_dir: str | Path,
    use_residual_target: bool = True,
    *,
    bundle: DatasetBundle | None = None,
    processed_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit one seed/task and save predictions, loss history, and a CPU checkpoint."""
    if task not in THRESHOLDS:
        raise ValueError(f"Unknown task {task!r}; choose from {tuple(THRESHOLDS)}")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if bundle is None:
        bundle = (
            load_dataset_bundle(task)
            if processed_root is None
            else load_dataset_bundle(task, processed_root=processed_root)
        )
    if bundle.config.name != task:
        raise ValueError(f"Bundle task {bundle.config.name!r} does not match {task!r}")

    config = TrainingConfig(
        seed=seed,
        epochs=epochs,
        use_residual_target=use_residual_target,
        lambda_group_consistency=1.0 if task == "drg_identity_amount" else 0.0,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = ResPay(bundle.x_train_numeric.shape[1], config.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    x_train = bundle.x_train_numeric.to(device)
    group_idx = bundle.train_group_idx.to(device)
    y_train = bundle.y_train.to(device)
    prior_train = bundle.y_train_prior.to(device)
    unit_target = y_train - prior_train if use_residual_target else y_train
    case_true = torch.tensor(
        bundle.train_df[bundle.config.target_col].to_numpy(np.float32), device=device
    )
    case_prior = torch.tensor(
        bundle.train_df[bundle.config.prior_col].to_numpy(np.float32), device=device
    )
    case_target = case_true - case_prior if use_residual_target else case_true

    if task == "dip_disease":
        train_codes = bundle.train_disease_df[bundle.config.code_col].astype(str).tolist()
        true_disease, prior_disease = y_train, prior_train
        residual_mean = float(unit_target.mean().item())
        residual_std = float(unit_target.std(unbiased=False).item())
        if residual_std < 1e-6:
            residual_std = 1.0
        train_units = None
    else:
        train_units = enrich_identity_units(bundle.train_disease_df, bundle.train_df)
        _, _, train_codes = aggregate_identity_by_disease(unit_target.detach(), train_units, device)
        true_disease = disease_targets_from_units(
            train_units, "target_group_amount", train_codes, device
        )
        prior_disease = disease_targets_from_units(
            train_units, "current_group_amount", train_codes, device
        )
        residual_mean, residual_std = 0.0, 1.0

    labels = ((true_disease - prior_disease).abs() >= THRESHOLDS[task]).float()
    if not 0 < int(labels.sum().item()) < len(labels):
        raise ValueError(f"{task}: training data has only one LR class")
    normalized_target = (unit_target - residual_mean) / residual_std
    normalized_case_target = (case_target - residual_mean) / residual_std
    if train_units is not None:
        disease_target = true_disease - prior_disease if use_residual_target else true_disease

    loss_history: list[dict[str, float | int]] = []
    positive_weight = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train, group_idx, len(y_train))
        unit_loss = regression_loss(output.prediction, normalized_target, config.mse_weight)
        case_loss = nn.functional.smooth_l1_loss(
            output.case_prediction, normalized_case_target, beta=config.huber_beta
        )
        if train_units is None:
            disease_logits = output.lr_logit
            consistency_loss = output.prediction.new_zeros(())
        else:
            disease_logits, _, check_codes = aggregate_identity_by_disease(
                output.lr_logit, train_units, device
            )
            if check_codes != train_codes:
                raise RuntimeError("DRG training disease order changed")
            disease_prediction, _, _ = aggregate_identity_by_disease(
                output.prediction, train_units, device
            )
            consistency_loss = nn.functional.smooth_l1_loss(
                disease_prediction, disease_target, beta=config.huber_beta
            )
        classification_loss, positive_weight = balanced_bce(disease_logits, labels)
        total_loss = (
            unit_loss
            + config.lambda_case * case_loss
            + config.lambda_group_consistency * consistency_loss
            + config.lr_loss_weight * classification_loss
        )
        total_loss.backward()
        optimizer.step()
        loss_history.append(
            {
                "epoch": epoch + 1,
                "regression_loss": float(unit_loss.item()),
                "record_loss": float(case_loss.item()),
                "aggregation_loss": float(consistency_loss.item()),
                "classification_loss": float(classification_loss.item()),
                "total_loss": float(total_loss.item()),
            }
        )

    predictions = predict_groups(
        model, bundle, device, residual_mean, residual_std, use_residual_target
    )
    predictions["task"] = TASK_LABELS[task]
    predictions["seed"] = seed
    checkpoint_path = output_dir / f"{task}_seed_{seed}_model.pt"
    loss_path = output_dir / f"{task}_seed_{seed}_loss.csv"
    pd.DataFrame(loss_history).to_csv(loss_path, index=False)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "config": asdict(config),
            "feature_cols": bundle.feature_cols,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "lr_threshold": THRESHOLDS[task],
            "use_residual_target": use_residual_target,
            "gate_threshold": GATE_THRESHOLDS[task],
            "inference_policy": "lr_score_gate",
        },
        checkpoint_path,
    )
    metadata: dict[str, object] = {
        "task": TASK_LABELS[task],
        "seed": seed,
        "train_diseases": len(train_codes),
        "test_diseases": len(predictions),
        "train_lr_diseases": int(labels.sum().item()),
        "lr_threshold": THRESHOLDS[task],
        "lr_loss_weight": config.lr_loss_weight,
        "positive_class_weight": positive_weight,
        "decision_probability_threshold": GATE_THRESHOLDS[task],
        "inference_policy": "lr_score_gate",
        "checkpoint_path": str(checkpoint_path),
        "loss_path": str(loss_path),
        "regression_config": asdict(config),
        "architecture": "record_attention_and_unit_summaries",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "openpyxl")
            },
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
            "cuda_runtime": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    (output_dir / f"{task}_seed_{seed}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return predictions, metadata
