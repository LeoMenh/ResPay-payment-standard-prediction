"""Selected manuscript comparison methods and group-level prediction export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from respay.dataset import DEFAULT_EXPERIMENT_TASKS, DatasetBundle, load_dataset_bundle
from respay.paths import RESULTS_ROOT
from respay.trainer import TASK_LABELS, enrich_identity_units

METHODS = ("Current standard", "Ridge", "MLP", "XGB")


def build_regressor(method: str, seed: int):
    if method == "Ridge":
        return Ridge(alpha=1.0)
    if method == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=15,
        )
    if method == "XGB":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("XGB requires xgboost; install the project dependencies.") from error
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported baseline method: {method}")


def aggregate_group_feature_profile(
    x_case: np.ndarray, group_idx: np.ndarray, num_groups: int
) -> np.ndarray:
    """Summarize case features by group using mean, population std, and log count."""
    counts = np.bincount(group_idx, minlength=num_groups).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    sums = np.zeros((num_groups, x_case.shape[1]), dtype=np.float32)
    np.add.at(sums, group_idx, x_case.astype(np.float32))
    means = sums / counts[:, None]
    squared_sums = np.zeros_like(sums)
    centered = x_case.astype(np.float32) - means[group_idx]
    np.add.at(squared_sums, group_idx, centered * centered)
    stds = np.sqrt(np.maximum(squared_sums / counts[:, None], 1e-8))
    return np.concatenate([means, stds, np.log1p(counts)[:, None]], axis=1)


def build_baseline_features(
    bundle: DatasetBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    train_case = bundle.x_train_numeric.detach().cpu().numpy().astype(np.float32)
    test_case = bundle.x_test_numeric.detach().cpu().numpy().astype(np.float32)
    train_idx = bundle.train_group_idx.detach().cpu().numpy().astype(np.int64)
    test_idx = bundle.test_group_idx.detach().cpu().numpy().astype(np.int64)
    train_x = aggregate_group_feature_profile(train_case, train_idx, len(bundle.train_disease_df))
    test_x = aggregate_group_feature_profile(test_case, test_idx, len(bundle.test_disease_df))
    train_y = bundle.y_train.detach().cpu().numpy().astype(np.float32)
    test_y = bundle.y_test.detach().cpu().numpy().astype(np.float32)
    test_prior = None
    if bundle.y_train_prior is not None and bundle.y_test_prior is not None:
        train_prior = bundle.y_train_prior.detach().cpu().numpy().astype(np.float32)
        test_prior = bundle.y_test_prior.detach().cpu().numpy().astype(np.float32)
        train_x = np.concatenate([train_x, train_prior[:, None]], axis=1)
        test_x = np.concatenate([test_x, test_prior[:, None]], axis=1)
        train_y = train_y - train_prior
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_x),
        scaler.transform(test_x),
        train_y,
        test_y,
        test_prior,
    )


def predict_group_baseline(bundle: DatasetBundle, method: str, seed: int) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Unsupported baseline method: {method}")
    if method == "Current standard":
        if bundle.y_test_prior is None:
            raise ValueError(f"{bundle.config.name} has no prior column for Current standard")
        return bundle.y_test_prior.detach().cpu().numpy().astype(np.float32)
    train_x, test_x, train_y, _, test_prior = build_baseline_features(bundle)
    model = build_regressor(method, seed)
    model.fit(train_x, train_y)
    prediction = np.asarray(model.predict(test_x), dtype=np.float32)
    if test_prior is not None:
        prediction = test_prior + prediction
    return prediction.astype(np.float32)


def baseline_predictions(bundle: DatasetBundle, method: str, seed: int) -> pd.DataFrame:
    """Return one prediction row per held-out disease for a selected baseline."""
    values = predict_group_baseline(bundle, method, seed)
    if bundle.config.name == "dip_disease":
        frame = pd.DataFrame(
            {
                "code": bundle.test_disease_df[bundle.config.code_col].astype(str).to_numpy(),
                "true": bundle.y_test.detach().cpu().numpy().astype(float),
                "prior": bundle.y_test_prior.detach().cpu().numpy().astype(float),
                "prediction": values,
            }
        )
    elif bundle.config.name == "drg_identity_amount":
        units = enrich_identity_units(bundle.test_disease_df, bundle.test_df)
        units["weighted_prediction"] = values * units["identity_weight"]
        frame = (
            units.groupby("disease_code", sort=True, as_index=False)
            .agg(
                true=("target_group_amount", "first"),
                prior=("current_group_amount", "first"),
                prediction=("weighted_prediction", "sum"),
            )
            .rename(columns={"disease_code": "code"})
        )
        frame["code"] = frame["code"].astype(str)
    else:
        raise ValueError(f"Unknown task: {bundle.config.name}")
    return frame.assign(task=TASK_LABELS[bundle.config.name], seed=seed, method=method)


def run_baselines(
    tasks: tuple[str, ...] = DEFAULT_EXPERIMENT_TASKS,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    output_dir: str | Path = RESULTS_ROOT / "baselines",
    processed_root: str | Path | None = None,
) -> Path:
    """Run every selected method and save a single long-form prediction table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for task in tasks:
        bundle = (
            load_dataset_bundle(task)
            if processed_root is None
            else load_dataset_bundle(task, processed_root=processed_root)
        )
        for seed in seeds:
            for method in METHODS:
                frames.append(baseline_predictions(bundle, method, seed))
    predictions_path = output_dir / "predictions.csv"
    pd.concat(frames, ignore_index=True).to_csv(predictions_path, index=False)
    settings = {
        "seeds": list(seeds),
        "tasks": list(tasks),
        "methods": list(METHODS),
        "regressors": {
            method: build_regressor(method, seeds[0]).get_params() for method in METHODS[1:]
        },
    }
    (output_dir / "experiment_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    return predictions_path
