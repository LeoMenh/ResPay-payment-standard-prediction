"""Validate experiment pairing and export numeric manuscript result tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from respay.baselines import METHODS

TASKS = ("DIP", "DRG")
THRESHOLDS = {"DIP": 0.45, "DRG": 0.25}
GATE_THRESHOLDS = {"DIP": 0.0, "DRG": 0.75}
PAIR_KEYS = ["task", "seed", "code"]
PREDICTION_COLUMNS = ["task", "seed", "code", "true", "prior", "prediction"]
PAIRING_ATOL = 1e-6


def _read(path: Path, sources: dict[str, str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction input does not exist: {path}")
    sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path, dtype={"code": str, "病种编码": str, "disease_code": str})
    return frame.loc[:, [column for column in frame if not column.startswith("Unnamed:")]]


def _validate_frame(frame: pd.DataFrame, label: str, *, method: bool = False) -> None:
    required = set(PREDICTION_COLUMNS)
    if method:
        required.add("method")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    identity = PAIR_KEYS + (["method"] if method else [])
    if frame.duplicated(identity).any():
        raise ValueError(f"{label} contains duplicate prediction identity rows")
    if frame[PAIR_KEYS].isna().any().any() or frame.code.astype(str).str.len().eq(0).any():
        raise ValueError(f"{label} contains an empty task, seed, or code")
    if set(frame.task.astype(str)) != set(TASKS):
        raise ValueError(f"{label} must contain both tasks {TASKS}")
    numeric = frame[["seed", "true", "prior", "prediction"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains non-finite seed or prediction values")
    if not np.equal(numeric[:, 0], np.floor(numeric[:, 0])).all():
        raise ValueError(f"{label} contains non-integer seed values")


def _check_pair(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    left = reference.set_index(PAIR_KEYS).sort_index()
    right = candidate.set_index(PAIR_KEYS).sort_index()
    if not left.index.equals(right.index):
        missing = left.index.difference(right.index).tolist()[:5]
        extra = right.index.difference(left.index).tolist()[:5]
        raise ValueError(
            f"{label} has mismatched task/seed/code pairs; missing={missing}, extra={extra}"
        )
    try:
        np.testing.assert_allclose(
            left[["true", "prior"]].to_numpy(dtype=float),
            right[["true", "prior"]].to_numpy(dtype=float),
            rtol=0,
            atol=PAIRING_ATOL,
        )
    except AssertionError as error:
        raise ValueError(
            f"{label} true targets and priors differ by more than {PAIRING_ATOL}"
        ) from error


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "R2": float(r2_score(truth, prediction)),
    }


def _summarize(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(keys, key_values))
        row["seeds"] = int(group.seed.nunique())
        for column in ("groups", "lr_groups", "prevalence"):
            if column in group:
                if group[column].nunique() != 1:
                    raise ValueError(f"{column} changes across seeds for {key}")
                row[column] = group[column].iloc[0]
        for value in values:
            row[value + "_mean"] = float(group[value].mean())
            row[value + "_std"] = float(group[value].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, name: str, output_dir: Path, outputs: list[Path]) -> None:
    path = output_dir / name
    frame.to_csv(path, index=False)
    outputs.append(path)


def _export_cohorts(
    canonical: pd.DataFrame,
    processed_root: Path,
    output_dir: Path,
    sources: dict[str, str],
    checks: list[str],
    outputs: list[Path],
) -> None:
    group_frames: list[pd.DataFrame] = []
    exposure_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []

    for task in TASKS:
        folder = processed_root / f"{task.lower()}_processed"
        metadata_path = folder / "split_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Processed metadata not found: {metadata_path}")
        sources[str(metadata_path.resolve())] = hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if task == "DIP":
            split_path = folder / "disease_split" / "test_disease_split.csv"
            raw = _read(split_path, sources)
            required = {"病种编码", "真实标准分值费用", "统一标准分值费用"}
            if required - set(raw.columns):
                raise ValueError(
                    "DIP processed test split missing columns: "
                    f"{sorted(required - set(raw.columns))}"
                )
            grouped = (
                raw.groupby("病种编码", sort=True)
                .agg(
                    true=("真实标准分值费用", "first"),
                    prior=("统一标准分值费用", "first"),
                    case_count=("病种编码", "size"),
                )
                .reset_index()
                .rename(columns={"病种编码": "code"})
            )
            grouped["exposure"] = grouped.case_count * (grouped.true - grouped.prior).abs()
            grouped["exposure_unit"] = "case-weighted standardized fee units"
            split = metadata["disease_split"]
            train_cases = int(split["train_rows"])
            test_cases = int(split["test_rows"])
            train_units = int(split["train_codes"])
            test_units = int(split["test_codes"])
            prediction_units = {"train": train_units, "test": test_units}
        else:
            summary_path = folder / "disease_summary.csv"
            summary = _read(summary_path, sources)
            summary = summary[summary.split == "test"].copy()
            required = {
                "disease_code",
                "case_count",
                "current_group_amount",
                "target_group_amount",
                "current_group_amount_raw",
                "target_group_amount_raw",
            }
            if required - set(summary.columns):
                raise ValueError(
                    "DRG disease summary missing columns: "
                    f"{sorted(required - set(summary.columns))}"
                )
            grouped = summary.rename(
                columns={
                    "disease_code": "code",
                    "target_group_amount": "true",
                    "current_group_amount": "prior",
                }
            )
            grouped["target_amount"] = grouped.target_group_amount_raw
            grouped["prior_amount"] = grouped.current_group_amount_raw
            grouped["count"] = grouped.case_count
            grouped["exposure"] = (
                grouped.case_count * (grouped.target_amount - grouped.prior_amount).abs()
            )
            grouped["exposure_unit"] = "RMB"
            split = metadata["disease_split"]
            rows = metadata["rows"]
            train_cases = int(rows["train_disease_split"])
            test_cases = int(rows["test_disease_split"])
            train_units = int(split["train_codes"])
            test_units = int(split["test_codes"])
            prediction_units = {
                part: int(rows[f"{part}_identity_split"]) for part in ("train", "test")
            }

        grouped["task"] = task
        grouped["realized_update"] = grouped.true - grouped.prior
        lower = 0.10 if task == "DIP" else 0.05
        grouped["stratum"] = pd.cut(
            grouped.realized_update.abs(),
            [-np.inf, lower, THRESHOLDS[task], np.inf],
            labels=["Low", "Medium", "High"],
            right=False,
        ).astype(str)
        grouped["high_label"] = (grouped.realized_update.abs() >= THRESHOLDS[task]).astype(int)

        forecast = canonical[canonical.task == task].drop_duplicates("code").set_index("code")
        cohort = grouped.set_index("code")
        if not cohort.index.sort_values().equals(forecast.index.sort_values()):
            raise ValueError(f"{task} processed test codes do not match prediction codes")
        aligned = cohort.loc[forecast.index]
        try:
            np.testing.assert_allclose(
                aligned[["true", "prior"]].to_numpy(dtype=float),
                forecast[["true", "prior"]].to_numpy(dtype=float),
                rtol=0,
                atol=PAIRING_ATOL,
            )
        except AssertionError as error:
            raise ValueError(f"{task} processed cohort targets do not match predictions") from error

        for stratum in ("Low", "Medium", "High"):
            subset = grouped[grouped.stratum == stratum]
            exposure_rows.append(
                {
                    "task": task,
                    "stratum": stratum,
                    "groups": len(subset),
                    "cases": int(subset.case_count.sum()),
                    "group_percent": 100 * len(subset) / len(grouped),
                    "case_percent": 100 * subset.case_count.sum() / grouped.case_count.sum(),
                    "exposure": float(subset.exposure.sum()),
                    "exposure_percent": (
                        100 * subset.exposure.sum() / grouped.exposure.sum()
                        if grouped.exposure.sum()
                        else np.nan
                    ),
                    "exposure_unit": grouped.exposure_unit.iloc[0],
                }
            )
        group_frames.append(grouped)
        cohort_rows.extend(
            [
                {
                    "task": task,
                    "partition": "train",
                    "groups": train_units,
                    "cases": train_cases,
                    "prediction_units": prediction_units["train"],
                },
                {
                    "task": task,
                    "partition": "test",
                    "groups": test_units,
                    "cases": test_cases,
                    "prediction_units": prediction_units["test"],
                },
            ]
        )
        checks.append(
            f"{task} processed held-out codes, true targets, and priors match predictions"
        )

    group_columns = [
        "task",
        "code",
        "true",
        "prior",
        "case_count",
        "realized_update",
        "stratum",
        "high_label",
        "exposure",
        "exposure_unit",
    ]
    optional = ["prior_amount", "target_amount", "count"]
    group_table = pd.concat(group_frames, ignore_index=True)
    _write(
        group_table[[column for column in group_columns + optional if column in group_table]],
        "figure1_group_updates_and_exposure.csv",
        output_dir,
        outputs,
    )
    _write(pd.DataFrame(exposure_rows), "figure1_exposure_summary.csv", output_dir, outputs)
    _write(pd.DataFrame(cohort_rows), "figure1_cohort_summary.csv", output_dir, outputs)


def _export_threshold_sensitivity(
    main: pd.DataFrame, output_dir: Path, outputs: list[Path]
) -> None:
    """Evaluate frozen predictions at all 21 inclusive score thresholds."""
    rows = []
    for (task, seed), group in main.groupby(["task", "seed"], sort=True):
        labels = (group.true - group.prior).abs().ge(THRESHOLDS[task])
        for threshold in np.round(np.linspace(0, 1, 21), 2):
            selected = group.lr_probability.ge(threshold)
            prediction = np.where(selected, group.prediction, group.prior)
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "threshold": threshold,
                    "groups": len(group),
                    "selected_groups": int(selected.sum()),
                    "selected_fraction": float(selected.mean()),
                    "is_chosen": bool(threshold == GATE_THRESHOLDS[task]),
                    "LR_recall": float((selected & labels).sum() / labels.sum()),
                    **_metrics(group.true.to_numpy(), prediction),
                }
            )
    scan = pd.DataFrame(rows)
    summary = _summarize(
        scan,
        ["task", "threshold"],
        ["MAE", "RMSE", "R2", "selected_groups", "selected_fraction", "LR_recall"],
    )
    summary["is_chosen"] = summary.threshold.eq(summary.task.map(GATE_THRESHOLDS))
    _write(scan, "gate_threshold_sensitivity_by_seed.csv", output_dir, outputs)
    _write(summary, "gate_threshold_sensitivity_summary.csv", output_dir, outputs)
    _write(scan[scan.is_chosen].copy(), "gate_by_seed.csv", output_dir, outputs)
    _write(summary[summary.is_chosen].copy(), "gate_summary.csv", output_dir, outputs)


def analyze_results(
    main_predictions: str | Path,
    direct_predictions: str | Path,
    baseline_predictions: str | Path,
    processed_root: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Validate paired predictions, then write numeric tables and source hashes."""
    output_dir = Path(output_dir)
    processed_root = Path(processed_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}
    checks: list[str] = []
    outputs: list[Path] = []

    main = _read(Path(main_predictions), sources)
    direct = _read(Path(direct_predictions), sources)
    baseline = _read(Path(baseline_predictions), sources)
    _validate_frame(main, "ResPay residual predictions")
    _validate_frame(direct, "Direct-target predictions")
    _validate_frame(baseline, "Baseline predictions", method=True)

    _check_pair(main, direct, "Direct-target predictions")
    checks.append(
        "Direct-target codes/seeds match exactly; targets/priors agree within pairing_atol"
    )
    methods = set(baseline.method.astype(str))
    if methods != set(METHODS):
        raise ValueError(f"Baseline methods must be exactly {METHODS}; found {sorted(methods)}")
    for method in METHODS:
        _check_pair(main, baseline[baseline.method == method], f"{method} baseline predictions")
        checks.append(
            f"{method} codes/seeds match exactly; targets/priors agree within pairing_atol"
        )

    if "lr_probability" not in main or "gated_prediction" not in main:
        raise ValueError("ResPay predictions require lr_probability and gated_prediction")
    if "lr_probability" not in direct or "gated_prediction" not in direct:
        raise ValueError("Direct-target predictions require lr_probability and gated_prediction")
    for label, frame in (("ResPay", main), ("Direct-target", direct)):
        probabilities = frame.lr_probability.to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or not np.logical_and(probabilities >= 0, probabilities <= 1).all()
        ):
            raise ValueError(f"{label} contains invalid LR probabilities")
        expected = np.where(
            probabilities >= frame.task.map(GATE_THRESHOLDS).to_numpy(),
            frame.prediction,
            frame.prior,
        )
        if not np.allclose(frame.gated_prediction, expected, atol=1e-10, rtol=0):
            raise ValueError(
                f"{label} gated predictions do not follow the inclusive task-specific "
                "threshold rule (DIP=0, DRG=0.75)"
            )

    seeds = sorted(main.seed.astype(int).unique().tolist())
    canonical = main[main.seed.astype(int) == seeds[0]].copy()
    expected_task_seeds = {(task, seed) for task in TASKS for seed in seeds}
    actual_task_seeds = {
        (str(task), int(seed))
        for task, seed in main[["task", "seed"]].drop_duplicates().itertuples(index=False)
    }
    if actual_task_seeds != expected_task_seeds:
        missing = sorted(expected_task_seeds - actual_task_seeds)
        extra = sorted(actual_task_seeds - expected_task_seeds)
        raise ValueError(
            "ResPay predictions have incomplete task/seed coverage; "
            f"missing={missing}, extra={extra}"
        )
    for seed in seeds:
        reference = canonical.copy()
        reference["seed"] = seed
        candidate = main[main.seed.astype(int) == seed]
        _check_pair(reference, candidate, f"Seed {seed} held-out cohort")
    checks.append("Every seed has exactly the canonical task/code/true/prior cohort")

    _export_cohorts(canonical, processed_root, output_dir, sources, checks, outputs)

    model_rows = main[["task", "seed", "code", "true", "prior", "gated_prediction"]].rename(
        columns={"gated_prediction": "prediction"}
    )
    model_rows["method"] = "ResPay"
    forecasts = pd.concat(
        [baseline[PREDICTION_COLUMNS + ["method"]], model_rows], ignore_index=True
    )
    forecasts = forecasts.sort_values(["task", "method", "seed", "code"]).reset_index(drop=True)
    _write(forecasts, "forecast_all_predictions.csv", output_dir, outputs)

    by_seed_rows: list[dict[str, object]] = []
    for (task, method, seed), group in forecasts.groupby(["task", "method", "seed"], sort=False):
        by_seed_rows.append(
            {
                "task": task,
                "method": method,
                "seed": int(seed),
                "groups": len(group),
                **_metrics(group.true.to_numpy(), group.prediction.to_numpy()),
            }
        )
    forecast_by_seed = pd.DataFrame(by_seed_rows)
    forecast_summary = _summarize(forecast_by_seed, ["task", "method"], ["MAE", "RMSE", "R2"])
    _write(forecast_by_seed, "forecast_all_by_seed.csv", output_dir, outputs)
    _write(forecast_summary, "forecast_all_summary.csv", output_dir, outputs)
    _export_threshold_sensitivity(main, output_dir, outputs)

    reduction_rows: list[dict[str, object]] = []
    paired_rows: list[pd.DataFrame] = []
    for task in TASKS:
        for method in METHODS:
            ours = (
                forecasts[(forecasts.task == task) & (forecasts.method == "ResPay")]
                .set_index(["seed", "code"])
                .sort_index()
            )
            other = (
                forecasts[(forecasts.task == task) & (forecasts.method == method)]
                .set_index(["seed", "code"])
                .loc[ours.index]
            )
            ours_metrics = forecast_by_seed[
                (forecast_by_seed.task == task) & (forecast_by_seed.method == "ResPay")
            ].set_index("seed")
            other_metrics = forecast_by_seed[
                (forecast_by_seed.task == task) & (forecast_by_seed.method == method)
            ].set_index("seed")
            for seed in seeds:
                row: dict[str, object] = {
                    "task": task,
                    "baseline": method,
                    "seed": seed,
                    "groups": int((ours.reset_index().seed == seed).sum()),
                }
                for metric in ("MAE", "RMSE"):
                    baseline_mean = float(other_metrics[metric].mean())
                    row[metric + "_reduction_pct"] = (
                        100
                        * (other_metrics.loc[seed, metric] - ours_metrics.loc[seed, metric])
                        / baseline_mean
                        if baseline_mean
                        else np.nan
                    )
                reduction_rows.append(row)
            pair = pd.DataFrame(
                {
                    "true": ours.true,
                    "prior": ours.prior,
                    "respay_absolute_error": (ours.prediction - ours.true).abs(),
                    "baseline_absolute_error": (other.prediction - other.true).abs(),
                }
            ).reset_index()
            pair["improvement"] = pair.baseline_absolute_error - pair.respay_absolute_error
            paired_rows.append(pair.assign(task=task, comparator=method))

    reductions = pd.DataFrame(reduction_rows)
    reduction_summary = _summarize(
        reductions, ["task", "baseline"], ["MAE_reduction_pct", "RMSE_reduction_pct"]
    )
    _write(reductions, "forecast_all_reductions_by_seed.csv", output_dir, outputs)
    _write(reduction_summary, "forecast_all_reductions_summary.csv", output_dir, outputs)
    paired = pd.concat(paired_rows, ignore_index=True)
    _write(paired, "forecast_all_paired_errors_by_seed.csv", output_dir, outputs)
    paired_summary = _summarize(
        paired,
        ["task", "comparator", "code"],
        ["respay_absolute_error", "baseline_absolute_error", "improvement"],
    )
    paired_summary = paired_summary.merge(
        canonical[["task", "code", "true", "prior"]],
        on=["task", "code"],
        validate="many_to_one",
    )
    paired_summary["outcome"] = np.select(
        [paired_summary.improvement_mean > 1e-6, paired_summary.improvement_mean < -1e-6],
        ["win", "loss"],
        default="tie",
    )
    _write(paired_summary, "forecast_all_paired_errors_summary.csv", output_dir, outputs)

    identification_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    score_frames = []
    for method in METHODS:
        frame = baseline[baseline.method == method].copy()
        frame["score"] = (
            0.0 if method == "Current standard" else (frame.prediction - frame.prior).abs()
        )
        frame["method"] = method
        score_frames.append(frame)
    frame = main.copy()
    frame["method"] = "ResPay"
    frame["score"] = frame.lr_probability
    score_frames.append(frame)
    for frame in score_frames:
        for (task, method, seed), group in frame.groupby(["task", "method", "seed"], sort=False):
            labels = (group.true - group.prior).abs().ge(THRESHOLDS[task]).astype(int).to_numpy()
            scores = group.score.to_numpy(dtype=float)
            if np.unique(labels).size != 2:
                raise ValueError(f"{task} seed {seed} must contain both LR identification classes")
            identification_rows.append(
                {
                    "task": task,
                    "method": method,
                    "seed": int(seed),
                    "groups": len(group),
                    "lr_groups": int(labels.sum()),
                    "prevalence": float(labels.mean()),
                    "AP": float(average_precision_score(labels, scores)),
                    "ROC_AUC": float(roc_auc_score(labels, scores)),
                }
            )
            score_rows.extend(
                {
                    "task": task,
                    "method": method,
                    "seed": int(seed),
                    "code": code,
                    "lr_label": int(label),
                    "score": float(score),
                }
                for code, label, score in zip(group.code, labels, scores)
            )
            precision, recall, _ = precision_recall_curve(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)
            for curve, x_values, y_values in (("PR", recall, precision), ("ROC", fpr, tpr)):
                curve_rows.extend(
                    {
                        "task": task,
                        "method": method,
                        "seed": int(seed),
                        "curve": curve,
                        "vertex": index,
                        "x": float(x),
                        "y": float(y),
                    }
                    for index, (x, y) in enumerate(zip(x_values, y_values))
                )
    identification = pd.DataFrame(identification_rows)
    _write(identification, "identification_by_seed.csv", output_dir, outputs)
    _write(
        _summarize(identification, ["task", "method"], ["AP", "ROC_AUC"]),
        "identification_summary.csv",
        output_dir,
        outputs,
    )
    _write(pd.DataFrame(curve_rows), "pr_roc_curves.csv", output_dir, outputs)
    _write(pd.DataFrame(score_rows), "identification_scores.csv", output_dir, outputs)

    ablation_rows: list[dict[str, object]] = []
    for variant, frame in (("residual", main), ("direct_target", direct)):
        for (task, seed), group in frame.groupby(["task", "seed"], sort=False):
            ablation_rows.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "variant": variant,
                    "groups": len(group),
                    **_metrics(group.true.to_numpy(), group.gated_prediction.to_numpy()),
                }
            )
    ablation = pd.DataFrame(ablation_rows)
    _write(ablation, "ablation_all_by_seed.csv", output_dir, outputs)
    _write(
        _summarize(ablation, ["task", "variant"], ["MAE", "RMSE", "R2"]),
        "ablation_all_summary.csv",
        output_dir,
        outputs,
    )
    differences = []
    for (task, seed), group in ablation.groupby(["task", "seed"], sort=False):
        indexed = group.set_index("variant")
        for metric in ("MAE", "RMSE", "R2"):
            residual_value = float(indexed.loc["residual", metric])
            direct_value = float(indexed.loc["direct_target", metric])
            differences.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "metric": metric,
                    "full_value": residual_value,
                    "ablation_value": direct_value,
                    "deterioration": (
                        residual_value - direct_value
                        if metric == "R2"
                        else direct_value - residual_value
                    ),
                }
            )
    difference_frame = pd.DataFrame(differences)
    _write(difference_frame, "paired_direct_all_by_seed.csv", output_dir, outputs)
    difference_summary = _summarize(difference_frame, ["task", "metric"], ["deterioration"])
    worse = (
        difference_frame.assign(worse=difference_frame.deterioration > 0)
        .groupby(["task", "metric"])
        .worse.sum()
        .rename("seeds_worse")
    )
    difference_summary = difference_summary.join(worse, on=["task", "metric"])
    _write(difference_summary, "paired_direct_all_summary.csv", output_dir, outputs)

    provenance = {
        "tasks": list(TASKS),
        "seeds": seeds,
        "baseline_methods": list(METHODS),
        "gate": "Inclusive score >= task threshold",
        "gate_thresholds": GATE_THRESHOLDS,
        "lr_thresholds": THRESHOLDS,
        "reported_variability": "Sample standard deviation across seeds (ddof=1)",
        "pairing_atol": PAIRING_ATOL,
        "threshold_sensitivity": "Frozen scores and numerical predictions; thresholds 0:0.05:1",
        "source_sha256": sources,
        "verified_checks": checks,
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(outputs)
        },
    }
    provenance_path = output_dir / "source_data_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return outputs
