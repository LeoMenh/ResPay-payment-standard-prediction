"""Synthetic contracts for numeric result exports and provenance."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from respay.analysis import analyze_results
from respay.baselines import METHODS

SEEDS = (42, 43, 44)
TASKS = ("DIP", "DRG")


def _predictions(tmp_path):
    processed = tmp_path / "processed"
    rows = []
    direct_rows = []
    baseline_rows = []
    for task in TASKS:
        count = 4
        task_codes = [f"{task}-{index}" for index in range(count)]
        truth = np.linspace(1.0, 4.0, count)
        prior = truth + np.array([0.0, 0.2, 0.6, -0.8])
        for seed_index, seed in enumerate(SEEDS):
            for index, code in enumerate(task_codes):
                base = dict(task=task, seed=seed, code=code, true=truth[index], prior=prior[index])
                prediction = truth[index] + 0.1 * (seed_index + 1)
                probability = [0.2, 0.5, 0.8, 0.9][index]
                threshold = 0.0 if task == "DIP" else 0.75
                gated = prediction if probability >= threshold else prior[index]
                rows.append(
                    {
                        **base,
                        "prediction": prediction,
                        "lr_probability": probability,
                        "gated_prediction": gated,
                    }
                )
                direct_prediction = truth[index] + 0.2 * (seed_index + 1)
                direct_rows.append(
                    {
                        **base,
                        "prediction": direct_prediction,
                        "lr_probability": probability,
                        "gated_prediction": (
                            direct_prediction if probability >= threshold else prior[index]
                        ),
                    }
                )
                for method_index, method in enumerate(METHODS):
                    baseline_rows.append(
                        {
                            **base,
                            "prediction": truth[index] + 0.3 * (method_index + 1),
                            "method": method,
                        }
                    )
    main = pd.DataFrame(rows)
    direct = pd.DataFrame(direct_rows)
    baseline = pd.DataFrame(baseline_rows)
    main_path, direct_path, baseline_path = (
        tmp_path / name for name in ("main.csv", "direct.csv", "baseline.csv")
    )
    main.to_csv(main_path, index=False)
    direct.to_csv(direct_path, index=False)
    baseline.to_csv(baseline_path, index=False)
    _write_processed(processed)
    return main_path, direct_path, baseline_path, processed, main, direct, baseline


def _write_processed(root):
    dip = root / "dip_processed"
    (dip / "disease_split").mkdir(parents=True)
    pd.DataFrame(
        {
            "病种编码": ["DIP-0", "DIP-1", "DIP-2", "DIP-3"],
            "真实标准分值费用": [1.0, 2.0, 3.0, 4.0],
            "统一标准分值费用": [1.0, 2.2, 3.6, 3.2],
        }
    ).to_csv(dip / "disease_split/test_disease_split.csv", index=False)
    pd.DataFrame(
        {"disease_split": {"train_codes": 6, "test_codes": 4, "train_rows": 12, "test_rows": 4}}
    ).to_json(dip / "split_metadata.json", indent=2)
    drg = root / "drg_processed"
    drg.mkdir(parents=True)
    pd.DataFrame(
        {
            "disease_code": ["DRG-0", "DRG-1", "DRG-2", "DRG-3"],
            "split": ["test"] * 4,
            "case_count": [1, 2, 3, 4],
            "current_group_amount_raw": [100, 200, 300, 400],
            "target_group_amount_raw": [100, 220, 360, 320],
            "current_group_amount": [1.0, 2.2, 3.6, 3.2],
            "target_group_amount": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(drg / "disease_summary.csv", index=False)
    (drg / "split_metadata.json").write_text(
        json.dumps(
            {
                "rows": {
                    "train_disease_split": 20,
                    "test_disease_split": 10,
                    "train_identity_split": 15,
                    "test_identity_split": 8,
                },
                "disease_split": {"train_codes": 6, "test_codes": 4},
            }
        )
    )


def test_analyze_exports_numeric_tables_and_sample_standard_deviation(tmp_path):
    main, direct, baseline, processed, main_frame, _, _ = _predictions(tmp_path)
    output = tmp_path / "results"
    outputs = analyze_results(main, direct, baseline, processed, output)

    names = {path.name for path in outputs}
    expected = {
        "figure1_group_updates_and_exposure.csv",
        "figure1_exposure_summary.csv",
        "figure1_cohort_summary.csv",
        "forecast_all_predictions.csv",
        "forecast_all_by_seed.csv",
        "forecast_all_summary.csv",
        "forecast_all_reductions_by_seed.csv",
        "forecast_all_reductions_summary.csv",
        "forecast_all_paired_errors_by_seed.csv",
        "forecast_all_paired_errors_summary.csv",
        "identification_by_seed.csv",
        "identification_summary.csv",
        "pr_roc_curves.csv",
        "identification_scores.csv",
        "ablation_all_by_seed.csv",
        "ablation_all_summary.csv",
        "paired_direct_all_by_seed.csv",
        "paired_direct_all_summary.csv",
        "gate_by_seed.csv",
        "gate_summary.csv",
        "gate_threshold_sensitivity_by_seed.csv",
        "gate_threshold_sensitivity_summary.csv",
    }
    assert names == expected
    by_seed = pd.read_csv(output / "forecast_all_by_seed.csv")
    observed = by_seed[(by_seed.task == "DIP") & (by_seed.method == "ResPay")].MAE
    summary = pd.read_csv(output / "forecast_all_summary.csv")
    actual = summary[(summary.task == "DIP") & (summary.method == "ResPay")].iloc[0]
    assert actual.MAE_std == pytest.approx(observed.std(ddof=1))
    assert len(main_frame) == 24
    provenance_path = output / "source_data_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert (
        provenance["source_sha256"][str(main.resolve())]
        == hashlib.sha256(main.read_bytes()).hexdigest()
    )
    assert set(provenance["output_sha256"]) == expected
    assert len(provenance["verified_checks"]) > 0


@pytest.mark.parametrize("field", ["true", "prior"])
@pytest.mark.parametrize("difference", [2e-6, 1.0])
def test_analyze_rejects_mismatched_targets_across_methods(tmp_path, field, difference):
    main, direct, baseline, processed, _, _, _ = _predictions(tmp_path)
    frame = pd.read_csv(baseline)
    frame.loc[0, field] += difference
    frame.to_csv(baseline, index=False)
    with pytest.raises(ValueError, match="pair|match|target"):
        analyze_results(main, direct, baseline, processed, tmp_path / "out")


def test_analyze_rejects_missing_method_rows(tmp_path):
    main, direct, baseline, processed, _, _, _ = _predictions(tmp_path)
    frame = pd.read_csv(baseline)
    frame = frame[frame.method != "MLP"]
    frame.to_csv(baseline, index=False)
    with pytest.raises(ValueError, match="method|missing|pair"):
        analyze_results(main, direct, baseline, processed, tmp_path / "out")


def test_analyze_rejects_duplicate_prediction_identity(tmp_path):
    main, direct, baseline, processed, _, _, _ = _predictions(tmp_path)
    frame = pd.read_csv(main)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(main, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_results(main, direct, baseline, processed, tmp_path / "out")


def test_analyze_rejects_seed_with_changed_cohort_identity(tmp_path):
    main, direct, baseline, processed, _, _, _ = _predictions(tmp_path)
    for path in (main, direct, baseline):
        frame = pd.read_csv(path)
        condition = (frame.task == "DIP") & (frame.seed == 43) & (frame.code == "DIP-0")
        frame.loc[condition, "code"] = "DIP-4"
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="seed|cohort|pair|codes"):
        analyze_results(main, direct, baseline, processed, tmp_path / "out")


def test_analyze_rejects_missing_task_seed_combination(tmp_path):
    main, direct, baseline, processed, _, _, _ = _predictions(tmp_path)
    for path in (main, direct, baseline):
        frame = pd.read_csv(path)
        mask = (frame.task == "DRG") & (frame.seed == 44)
        frame = frame[~mask]
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="seed|combination|coverage"):
        analyze_results(main, direct, baseline, processed, tmp_path / "out")


def test_identification_scores_follow_method_definitions(tmp_path):
    main, direct, baseline, processed, main_frame, _, baseline_frame = _predictions(tmp_path)
    output = tmp_path / "score-results"
    analyze_results(main, direct, baseline, processed, output)
    scores = pd.read_csv(output / "identification_scores.csv")
    respay = scores[scores.method == "ResPay"].merge(
        main_frame[["task", "seed", "code", "lr_probability"]],
        on=["task", "seed", "code"],
        validate="one_to_one",
    )
    np.testing.assert_allclose(respay.score, respay.lr_probability)
    for method in METHODS:
        actual = scores[scores.method == method].merge(
            baseline_frame[baseline_frame.method == method],
            on=["task", "seed", "code"],
            suffixes=("_score", "_prediction"),
            validate="one_to_one",
        )
        expected = 0.0 if method == "Current standard" else (actual.prediction - actual.prior).abs()
        np.testing.assert_allclose(actual.score, expected)


def test_analysis_cli_runs_end_to_end_on_synthetic_predictions(tmp_path):
    main, direct, baseline, processed, _, _, _ = _predictions(tmp_path)
    output = tmp_path / "cli-results"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "respay.cli.analyze_results",
            "--main-predictions",
            str(main),
            "--direct-predictions",
            str(direct),
            "--baseline-predictions",
            str(baseline),
            "--processed-root",
            str(processed),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Exported 22 numeric CSV tables" in result.stdout
    provenance = json.loads((output / "source_data_provenance.json").read_text())
    assert provenance["seeds"] == list(SEEDS)


def test_analysis_accepts_float32_target_roundoff_but_preserves_method_targets(tmp_path):
    main, direct, baseline, processed, _, _, baseline_frame = _predictions(tmp_path)
    baseline_frame[["true", "prior"]] = (
        baseline_frame[["true", "prior"]].astype(np.float32).astype(float)
    )
    baseline_frame.to_csv(baseline, index=False)
    output = tmp_path / "out"
    analyze_results(main, direct, baseline, processed, output)
    actual = pd.read_csv(output / "forecast_all_predictions.csv")
    ridge = actual[actual.method.eq("Ridge")].sort_values(["task", "seed", "code"])
    expected = baseline_frame[baseline_frame.method.eq("Ridge")].sort_values(
        ["task", "seed", "code"]
    )
    np.testing.assert_allclose(ridge.true, expected.true, atol=1e-14, rtol=0)
    np.testing.assert_allclose(ridge.prior, expected.prior, atol=1e-14, rtol=0)


def test_threshold_scan_uses_frozen_scores_inclusive_endpoints_and_selected_policy(tmp_path):
    main, direct, baseline, processed, _, _, _ = _predictions(tmp_path)
    # A saturated score must still pass the inclusive gate at threshold one.
    frame = pd.read_csv(main)
    frame.loc[frame.code.eq("DIP-3"), "lr_probability"] = 1.0
    frame.to_csv(main, index=False)
    output = tmp_path / "out"
    analyze_results(main, direct, baseline, processed, output)
    scan = pd.read_csv(output / "gate_threshold_sensitivity_by_seed.csv")
    assert len(scan) == 126
    assert scan.groupby(["task", "seed"]).size().eq(21).all()
    assert scan[scan.threshold.eq(0)].selected_groups.eq(4).all()
    dip = scan[scan.task.eq("DIP") & scan.seed.eq(42)].set_index("threshold")
    assert dip.loc[1.0, "selected_groups"] == 1
    assert dip.loc[1.0, "LR_recall"] == 0.5
    assert dip.loc[0.5, "selected_groups"] == 3
    assert dip.loc[0.5, "MAE"] == pytest.approx(0.075)
    chosen = pd.read_csv(output / "gate_by_seed.csv").set_index(["task", "seed"])
    forecast = pd.read_csv(output / "forecast_all_by_seed.csv")
    forecast = forecast[forecast.method.eq("ResPay")].set_index(["task", "seed"])
    np.testing.assert_allclose(
        chosen.loc[forecast.index, ["MAE", "RMSE", "R2"]], forecast[["MAE", "RMSE", "R2"]]
    )
