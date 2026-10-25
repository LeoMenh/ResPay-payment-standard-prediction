"""Synthetic checks for final cohorts and train-fitted dataset transforms."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        pytest.fail(f"Required standalone module missing: {error.name}")


def _dip_raw(path):
    features = [
        "药占比",
        "耗占比",
        "检查占比",
        "化验占比",
        "床位费占比",
        "诊察费占比",
        "护理费占比",
        "治疗费占比",
        "手术费占比",
        "其他费占比",
        "甲类药品费用",
        "甲类药品使用率",
        "乙类药品费用",
        "乙类药品使用率",
        "天数",
    ]
    rows = []
    for i, code in enumerate("ABCDE"):
        for year, amount in [(2023, 100 + i), (2024, 200 + i)]:
            for _ in range(2):
                rows.append(
                    {
                        "病种编码": code,
                        "出院日期": f"{year}-06-01",
                        "标准分值": amount / 10,
                        "标准分值费用": amount,
                        "姓名": "PRIVATE",
                        "病案号": "PRIVATE",
                        **{name: float(i + 1) for name in features},
                    }
                )
    rows.append({**rows[0], "标准分值费用": 999, "标准分值": 999})
    rows.append({**rows[0], "病种编码": "SOURCE_ONLY"})
    pd.DataFrame(rows).to_csv(path, index=False)


def _drg_raw(root):
    root.mkdir()
    cost_columns = [
        "标准费用",
        "总金额",
        "药品费用",
        "材料费用",
        "药耗费用",
        "检查费用",
        "检验费用",
        "医疗服务费用",
        "科内使用耗材金额",
        "科内治疗费",
        "手术费",
        "放疗费",
        "其他费用",
    ]
    for year in (2023, 2024, 2025):
        rows = []
        for i, code in enumerate("ABCDE", 1):
            for month in [1, 4] if year == 2025 else [6]:
                for identity in ["职工", "职工", "居民"]:
                    rows.append(
                        {
                            "分组编码": code,
                            "权重": i * (2 if month == 4 else 1),
                            "结算月份": f"{year}-{month:02d}",
                            "医保类型": identity,
                            "住院天数": i,
                            "姓名": "PRIVATE",
                            **{name: float(10 * i) for name in cost_columns},
                        }
                    )
        pd.DataFrame(rows).to_excel(root / f"{year}年1-12月国新分组数据.xlsx", index=False)
    pd.DataFrame(
        {
            "DRG编码": list("ABCDE"),
            "权重": [1, 2, 3, 4, 5],
            "职工": [100, 200, 300, 400, 500],
            "居民": [50, 100, 150, 200, 250],
            "省直": [80, 160, 240, 320, 400],
        }
    ).to_excel(root / "2024年1.1版权重费率表.xlsx", sheet_name="济南市 省内异地", index=False)
    pd.DataFrame(
        {
            "DRG": list("ABCDE"),
            "权重": [2, 4, 6, 8, 10],
            "标准费用": [200, 400, 600, 800, 1000],
            "标准费用.1": [100, 200, 300, 400, 500],
            "标准费用.2": [160, 320, 480, 640, 800],
        }
    ).to_excel(
        root / "2025年2.0版权重费率表费率（职工11800，居民9279.72）.xlsx",
        sheet_name="济南市 省异地",
        startrow=2,
        index=False,
    )


def test_dip_fixed_group_split_uses_2023_cases_and_next_year_modes(tmp_path):
    dip = _module("respay.data.dip")
    raw = tmp_path / "cohort.csv"
    _dip_raw(raw)
    tables = dip.build_disease_split_tables(dip.build_processed_dip_table(raw))
    assert tables["train_codes"]["病种编码"].tolist() == ["B", "C", "D", "E"]
    assert tables["test_codes"]["病种编码"].tolist() == ["A"]
    test = tables["test_disease_split"]
    assert len(test) == 3
    assert test["年份"].eq(2023).all()
    assert test["统一标准分值费用"].eq(100).all()
    assert test["真实标准分值费用"].eq(200).all()


@pytest.mark.parametrize("scaled", [True, False])
def test_bundle_imputes_and_scales_from_training_rows_only(scaled):
    dataset = _module("respay.dataset")
    config = dataset.DatasetConfig(
        "synthetic",
        Path("train"),
        Path("test"),
        "target",
        "code",
        ["x", "empty"],
        "prior",
        scale_features_in_loader=scaled,
    )
    train = pd.DataFrame(
        {
            "code": ["B", "A", "A"],
            "target": [8, 4, 4],
            "prior": [6, 3, 3],
            "x": [2.0, 4.0, np.nan],
            "empty": [np.nan] * 3,
        }
    )
    test = pd.DataFrame(
        {
            "code": ["Z", "Y"],
            "target": [10, 20],
            "prior": [9, 19],
            "x": [1000.0, np.nan],
            "empty": [np.nan, 100.0],
        }
    )
    bundle = dataset.build_dataset_bundle_from_frames(config, train, test)
    assert bundle.train_df["x"].tolist() == [2, 4, 3]
    assert bundle.test_df["x"].tolist() == [1000, 3]
    assert bundle.train_group_idx.tolist() == [1, 0, 0]
    assert bundle.y_train.tolist() == [4, 8]
    assert bundle.y_train_prior.tolist() == [3, 6]
    assert bundle.y_test.tolist() == [20, 10]
    assert np.isnan(train.loc[2, "x"])
    expected_x = [997 / np.sqrt(2 / 3), 0] if scaled else [1000, 3]
    np.testing.assert_allclose(bundle.x_test_numeric[:, 0].numpy(), expected_x, rtol=1e-6)
    assert bundle.x_test_numeric[:, 1].tolist() == [0, 100]


def test_drg_normalization_fits_disease_target_and_case_feature_stats_on_train():
    drg = _module("respay.data.drg")
    train = pd.DataFrame(
        {
            "病种编码": ["A", "A", "B"],
            "真实病例结构加权标准金额": [10, 10, 30],
            "当前病例结构加权标准金额": [5, 5, 15],
            "总金额": [2, 4, 6],
        }
    )
    test = pd.DataFrame(
        {
            "病种编码": ["Z"],
            "真实病例结构加权标准金额": [1000],
            "当前病例结构加权标准金额": [500],
            "总金额": [1000],
        }
    )
    _, splits, metadata = drg.attach_train_based_normalized_amounts(
        pd.concat([train, test]),
        {"train_disease_split": train, "test_disease_split": test},
    )
    assert metadata["target_amount_mean"] == 20
    assert metadata["target_amount_std"] == 10
    assert splits["test_disease_split"]["标准化真实病例结构加权标准金额"].iloc[0] == 98
    assert splits["test_disease_split"]["标准化总金额"].iloc[0] == pytest.approx(
        996 / np.sqrt(8 / 3)
    )


def test_drg_identity_shares_recover_source_case_mix_target():
    drg = _module("respay.data.drg")
    frame = pd.DataFrame(
        {
            "病种编码": ["A"] * 3,
            "身份类别": ["职工", "职工", "居民"],
            "标准化个体真实标准金额": [20, 20, 5],
            "标准化个体当前标准金额": [10, 10, 1],
            "标准化真实病例结构加权标准金额": [15] * 3,
            "标准化当前病例结构加权标准金额": [7] * 3,
            "标准化总金额": [1, 3, 9],
            "标准化目标泄漏": [999] * 3,
        }
    )
    units = drg.build_identity_split_tables(
        {
            "train_disease_split": frame,
            "test_disease_split": frame,
        }
    )["train_identity_split"].set_index("identity_type")
    assert units.loc["职工", "identity_weight"] == pytest.approx(2 / 3)
    assert units.loc["居民", "identity_weight"] == pytest.approx(1 / 3)
    assert (units.target_identity_amount * units.identity_weight).sum() == pytest.approx(15)
    assert units.loc["职工", "标准化总金额"] == 2
    assert "标准化目标泄漏" not in units.columns
    assert units["case_count"].sum() == 3


@pytest.mark.parametrize("task", ["dip_disease", "drg_identity_amount"])
def test_explicit_paths_export_only_final_splits_and_load_custom_root(tmp_path, task):
    dataset = _module("respay.dataset")
    output = tmp_path / "processed"
    if task == "dip_disease":
        raw = tmp_path / "source.csv"
        _dip_raw(raw)
        outputs = _module("respay.data.dip").export_dip_datasets(raw, output / "dip_processed")
        expected = {
            "train_disease_split",
            "test_disease_split",
            "train_codes",
            "test_codes",
            "metadata",
        }
    else:
        raw = tmp_path / "source"
        _drg_raw(raw)
        outputs = _module("respay.data.drg").export_drg_datasets(output / "drg_processed", raw)
        expected = {
            "train_identity_split",
            "test_identity_split",
            "train_codes",
            "test_codes",
            "disease_summary",
            "metadata",
        }
    assert set(outputs) == expected
    assert set(output.rglob("*.csv")) == {p for p in outputs.values() if p.suffix == ".csv"}
    for path in outputs.values():
        assert path.is_relative_to(output)
        assert "PRIVATE" not in path.read_text(encoding="utf-8-sig")
    metadata = json.loads(outputs["metadata"].read_text())
    assert metadata["disease_split"]["train_codes"] == 4
    assert metadata["disease_split"]["test_codes"] == 1
    bundle = dataset.load_dataset_bundle(task, processed_root=output)
    assert len(bundle.y_test) == (1 if task == "dip_disease" else 2)
    assert (
        bundle.config.train_path
        == outputs["train_disease_split" if task == "dip_disease" else "train_identity_split"]
    )
    if task == "drg_identity_amount":
        assert bundle.train_df["case_count"].sum() == 36
        assert bundle.test_df["case_count"].sum() == 9
        assert set(bundle.test_df["disease_code"]) == {"A"}
        summary = pd.read_csv(outputs["disease_summary"]).set_index("disease_code")
        assert summary.loc["A", "split"] == "test"
        assert summary.loc["A", "case_count"] == 9
        assert summary["case_count"].sum() == 45
        assert summary.loc["A", "current_group_amount_raw"] == pytest.approx(250 / 3)
        assert summary.loc["A", "target_group_amount_raw"] == pytest.approx(500 / 3)


def test_prepare_cli_handles_both_raw_formats_and_actionable_missing_files(tmp_path):
    raw_csv = tmp_path / "dip.csv"
    raw_xlsx = tmp_path / "drg"
    _dip_raw(raw_csv)
    _drg_raw(raw_xlsx)
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "respay.cli.prepare_data",
            "--dip-raw-path",
            str(raw_csv),
            "--drg-raw-dir",
            str(raw_xlsx),
            "--output-root",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "drg_processed/identity_split/test_identity_split.csv").exists()
    assert (output / "dip_processed/disease_split/test_disease_split.csv").exists()
    for task, flag in [("dip_disease", "--dip-raw-path"), ("drg_identity_amount", "--drg-raw-dir")]:
        missing = tmp_path / "missing"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "respay.cli.prepare_data",
                "--tasks",
                task,
                flag,
                str(missing),
                "--output-root",
                str(tmp_path / "unused"),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert str(missing) in result.stderr
        assert flag in result.stderr


def test_missing_processed_data_explains_preprocessing_command(tmp_path):
    dataset = _module("respay.dataset")
    with pytest.raises(FileNotFoundError, match="respay.cli.prepare_data"):
        dataset.load_dataset_bundle("dip_disease", processed_root=tmp_path)
