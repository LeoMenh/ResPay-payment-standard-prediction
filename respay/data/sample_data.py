"""Build public example inputs from scratch, without reading any private files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from respay.data.drg import (
    DEFAULT_RAW_DATA_PATH_DRG_1_1,
    DEFAULT_RAW_DATA_PATH_DRG_2_0,
    DRG_MONETARY_FEATURE_COLS,
    SAMPLE_FILENAMES,
)
from respay.dataset import DIP_FEATURE_CANDIDATES

SAMPLE_DATA_ROOT = Path(__file__).resolve().parent / "samples"


def generate_sample_data(output_root: str | Path = SAMPLE_DATA_ROOT) -> list[Path]:
    """Write one DIP CSV and five DRG workbooks with fully invented records.

    The 60 artificial groups provide enough units for MLP validation and both LR
    classes on the fixed train/test split. Values do not match hospital statistics.
    Only input column labels, insurance categories, and workbook layout are reused.
    """
    output_root = Path(output_root)
    dip_root, drg_root = output_root / "dip", output_root / "drg"
    dip_root.mkdir(parents=True, exist_ok=True)
    drg_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261026)
    dip_rows = []
    for index in range(60):
        current = 0.8 + index * 0.025
        correction = (0.03, -0.07, 0.18, -0.28, 0.7, -0.65)[index % 6]
        for year, standard in ((2023, current), (2024, current + correction)):
            for record in range(2 + index % 4):
                features = {}
                for column in DIP_FEATURE_CANDIDATES:
                    if column == "统一标准分值费用":
                        continue  # The reader derives this field from 标准分值费用.
                    if "占比" in column or "使用率" in column:
                        features[column] = round(float(rng.uniform(0.02, 0.3)), 6)
                    elif column == "天数":
                        features[column] = int(rng.integers(2, 16))
                    else:
                        features[column] = round(float(rng.uniform(0.1, 2.0)), 6)
                dip_rows.append(
                    {
                        "病种编码": f"SYN_DIP_{index + 1:03d}",
                        "病种名称": f"合成DIP组{index + 1:03d}",
                        "出院日期": f"{year}-06-{record + 1:02d}",
                        "标准分值": round(standard * 100, 6),
                        "标准分值费用": round(standard, 6),
                        **features,
                    }
                )
    dip_path = dip_root / "2023-2024_DIP_synthetic_data.csv"
    pd.DataFrame(dip_rows).to_csv(dip_path, index=False, encoding="utf-8-sig")
    paths = [dip_path]

    current_weights = 0.7 + np.arange(60) * 0.045
    changes = np.resize(np.array([0.02, -0.04, 0.55, -0.45, 0.12, -0.10]), 60)
    target_weights = current_weights + changes
    base_rates = {"职工": 9000.0, "居民": 7000.0, "省直": 8000.0}
    for year in (2023, 2024, 2025):
        rows = []
        # 2025 contains source (Q1) and target (Q2) version observations.
        for month in (2, 6) if year == 2025 else (6,):
            weights = target_weights if year == 2025 and month >= 4 else current_weights
            for index, weight in enumerate(weights):
                for identity_index, (identity, rate) in enumerate(base_rates.items()):
                    for _ in range(1 + (index + identity_index) % 3):
                        amount = float(weight * rate)
                        costs = {
                            column: round(amount * float(rng.uniform(0.01, 0.15)), 2)
                            for column in DRG_MONETARY_FEATURE_COLS
                        }
                        costs["标准费用"] = round(amount, 2)
                        costs["总金额"] = round(amount * float(rng.uniform(0.8, 1.3)), 2)
                        rows.append(
                            {
                                "分组编码": f"SYN_DRG_{index + 1:03d}",
                                "分组名称": f"合成DRG组{index + 1:03d}",
                                "结算月份": f"{year}-{month:02d}",
                                "医保类型": identity,
                                "权重": round(float(weight), 6),
                                "住院天数": int(rng.integers(2, 16)),
                                **costs,
                            }
                        )
        path = drg_root / f"{year}_DRG_synthetic_data.xlsx"
        pd.DataFrame(rows).to_excel(path, sheet_name="合成病例", index=False)
        paths.append(path)

    codes = [f"SYN_DRG_{index + 1:03d}" for index in range(60)]
    for target, weights, filename, sheet, header_rows in (
        (False, current_weights, DEFAULT_RAW_DATA_PATH_DRG_1_1.name, "济南市 省内异地", 0),
        (True, target_weights, DEFAULT_RAW_DATA_PATH_DRG_2_0.name, "济南市 省异地", 2),
    ):
        columns = {"DRG" if target else "DRG编码": codes, "权重": weights.round(6)}
        for position, (identity, rate) in enumerate(base_rates.items()):
            column = ("标准费用" + (f".{position}" if position else "")) if target else identity
            columns[column] = (weights * rate).round(6)
        path = drg_root / SAMPLE_FILENAMES[filename]
        # The target reader expects the header in Excel row three.
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(columns).to_excel(
                writer, sheet_name=sheet, startrow=header_rows, index=False
            )
            if header_rows:
                writer.sheets[sheet]["A1"] = "SYNTHETIC EXAMPLE ONLY"
                writer.sheets[sheet]["A2"] = "All codes, weights and amounts are invented."
        paths.append(path)
    return paths


if __name__ == "__main__":
    for generated_path in generate_sample_data():
        print(generated_path)
