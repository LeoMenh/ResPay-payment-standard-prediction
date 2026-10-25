"""Build the fixed DIP 2023-to-2024 disease cohort from a raw CSV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from respay.dataset import DIP_FEATURE_CANDIDATES
from respay.paths import PROCESSED_DATA_ROOT, RAW_DATA_ROOT

DEFAULT_RAW_DATA_PATH = RAW_DATA_ROOT / "dip" / "df_sample-1.csv"
OUTPUT_ROOT = PROCESSED_DATA_ROOT / "dip_processed"


def _read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to read CSV")


def load_dip_data(file_path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Read a raw CSV without relying on auxiliary header files."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DIP raw CSV not found: {file_path}. Supply --dip-raw-path PATH.")

    df_dip = _read_csv_with_fallback(file_path)
    df_dip = df_dip.loc[:, ~df_dip.columns.astype(str).str.startswith("Unnamed:")].copy()

    return df_dip


def _mode_value(series: pd.Series):
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan

    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return cleaned.iloc[0]


def _ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    if "出院日期" not in df.columns:
        raise ValueError("Missing discharge date column: 出院日期")

    enriched = df.copy()
    enriched["出院日期"] = pd.to_datetime(enriched["出院日期"], errors="coerce")
    enriched["年份"] = enriched["出院日期"].dt.year.astype("Int64")
    enriched = enriched.dropna(subset=["年份", "病种编码"]).copy()
    enriched["年份"] = enriched["年份"].astype(int)
    return enriched


def _attach_year_code_unified_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {"年份", "病种编码", "标准分值", "标准分值费用"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    summary = (
        df.groupby(["年份", "病种编码"], dropna=False)
        .agg(
            统一标准分值=("标准分值", _mode_value),
            统一标准分值费用=("标准分值费用", _mode_value),
        )
        .reset_index()
    )

    enriched = df.merge(summary, on=["年份", "病种编码"], how="left")

    next_year_summary = summary.copy()
    next_year_summary["年份"] = next_year_summary["年份"] - 1
    next_year_summary = next_year_summary.rename(
        columns={
            "统一标准分值": "真实标准分值",
            "统一标准分值费用": "真实标准分值费用",
        }
    )
    next_year_summary = next_year_summary[["年份", "病种编码", "真实标准分值", "真实标准分值费用"]]

    enriched = enriched.merge(next_year_summary, on=["年份", "病种编码"], how="left")
    enriched["目标年份"] = enriched["年份"] + 1
    return enriched, summary


def _save_dataframe(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")


def build_processed_dip_table(file_path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Attach year-wise modes and the following year's standards to each case."""
    raw_df = load_dip_data(file_path)
    year_df = _ensure_year_column(raw_df)
    processed_df, _ = _attach_year_code_unified_values(year_df)
    return processed_df


def build_disease_split_tables(
    processed_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split eligible source-year disease codes 80:20 with fixed seed 42."""
    source_df = processed_df.loc[processed_df["年份"] == 2023].copy()
    source_df = source_df.dropna(subset=["真实标准分值费用"]).copy()

    available_codes = sorted(source_df["病种编码"].dropna().unique().tolist())
    if len(available_codes) < 2:
        raise ValueError("At least two disease codes are required for the fixed group split.")

    rng = np.random.default_rng(42)
    shuffled_codes = rng.permutation(available_codes)
    split_index = int(round(len(shuffled_codes) * 0.8))
    split_index = min(max(split_index, 1), len(shuffled_codes) - 1)

    train_codes = set(shuffled_codes[:split_index])
    test_codes = set(shuffled_codes[split_index:])

    train_df = source_df[source_df["病种编码"].isin(train_codes)].copy()
    test_df = source_df[source_df["病种编码"].isin(test_codes)].copy()

    return {
        "train_disease_split": train_df,
        "test_disease_split": test_df,
        "train_codes": pd.DataFrame({"病种编码": sorted(train_codes)}),
        "test_codes": pd.DataFrame({"病种编码": sorted(test_codes)}),
    }


def export_dip_datasets(
    raw_data_path: str | Path = DEFAULT_RAW_DATA_PATH,
    output_root: str | Path = OUTPUT_ROOT,
) -> dict[str, Path]:
    """Export only selected disease splits, code lists, and cohort metadata."""
    output_root = Path(output_root)
    processed_df = build_processed_dip_table(raw_data_path)

    outputs: dict[str, Path] = {}

    disease_split_tables = build_disease_split_tables(processed_df)
    disease_split_dir = output_root / "disease_split"
    export_columns = [
        "病种编码",
        *DIP_FEATURE_CANDIDATES,
        "年份",
        "目标年份",
        "统一标准分值",
        "真实标准分值",
        "真实标准分值费用",
    ]
    for name, table in disease_split_tables.items():
        table = table.loc[:, [col for col in export_columns if col in table.columns]]
        file_path = disease_split_dir / f"{name}.csv"
        _save_dataframe(table, file_path)
        outputs[name] = file_path

    metadata = {
        "raw_data_path": str(Path(raw_data_path)),
        "output_root": str(output_root),
        "disease_split": {
            "source_year": 2023,
            "target_year": 2024,
            "seed": 42,
            "train_fraction": 0.8,
            "train_rows": int(len(disease_split_tables["train_disease_split"])),
            "test_rows": int(len(disease_split_tables["test_disease_split"])),
            "train_codes": int(len(disease_split_tables["train_codes"])),
            "test_codes": int(len(disease_split_tables["test_codes"])),
        },
    }
    metadata_path = output_root / "split_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    outputs["metadata"] = metadata_path

    return outputs
