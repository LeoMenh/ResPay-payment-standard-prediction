"""Build the DRG 1.1-to-2.0 cohort and source-period insurance units."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from respay.paths import PROCESSED_DATA_ROOT, RAW_DATA_ROOT

DEFAULT_RAW_DATA_DIR = RAW_DATA_ROOT / "drg"
DEFAULT_RAW_DATA_PATH_2023 = DEFAULT_RAW_DATA_DIR / "2023年1-12月国新分组数据.xlsx"
DEFAULT_RAW_DATA_PATH_2024 = DEFAULT_RAW_DATA_DIR / "2024年1-12月国新分组数据.xlsx"
DEFAULT_RAW_DATA_PATH_2025 = DEFAULT_RAW_DATA_DIR / "2025年1-12月国新分组数据.xlsx"
DEFAULT_RAW_DATA_PATH_DRG_1_1 = DEFAULT_RAW_DATA_DIR / "2024年1.1版权重费率表.xlsx"
DEFAULT_RAW_DATA_PATH_DRG_2_0 = (
    DEFAULT_RAW_DATA_DIR / "2025年2.0版权重费率表费率（职工11800，居民9279.72）.xlsx"
)

OUTPUT_ROOT = PROCESSED_DATA_ROOT / "drg_processed"
DRG_STANDARD_AMOUNT_COLS = [
    "真实病例结构加权标准金额",
    "当前病例结构加权标准金额",
    "个体真实标准金额",
    "个体当前标准金额",
    "个体当前基础费率",
    "个体目标基础费率",
]
DRG_MONETARY_FEATURE_COLS = [
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
DRG_NON_MONETARY_FEATURE_COLS = [
    "统一权重",
    "职工病例数",
    "居民病例数",
    "省直病例数",
    "住院天数",
]


def _mode_value(series: pd.Series):
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return cleaned.iloc[0]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _version_base_rates(standard_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    identity_cols = {
        "职工": "标准费率_职工",
        "居民": "标准费率_居民",
        "省直": "标准费率_省直",
    }
    for version, group in standard_df.groupby("DRG版本", dropna=False):
        version_key = str(version)
        rates[version_key] = {}
        standard_weight = pd.to_numeric(group["标准权重"], errors="coerce")
        for identity, amount_col in identity_cols.items():
            if amount_col not in group.columns:
                continue
            amount = pd.to_numeric(group[amount_col], errors="coerce")
            base_rate = _safe_divide(amount, standard_weight)
            base_rate = base_rate.replace([np.inf, -np.inf], np.nan).dropna()
            if base_rate.empty:
                continue
            mode = base_rate.round(6).mode(dropna=True)
            rates[version_key][identity] = float(
                mode.iloc[0] if not mode.empty else base_rate.median()
            )
    return rates


def _pick_identity_value(row: pd.Series, col_by_identity: dict[str, str]) -> float:
    identity = row.get("身份类别")
    col = col_by_identity.get(str(identity))
    if col is None:
        return np.nan
    return row.get(col, np.nan)


def _standardize_series(series: pd.Series, mean: float, std: float) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") - mean) / std


def _fit_mean_std(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 1e-8:
        std = 1.0
    return mean, std


def _ensure_file_exists(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(
            f"DRG raw workbook not found: {file_path}. Supply --drg-raw-dir DIR "
            "containing the three annual case workbooks and both version rate tables."
        )


def _pick_case_sheet(file_path: Path) -> str:
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        probe = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
        if len(probe.columns) > 0 and probe.shape[0] >= 1:
            return sheet_name
    raise ValueError(f"No usable sheet in workbook: {file_path}")


def _parse_settlement_time(df: pd.DataFrame) -> pd.Series:
    if "结算月份" in df.columns:
        parsed = pd.to_datetime(df["结算月份"].astype(str).str[:7], format="%Y-%m", errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed

    for col in ("结算日期", "出院日期", "入院日期"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed

    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _normalize_identity(row: pd.Series) -> str | float:
    text_parts = []
    for col in ("医保类型", "合同单位", "人员类型", "人员类型.1"):
        value = row.get(col)
        if pd.notna(value):
            text_parts.append(str(value))
    merged = "|".join(text_parts)

    if "省直" in merged:
        return "省直"
    if ("职工" in merged) or ("退休" in merged):
        return "职工"
    if "居民" in merged:
        return "居民"
    return np.nan


def _mark_drg_version(year: int, month: int) -> str | float:
    if pd.isna(year) or pd.isna(month):
        return np.nan
    if year < 2025:
        return "1.1"
    if year == 2025 and month <= 3:
        return "1.1"
    if year == 2025 and month >= 4:
        return "2.0"
    return np.nan


def load_case_data(file_path: Path, source_year: int) -> pd.DataFrame:
    """Read one annual workbook and derive settlement version and insurance identity."""
    _ensure_file_exists(file_path)
    sheet = _pick_case_sheet(file_path)
    df = pd.read_excel(file_path, sheet_name=sheet)

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    df["数据来源年份"] = source_year

    if "合同单位2" in df.columns and "合同单位" not in df.columns:
        df = df.rename(columns={"合同单位2": "合同单位"})

    required_cols = {"分组编码", "权重"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {sorted(missing)}")

    df = df.rename(columns={"分组编码": "病种编码"})
    df["结算时间"] = _parse_settlement_time(df)
    df["年份"] = df["结算时间"].dt.year
    df["月份"] = df["结算时间"].dt.month
    df["DRG版本"] = df.apply(lambda row: _mark_drg_version(row["年份"], row["月份"]), axis=1)
    df["身份类别"] = df.apply(_normalize_identity, axis=1)

    numeric_cols = [
        "权重",
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
        "特药谈判药",
        "协议期内谈判药",
        "化疗药物",
        "骨髓抑制药物",
        "住院天数",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_case_data(raw_data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    """Combine annual workbooks and retain cases with valid codes, weights, and versions."""
    parts = [
        load_case_data(raw_data_dir / DEFAULT_RAW_DATA_PATH_2023.name, 2023),
        load_case_data(raw_data_dir / DEFAULT_RAW_DATA_PATH_2024.name, 2024),
        load_case_data(raw_data_dir / DEFAULT_RAW_DATA_PATH_2025.name, 2025),
    ]
    case_df = pd.concat(parts, ignore_index=True)
    case_df = case_df.dropna(subset=["病种编码", "权重", "年份", "DRG版本"]).copy()
    case_df["年份"] = case_df["年份"].astype(int)
    case_df["月份"] = case_df["月份"].astype(int)
    return case_df


def load_drg_1_1_standard_table(file_path: Path = DEFAULT_RAW_DATA_PATH_DRG_1_1) -> pd.DataFrame:
    """Read the fixed source-version rate-table sheet."""
    _ensure_file_exists(file_path)
    df = pd.read_excel(file_path, sheet_name="济南市 省内异地")
    df = df.rename(
        columns={
            "DRG编码": "病种编码",
            "权重": "标准权重",
            "职工": "标准费率_职工",
            "居民": "标准费率_居民",
            "省直": "标准费率_省直",
        }
    )

    keep_cols = ["病种编码", "标准权重", "标准费率_职工", "标准费率_居民", "标准费率_省直"]
    df = df[[col for col in keep_cols if col in df.columns]].copy()
    df["DRG版本"] = "1.1"

    for col in ["标准权重", "标准费率_职工", "标准费率_居民", "标准费率_省直"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_drg_2_0_standard_table(file_path: Path = DEFAULT_RAW_DATA_PATH_DRG_2_0) -> pd.DataFrame:
    """Read the target-version rate table after its two introductory rows."""
    _ensure_file_exists(file_path)
    df = pd.read_excel(file_path, sheet_name="济南市 省异地", header=2)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()

    rename_map = {
        "DRG": "病种编码",
        "权重": "标准权重",
        "标准费用": "标准费率_职工",
        "标准费用.1": "标准费率_居民",
        "标准费用.2": "标准费率_省直",
    }
    df = df.rename(columns=rename_map)

    if "标准费率_省直" not in df.columns:
        candidates = [col for col in df.columns if str(col).startswith("标准费用")]
        if len(candidates) >= 3:
            df = df.rename(columns={candidates[2]: "标准费率_省直"})

    keep_cols = ["病种编码", "标准权重", "标准费率_职工", "标准费率_居民", "标准费率_省直"]
    df = df[[col for col in keep_cols if col in df.columns]].copy()
    df["DRG版本"] = "2.0"

    for col in ["标准权重", "标准费率_职工", "标准费率_居民", "标准费率_省直"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_standard_tables(raw_data_dir: Path = DEFAULT_RAW_DATA_DIR) -> pd.DataFrame:
    """Combine both version-specific standard-payment tables."""
    standard_df = pd.concat(
        [
            load_drg_1_1_standard_table(raw_data_dir / DEFAULT_RAW_DATA_PATH_DRG_1_1.name),
            load_drg_2_0_standard_table(raw_data_dir / DEFAULT_RAW_DATA_PATH_DRG_2_0.name),
        ],
        ignore_index=True,
    )
    standard_df = standard_df.dropna(subset=["病种编码"]).copy()
    return standard_df


def attach_unified_weight(case_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach year/version/code modal weights without dropping source cases."""
    summary = (
        case_df.groupby(["年份", "DRG版本", "病种编码"], dropna=False)
        .agg(统一权重=("权重", _mode_value), 权重唯一值个数=("权重", "nunique"))
        .reset_index()
    )

    enriched = case_df.merge(summary, on=["年份", "DRG版本", "病种编码"], how="left")
    return enriched, summary


def attach_cross_version_target(case_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict to common codes and attach version 2.0 modal target weights."""
    version_summary = (
        case_df.groupby(["DRG版本", "病种编码"], dropna=False)
        .agg(版本统一权重=("权重", _mode_value), 版本内权重唯一值个数=("权重", "nunique"))
        .reset_index()
    )

    version_weight_pivot = version_summary.pivot(
        index="病种编码", columns="DRG版本", values="版本统一权重"
    )
    version_weight_pivot.columns = version_weight_pivot.columns.astype(str)
    common_codes = version_weight_pivot.dropna(subset=["1.1", "2.0"]).index.astype(str).tolist()

    case_df = case_df.copy()
    case_df["病种编码"] = case_df["病种编码"].astype(str)
    case_df = case_df[case_df["病种编码"].isin(common_codes)].copy()

    weight_20_map = (
        version_summary.loc[
            version_summary["DRG版本"].astype(str) == "2.0", ["病种编码", "版本统一权重"]
        ]
        .dropna(subset=["病种编码"])
        .set_index("病种编码")["版本统一权重"]
        .to_dict()
    )

    case_df = case_df.merge(version_summary, on=["DRG版本", "病种编码"], how="left")

    # Keep source-version cases only; subsequent-version modes supply the labels.
    case_df = case_df[case_df["DRG版本"].astype(str) == "1.1"].copy()
    case_df["真实权重"] = case_df["病种编码"].map(weight_20_map)
    case_df["真实权重来源版本"] = "2.0"
    case_df = case_df.dropna(subset=["真实权重"]).copy()

    return case_df, version_summary


def attach_standard_payment_amounts(
    case_df: pd.DataFrame, standard_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach identity rates with version-base-rate fallback and case-mix means."""
    base_rates = _version_base_rates(standard_df)
    target_standard_df = (
        standard_df.loc[standard_df["DRG版本"].astype(str) == "2.0"]
        .drop(columns=["DRG版本"], errors="ignore")
        .rename(
            columns={
                "标准权重": "目标标准权重",
                "标准费率_职工": "目标标准金额_职工",
                "标准费率_居民": "目标标准金额_居民",
                "标准费率_省直": "目标标准金额_省直",
            }
        )
    )

    merged = case_df.merge(
        standard_df,
        on=["DRG版本", "病种编码"],
        how="left",
        suffixes=("", "_费率表"),
    )
    merged = merged.merge(target_standard_df, on="病种编码", how="left")

    current_amount_cols = {
        "职工": "标准费率_职工",
        "居民": "标准费率_居民",
        "省直": "标准费率_省直",
    }
    target_amount_cols = {
        "职工": "目标标准金额_职工",
        "居民": "目标标准金额_居民",
        "省直": "目标标准金额_省直",
    }
    merged["个体当前标准金额"] = merged.apply(
        lambda row: _pick_identity_value(row, current_amount_cols), axis=1
    )
    merged["个体真实标准金额"] = merged.apply(
        lambda row: _pick_identity_value(row, target_amount_cols), axis=1
    )
    for amount_col in ["个体当前标准金额", "个体真实标准金额"]:
        merged[amount_col] = pd.to_numeric(merged[amount_col], errors="coerce")
        merged.loc[merged[amount_col] <= 0, amount_col] = np.nan

    merged["个体当前基础费率"] = merged["身份类别"].map(base_rates.get("1.1", {}))
    merged["个体目标基础费率"] = merged["身份类别"].map(base_rates.get("2.0", {}))
    merged["个体当前标准金额"] = merged["个体当前标准金额"].fillna(
        merged["统一权重"] * merged["个体当前基础费率"]
    )
    merged["个体真实标准金额"] = merged["个体真实标准金额"].fillna(
        merged["真实权重"] * merged["个体目标基础费率"]
    )
    if "标准费用" in merged.columns:
        merged["个体当前标准金额"] = merged["个体当前标准金额"].fillna(merged["标准费用"])

    payment_summary = (
        merged.groupby("病种编码", dropna=False)
        .agg(
            当前病例结构加权标准金额=("个体当前标准金额", "mean"),
            真实病例结构加权标准金额=("个体真实标准金额", "mean"),
            当前标准金额缺失病例数=("个体当前标准金额", lambda s: int(s.isna().sum())),
            目标标准金额缺失病例数=("个体真实标准金额", lambda s: int(s.isna().sum())),
            职工病例数=("身份类别", lambda s: int((s == "职工").sum())),
            居民病例数=("身份类别", lambda s: int((s == "居民").sum())),
            省直病例数=("身份类别", lambda s: int((s == "省直").sum())),
        )
        .reset_index()
    )
    merged = merged.merge(payment_summary, on="病种编码", how="left")

    return merged


def build_disease_split_tables(
    processed_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split eligible source-version disease codes 80:20 with fixed seed 42."""
    usable_df = processed_df.dropna(subset=["病种编码", "真实病例结构加权标准金额"]).copy()
    usable_df["病种编码"] = usable_df["病种编码"].astype(str)
    usable_df = usable_df.loc[usable_df["DRG版本"].astype(str) == "1.1"].copy()
    available_codes = sorted(usable_df["病种编码"].dropna().unique().tolist())

    if len(available_codes) < 2:
        raise ValueError("At least two disease codes are required for the fixed group split.")

    rng = np.random.default_rng(42)
    shuffled_codes = rng.permutation(available_codes)
    split_index = int(round(len(shuffled_codes) * 0.8))
    split_index = min(max(split_index, 1), len(shuffled_codes) - 1)

    train_codes = set(shuffled_codes[:split_index])
    test_codes = set(shuffled_codes[split_index:])

    train_df = usable_df[usable_df["病种编码"].isin(train_codes)].copy()
    test_df = usable_df[usable_df["病种编码"].isin(test_codes)].copy()

    return {
        "train_disease_split": train_df,
        "test_disease_split": test_df,
        "train_codes": pd.DataFrame({"病种编码": sorted(train_codes)}),
        "test_codes": pd.DataFrame({"病种编码": sorted(test_codes)}),
    }


def attach_train_based_normalized_amounts(
    all_cases_df: pd.DataFrame,
    split_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object]]:
    """Normalize targets by training diseases and covariates by training cases."""
    train_df = split_tables["train_disease_split"]
    train_disease_target = train_df.groupby("病种编码", sort=True, dropna=False)[
        "真实病例结构加权标准金额"
    ].first()
    target_mean, target_std = _fit_mean_std(train_disease_target)

    feature_stats: dict[str, dict[str, float]] = {}
    for col in [*DRG_MONETARY_FEATURE_COLS, *DRG_NON_MONETARY_FEATURE_COLS]:
        if col not in train_df.columns:
            continue
        mean, std = _fit_mean_std(train_df[col])
        feature_stats[col] = {"mean": mean, "std": std}

    def _apply(table: pd.DataFrame) -> pd.DataFrame:
        out = table.copy()
        for col in DRG_STANDARD_AMOUNT_COLS:
            if col in out.columns:
                out[f"标准化{col}"] = _standardize_series(out[col], target_mean, target_std)
        for col, stats in feature_stats.items():
            if col in out.columns:
                out[f"标准化{col}"] = _standardize_series(out[col], stats["mean"], stats["std"])
        return out

    normalized_all_cases = _apply(all_cases_df)
    normalized_split_tables = {
        name: _apply(table) if name in {"train_disease_split", "test_disease_split"} else table
        for name, table in split_tables.items()
    }
    metadata = {
        "standardization_rule": (
            "DRG amount targets and amount-valued features are z-scored with statistics fitted "
            "on the training disease split only. The prediction target uses disease-level "
            "真实病例结构加权标准金额 mean/std."
        ),
        "target_amount_mean": target_mean,
        "target_amount_std": target_std,
        "target_col": "标准化真实病例结构加权标准金额",
        "prior_col": "标准化当前病例结构加权标准金额",
        "amount_feature_stats": feature_stats,
    }
    return normalized_all_cases, normalized_split_tables, metadata


def build_identity_split_tables(split_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build final insurance units with source-period shares for disease-level aggregation."""
    target_unit_col = "标准化个体真实标准金额"
    prior_unit_col = "标准化个体当前标准金额"
    target_group_col = "标准化真实病例结构加权标准金额"
    prior_group_col = "标准化当前病例结构加权标准金额"
    identity_values = {
        "职工": "employee",
        "居民": "resident",
        "省直": "province",
    }

    def _build(table: pd.DataFrame) -> pd.DataFrame:
        required = [
            "病种编码",
            "身份类别",
            target_unit_col,
            prior_unit_col,
            target_group_col,
            prior_group_col,
        ]
        missing = [col for col in required if col not in table.columns]
        if missing:
            raise ValueError(f"DRG identity split missing required columns: {missing}")

        working = table.dropna(subset=required).copy()
        working["病种编码"] = working["病种编码"].astype(str)
        working["身份类别"] = working["身份类别"].astype(str)
        working = working[working["身份类别"].isin(identity_values.keys())].copy()
        if working.empty:
            raise ValueError("DRG identity split has no usable identity rows.")

        summary = (
            working.groupby(["病种编码", "身份类别"], sort=True, dropna=False)
            .agg(
                case_count=("病种编码", "size"),
                target_identity_amount=(target_unit_col, _mode_value),
                current_identity_amount=(prior_unit_col, _mode_value),
                target_group_amount=(target_group_col, _mode_value),
                current_group_amount=(prior_group_col, _mode_value),
            )
            .reset_index()
        )
        total_cases = summary.groupby("病种编码")["case_count"].transform("sum").replace(0, np.nan)
        summary["identity_weight"] = summary["case_count"] / total_cases
        summary["identity_code"] = (
            summary["病种编码"].astype(str) + "__" + summary["身份类别"].astype(str)
        )
        summary = summary.rename(columns={"病种编码": "disease_code", "身份类别": "identity_type"})

        for raw_identity, english_name in identity_values.items():
            summary[f"identity_{english_name}"] = (summary["identity_type"] == raw_identity).astype(
                float
            )

        feature_cols = [
            col
            for col in working.columns
            if str(col).startswith("标准化")
            and col not in {target_unit_col, target_group_col}
            and "真实" not in str(col)
            and "目标" not in str(col)
            and pd.api.types.is_numeric_dtype(working[col])
        ]
        if feature_cols:
            feature_summary = (
                working.groupby(["病种编码", "身份类别"], sort=True, dropna=False)[feature_cols]
                .mean()
                .reset_index()
                .rename(columns={"病种编码": "disease_code", "身份类别": "identity_type"})
            )
            summary = summary.merge(
                feature_summary, on=["disease_code", "identity_type"], how="left"
            )

        return summary.sort_values(["disease_code", "identity_type"]).reset_index(drop=True)

    return {
        "train_identity_split": _build(split_tables["train_disease_split"]),
        "test_identity_split": _build(split_tables["test_disease_split"]),
    }


def _save_dataframe(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")


def export_drg_datasets(
    output_root: str | Path = OUTPUT_ROOT,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> dict[str, Path]:
    """Export final insurance-unit splits, code lists, and normalization metadata."""
    output_root = Path(output_root)
    raw_data_dir = Path(raw_data_dir)

    case_df = load_all_case_data(raw_data_dir)
    standard_df = load_all_standard_tables(raw_data_dir)
    enriched_case_df, weight_summary = attach_unified_weight(case_df)
    enriched_case_df, version_summary = attach_cross_version_target(enriched_case_df)
    enriched_case_df = attach_standard_payment_amounts(enriched_case_df, standard_df)
    split_tables = build_disease_split_tables(enriched_case_df)
    enriched_case_df, split_tables, normalization_metadata = attach_train_based_normalized_amounts(
        enriched_case_df,
        split_tables,
    )

    outputs: dict[str, Path] = {}

    split_dir = output_root / "disease_split"
    for name in ("train_codes", "test_codes"):
        path = split_dir / f"{name}.csv"
        _save_dataframe(split_tables[name], path)
        outputs[name] = path

    # Compact group summaries preserve source case counts and raw-currency exposure.
    summaries = []
    for split in ("train", "test"):
        table = split_tables[f"{split}_disease_split"]
        summary = (
            table.groupby("病种编码", sort=True)
            .agg(
                case_count=("病种编码", "size"),
                current_group_amount_raw=("当前病例结构加权标准金额", "first"),
                target_group_amount_raw=("真实病例结构加权标准金额", "first"),
                current_group_amount=("标准化当前病例结构加权标准金额", "first"),
                target_group_amount=("标准化真实病例结构加权标准金额", "first"),
            )
            .reset_index()
            .rename(columns={"病种编码": "disease_code"})
        )
        summary.insert(1, "split", split)
        summaries.append(summary)
    summary_path = output_root / "disease_summary.csv"
    _save_dataframe(pd.concat(summaries, ignore_index=True), summary_path)
    outputs["disease_summary"] = summary_path

    identity_tables = build_identity_split_tables(split_tables)
    identity_dir = output_root / "identity_split"
    for name, table in identity_tables.items():
        path = identity_dir / f"{name}.csv"
        _save_dataframe(table, path)
        outputs[name] = path

    conflict_by_year = (
        enriched_case_df.groupby(["年份", "病种编码"])["权重"]
        .nunique()
        .reset_index(name="权重唯一值个数")
    )
    conflict_count = int((conflict_by_year["权重唯一值个数"] > 1).sum())

    metadata = {
        "output_root": str(output_root),
        "raw_data_dir": str(raw_data_dir),
        "rows": {
            "all_cases": int(len(enriched_case_df)),
            "train_disease_split": int(len(split_tables["train_disease_split"])),
            "test_disease_split": int(len(split_tables["test_disease_split"])),
            "train_identity_split": int(len(identity_tables["train_identity_split"])),
            "test_identity_split": int(len(identity_tables["test_identity_split"])),
        },
        "common_disease_codes": int(enriched_case_df["病种编码"].nunique()),
        "disease_split": {
            "seed": 42,
            "train_fraction": 0.8,
            "train_codes": int(len(split_tables["train_codes"])),
            "test_codes": int(len(split_tables["test_codes"])),
        },
        "identity_split": {
            "train_identity_units": int(len(identity_tables["train_identity_split"])),
            "test_identity_units": int(len(identity_tables["test_identity_split"])),
            "target_col": "target_identity_amount",
            "prior_col": "current_identity_amount",
            "group_target_col": "target_group_amount",
            "group_prior_col": "current_group_amount",
            "weight_col": "identity_weight",
        },
        "version_distribution": enriched_case_df["DRG版本"].value_counts(dropna=False).to_dict(),
        "year_distribution": enriched_case_df["年份"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict(),
        "weight_conflicts": {
            "year_code_conflict_count": conflict_count,
            "year_version_code_conflict_count": int((weight_summary["权重唯一值个数"] > 1).sum()),
            "version_code_conflict_count": int((version_summary["版本内权重唯一值个数"] > 1).sum()),
        },
        "base_rates": _version_base_rates(standard_df),
        "target_rule": (
            "Use version 1.1 cases and current/2.0 identity-specific standard amounts. "
            "Source-period case shares define disease-level means; training disease targets "
            "supply the normalization statistics."
        ),
        "normalization": normalization_metadata,
        "target_col": "标准化真实病例结构加权标准金额",
        "prior_col": "标准化当前病例结构加权标准金额",
    }

    metadata_path = output_root / "split_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    outputs["metadata"] = metadata_path

    return outputs
