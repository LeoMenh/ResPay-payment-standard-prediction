"""Load final DIP and DRG splits with preprocessing fitted on training rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from respay.paths import PROCESSED_DATA_ROOT

DEFAULT_EXPERIMENT_TASKS = ("dip_disease", "drg_identity_amount")


DIP_FEATURE_CANDIDATES = [
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
    "统一标准分值费用",
]


DRG_IDENTITY_FEATURE_CANDIDATES = [
    "current_identity_amount",
    "current_group_amount",
    "identity_weight",
    "case_count",
    "identity_employee",
    "identity_resident",
    "identity_province",
    "标准化当前病例结构加权标准金额",
    "标准化个体当前标准金额",
    "标准化个体当前基础费率",
    "标准化统一权重",
    "标准化总金额",
    "标准化药品费用",
    "标准化材料费用",
    "标准化药耗费用",
    "标准化检查费用",
    "标准化检验费用",
    "标准化医疗服务费用",
    "标准化科内使用耗材金额",
    "标准化科内治疗费",
    "标准化手术费",
    "标准化放疗费",
    "标准化其他费用",
    "标准化住院天数",
]


@dataclass
class DatasetConfig:
    """Columns and preprocessing policy for one final experiment task."""

    name: str
    train_path: Path
    test_path: Path
    target_col: str
    code_col: str
    feature_candidates: list[str]
    prior_col: str
    scale_features_in_loader: bool = True


@dataclass
class DatasetBundle:
    """Case or insurance-unit features, group labels, and aggregation indices."""

    config: DatasetConfig
    feature_cols: list[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_disease_df: pd.DataFrame
    test_disease_df: pd.DataFrame
    x_train_numeric: torch.Tensor
    x_test_numeric: torch.Tensor
    train_group_idx: torch.Tensor
    test_group_idx: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    y_train_prior: torch.Tensor
    y_test_prior: torch.Tensor


TASK_CONFIGS: dict[str, DatasetConfig] = {
    "dip_disease": DatasetConfig(
        name="dip_disease",
        train_path=PROCESSED_DATA_ROOT
        / "dip_processed"
        / "disease_split"
        / "train_disease_split.csv",
        test_path=PROCESSED_DATA_ROOT
        / "dip_processed"
        / "disease_split"
        / "test_disease_split.csv",
        target_col="真实标准分值费用",
        code_col="病种编码",
        feature_candidates=DIP_FEATURE_CANDIDATES,
        prior_col="统一标准分值费用",
        scale_features_in_loader=False,
    ),
    "drg_identity_amount": DatasetConfig(
        name="drg_identity_amount",
        train_path=PROCESSED_DATA_ROOT
        / "drg_processed"
        / "identity_split"
        / "train_identity_split.csv",
        test_path=PROCESSED_DATA_ROOT
        / "drg_processed"
        / "identity_split"
        / "test_identity_split.csv",
        target_col="target_identity_amount",
        code_col="identity_code",
        feature_candidates=DRG_IDENTITY_FEATURE_CANDIDATES,
        prior_col="current_identity_amount",
        scale_features_in_loader=True,
    ),
}


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def _pick_feature_cols(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: DatasetConfig
) -> list[str]:
    shared_cols = set(train_df.columns).intersection(set(test_df.columns))
    feature_cols: list[str] = []
    missing_cols: list[str] = []

    for col in config.feature_candidates:
        if col in shared_cols:
            feature_cols.append(col)
        else:
            missing_cols.append(col)

    if not feature_cols:
        raise ValueError(f"No usable features for task {config.name}")
    if missing_cols:
        raise ValueError(
            f"Task {config.name} is missing expected features: {missing_cols}. "
            "Rerun preprocessing to generate the complete standardized features."
        )
    return feature_cols


def _mode_value(series: pd.Series):
    cleaned = series.dropna()
    if cleaned.empty:
        return np.nan
    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return cleaned.iloc[0]


def _build_disease_view(
    df: pd.DataFrame,
    config: DatasetConfig,
    target_col: str,
) -> tuple[pd.DataFrame, torch.Tensor]:
    working = df.copy()
    working[config.code_col] = working[config.code_col].astype(str)

    target_conflict = (
        working.groupby(config.code_col, dropna=False)[target_col]
        .nunique(dropna=True)
        .reset_index(name="目标唯一值个数")
    )
    conflict_count = int((target_conflict["目标唯一值个数"] > 1).sum())
    if conflict_count > 0:
        print(
            f"[Data warning] Task {config.name} has {conflict_count} groups with multiple "
            "target values; using the mode as the group label."
        )

    aggregations = {
        "病例数": (config.code_col, "size"),
        "真实值": (target_col, _mode_value),
    }
    if config.prior_col in working.columns:
        aggregations["基准值"] = (config.prior_col, _mode_value)

    disease_df = (
        working.groupby(config.code_col, sort=True, dropna=False).agg(**aggregations).reset_index()
    )
    disease_df = disease_df.merge(target_conflict, on=config.code_col, how="left")
    disease_df["病种分组索引"] = np.arange(len(disease_df), dtype=np.int64)

    code_to_group = disease_df.set_index(config.code_col)["病种分组索引"].to_dict()
    group_idx = working[config.code_col].map(code_to_group).astype(np.int64).to_numpy()
    return disease_df, torch.tensor(group_idx, dtype=torch.long)


def load_dataset_bundle(
    task_name: str,
    processed_root: str | Path = PROCESSED_DATA_ROOT,
) -> DatasetBundle:
    """Load a final split beneath the default or explicitly supplied processed root."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Choose from {tuple(TASK_CONFIGS)}")
    template = TASK_CONFIGS[task_name]
    processed_root = Path(processed_root)
    config = replace(
        template,
        train_path=processed_root / template.train_path.relative_to(PROCESSED_DATA_ROOT),
        test_path=processed_root / template.test_path.relative_to(PROCESSED_DATA_ROOT),
    )
    for path in (config.train_path, config.test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Processed split not found: {path}. Run python -m respay.cli.prepare_data "
                f"--tasks {task_name} --output-root {processed_root} with the raw-data arguments."
            )
    return build_dataset_bundle_from_frames(
        config, _load_csv(config.train_path), _load_csv(config.test_path)
    )


def build_dataset_bundle_from_frames(
    config: DatasetConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> DatasetBundle:
    """Build a bundle while fitting all preprocessing on the supplied train frame."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    required_cols = {config.code_col, config.target_col, config.prior_col}
    missing_train = required_cols - set(train_df.columns)
    missing_test = required_cols - set(test_df.columns)
    if missing_train:
        raise ValueError(f"Training data is missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Test data is missing columns: {sorted(missing_test)}")

    feature_cols = _pick_feature_cols(train_df, test_df, config)

    train_df = train_df.dropna(subset=[config.code_col, config.target_col]).copy()
    test_df = test_df.dropna(subset=[config.code_col, config.target_col]).copy()

    numeric_cols = list(dict.fromkeys(feature_cols + [config.target_col]))
    if config.prior_col in train_df.columns and config.prior_col in test_df.columns:
        numeric_cols.append(config.prior_col)
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    train_fill = train_df[feature_cols].median(numeric_only=True)
    train_df[feature_cols] = train_df[feature_cols].fillna(train_fill).fillna(0.0)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_fill).fillna(0.0)

    train_df = train_df.dropna(subset=[config.target_col]).copy()
    test_df = test_df.dropna(subset=[config.target_col]).copy()

    if config.scale_features_in_loader:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[feature_cols].astype(float))
        x_test = scaler.transform(test_df[feature_cols].astype(float))
    else:
        x_train = train_df[feature_cols].astype(float).to_numpy()
        x_test = test_df[feature_cols].astype(float).to_numpy()

    train_disease_df, train_group_idx = _build_disease_view(train_df, config, config.target_col)
    test_disease_df, test_group_idx = _build_disease_view(test_df, config, config.target_col)

    y_train = train_disease_df["真实值"].values.astype(np.float32)
    y_test = test_disease_df["真实值"].values.astype(np.float32)
    train_prior = train_disease_df["基准值"].astype(np.float32).to_numpy()
    test_prior = test_disease_df["基准值"].astype(np.float32).to_numpy()
    y_train_prior = torch.tensor(train_prior, dtype=torch.float32)
    y_test_prior = torch.tensor(test_prior, dtype=torch.float32)

    return DatasetBundle(
        config=config,
        feature_cols=feature_cols,
        train_df=train_df,
        test_df=test_df,
        train_disease_df=train_disease_df,
        test_disease_df=test_disease_df,
        x_train_numeric=torch.tensor(x_train, dtype=torch.float32),
        x_test_numeric=torch.tensor(x_test, dtype=torch.float32),
        train_group_idx=train_group_idx,
        test_group_idx=test_group_idx,
        y_train=torch.tensor(y_train, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.float32),
        y_train_prior=y_train_prior,
        y_test_prior=y_test_prior,
    )
