# ResPay experiment code

This repository contains the data preparation, selected models, baselines, and numeric analysis used for the experiments reported in the paper. It does not include plotting scripts or figures.

This release includes fully synthetic example inputs for running the complete
workflow. Column labels and file layouts follow the input readers; all records,
group codes/names, features, weights, and numerical targets are invented. Example
output demonstrates execution and does not reproduce the paper's numerical results.

## Install

Use Python 3.10 or newer. From the repository root, install the package and development tools:

```bash
python -m pip install -e '.[dev]'
```

The project dependencies include PyTorch, pandas, NumPy, scikit-learn, XGBoost, and openpyxl. Use a PyTorch build suitable for the machine if GPU execution is required; the default run uses CUDA when available and CPU otherwise.

## Run with the public example data

From the repository root:

```bash
python -m respay.cli.run_all \
  --dip-raw-path respay/data/samples/dip/2023-2024_DIP_synthetic_data.csv \
  --drg-raw-dir respay/data/samples/drg \
  --output-root results/sample \
  --seeds 42 43 --epochs 3 --device cpu --threads 1
```

This runs both data preparation pipelines, residual/direct-target ResPay training,
all four baselines, and all numeric analyses. Two seeds allow meaningful sample-SD
output while keeping the demonstration small. Tables appear in
`results/sample/tables/`; processed inputs, predictions, and checkpoints stay under
`results/sample/`. The learned baselines may take longer than the
three-epoch neural demo and may report non-convergence on synthetic data.

The six input files are under [`respay/data/samples/`](respay/data/samples/README.md).
They are included in package distributions. To regenerate them from scratch:

```bash
python -m respay.data.sample_data
```

The generator creates all values from scratch using a fixed random seed. Years,
column labels, insurance categories, and workbook layouts are input-format
conventions. Explicit relative paths assume this directory as the working directory;
from elsewhere, pass absolute paths to the examples.

Neural CLI runs default to one PyTorch thread (`--threads 1`). CPU training uses
deterministic algorithms. Use `--device cuda` for a suitable GPU or `--device cpu`
for CPU execution; omit the option to select CUDA when available and CPU otherwise.

## Individual stages

Each command below runs from the repository root. The installed package can also be invoked from another working directory; default paths remain relative to this repository, while explicit relative paths are interpreted from the shell's current directory.

Prepare and export fixed train/test cohorts:

```bash
python -m respay.cli.prepare_data \
  --tasks dip_disease drg_identity_amount \
  --dip-raw-path respay/data/samples/dip/2023-2024_DIP_synthetic_data.csv \
  --drg-raw-dir respay/data/samples/drg \
  --output-root results/sample/processed
```

Train the residual ResPay model:

```bash
python -m respay.cli.train_respay \
  --processed-root results/sample/processed --output-dir results/sample/respay \
  --seeds 42 43 --epochs 3 --device cpu --threads 1
```

Run the direct-target ablation (same model without residual targets):

```bash
python -m respay.cli.train_respay --direct-target \
  --processed-root results/sample/processed --output-dir results/sample/respay \
  --seeds 42 43 --epochs 3 --device cpu --threads 1
```

Run Current standard, Ridge, MLP, and XGB:

```bash
python -m respay.cli.run_baselines \
  --processed-root results/sample/processed --output-dir results/sample/baselines \
  --seeds 42 43
```

Export numeric tables after all three model/baseline prediction files have been generated:

```bash
python -m respay.cli.analyze_results \
  --main-predictions results/sample/respay/residual/predictions.csv \
  --direct-predictions results/sample/respay/direct_target/predictions.csv \
  --baseline-predictions results/sample/baselines/predictions.csv \
  --processed-root results/sample/processed --output-dir results/sample/tables
```

Use `--help` on any command for path, task, seed, and device options. The commands above write processed data, predictions and analysis tables under `results/sample/`. `results/` and model checkpoints are ignored by Git.

## Classification score thresholds

The final correction rule is inclusive: apply the numerical prediction when the LR
score is at least the task threshold, otherwise retain the current standard. The
score threshold is **0 for DIP** and **0.75 for DRG**, for both residual and direct-target
models. DIP therefore applies every predicted correction while retaining the jointly
trained classifier for LR identification. DRG logits are aggregated with source-period
insurance shares before sigmoid and thresholding. These score thresholds are distinct
from the observed LR label boundaries, **0.45 for DIP** and **0.25 for DRG**.
Checkpoints, run metadata and analysis provenance record the task-specific thresholds.
These values were selected from the present empirical sensitivity experiments. For new
data, compare candidate thresholds on validation data and select the best value for
the intended error criterion before final evaluation.

## Result files

The analysis stage validates exact held-out task/seed/code pairing, with an absolute
tolerance of `1e-6` for true targets and current-standard priors to accommodate
float32 model outputs versus float64 cohort tables. It preserves each method's
saved targets when calculating its errors. Summary tables report the mean and
sample standard deviation (`ddof=1`) over seeds.

- `forecast_all_predictions.csv`, `forecast_all_by_seed.csv`, `forecast_all_summary.csv`: predictions and MAE, RMSE, and R2 for ResPay and each baseline, by seed and summarized across seeds.
- `forecast_all_reductions_by_seed.csv`, `forecast_all_reductions_summary.csv`: ResPay MAE/RMSE reduction percentages against each baseline.
- `forecast_all_paired_errors_by_seed.csv`, `forecast_all_paired_errors_summary.csv`: per-disease paired absolute errors and ResPay improvements against each baseline, including win/loss/tie summaries.
- `identification_by_seed.csv`, `identification_summary.csv`, `identification_scores.csv`, `pr_roc_curves.csv`: LR disease identification metrics (average precision and ROC AUC), per-disease scores, and numeric PR/ROC curve coordinates.
- `ablation_all_by_seed.csv`, `ablation_all_summary.csv`, `paired_direct_all_by_seed.csv`, `paired_direct_all_summary.csv`: residual versus direct-target regression metrics and paired seed-level differences.
- `figure1_group_updates_and_exposure.csv`, `figure1_exposure_summary.csv`, `figure1_cohort_summary.csv`: held-out disease updates, group/case counts, and exposure summaries used in the paper's numeric cohort/exposure results. These are data tables only; no figures are generated.
- `source_data_provenance.json`: hashes of prediction inputs and processed cohort sources, verified pairing checks, seeds, thresholds, and output hashes.
- `gate_threshold_sensitivity_by_seed.csv`, `gate_threshold_sensitivity_summary.csv`:
  all 21 score thresholds from 0 to 1 in steps of 0.05, with frozen model scores and
  numerical forecasts; includes error metrics, selected counts/fractions and LR recall.
- `gate_by_seed.csv`, `gate_summary.csv`: the selected DIP 0 / DRG 0.75 policy.

The sample preparation command writes compact model inputs and metadata under `results/sample/processed/`. For DIP, the disease split contains the selected train/test rows and code lists. For DRG, outputs include disease-code and identity-unit splits, a compact disease summary, and split/normalization metadata. Temporary intermediate tables are not part of the public output contract.
