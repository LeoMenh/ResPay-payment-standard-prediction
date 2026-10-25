# Reproducing the manuscript experiments

Verified on 25 October 2026 using this standalone package. The manuscript was left
unchanged. Raw hospital files and trained weights are local artifacts, excluded from
Git. Reproducing the published numbers requires access to those files.

## Verified scope

| Stage | Verification |
| --- | --- |
| DIP preparation | Rebuilt from `df_sample-1.csv`; all model input tensors, targets, priors and group indices exactly match the archived split. |
| DRG preparation | Rebuilt from the three annual case workbooks and two rate tables; all model tensors exactly match the archive, including identity shares. |
| DIP refitting | Both variants, all five seeds, 70 CPU epochs and all four baselines reproduce the archived predictions to `1e-12`. |
| DRG checkpoint inference | All ten retained CUDA-trained weights replay on CPU; maximum prediction difference `4.77e-7`. Current inference uses threshold 0.75. |
| Numeric analysis | All 22 tables from archived predictions agree with the manuscript source tables to absolute tolerance `1e-10`. Checkpoint replay also reproduces all 22 tables within floating-point tolerances (`2e-6`, or `2e-5` percentage points for reductions). |
| CPU refitting of DRG | Both variants and all baselines execute for all five seeds. Neural refits differ from the original CUDA fits; these are saved as a separate experiment. |

Seeds are 42--46. Each seed is evaluated separately, then summarized with mean and
sample SD (`ddof=1`). There is no averaging of predictions across seeds. The LR label
boundaries are DIP 0.45 and DRG 0.25. Inclusive inference gates are DIP 0 and DRG 0.75.
The sensitivity experiment varies only the score gate, using frozen trained outputs.

The verified DIP model inputs have 11,286 training records / 877 groups and 3,956
test records / 219 groups. DRG has 601 training insurance units and 141 test units;
evaluation aggregates the latter to 69 diseases. All units for a disease remain in
one partition. Original source case counts are retained in preprocessing metadata.

## Environment

Use Python 3.12.2 with `requirements-reproduction.txt` for the verified environment.
The tested torch runtime is `2.9.1+cu128`; no usable CUDA device was available during
this audit. Install from the `ResPay/` directory:

```bash
python -m pip install -e '.[dev]' -c requirements-reproduction.txt
```

For CPU refitting, use one thread for PyTorch and numerical libraries:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

The neural CLIs default to `--threads 1`; CPU training enables deterministic
algorithms. Exact historical DRG CUDA refitting has not been verified in this audit.
For example, the five-seed residual model refitted on CPU gives DRG MAE 0.170276 and
RMSE 0.604638, versus the manuscript's 0.170243 and 0.604844. Switching the score
threshold to 0.75 does not require refitting weights and does not remove training
differences between devices.

## Rebuild inputs on this server

All commands below run from `/home/liangyuyang/DRG-DIP/ResPay`. These are the source
files actually verified, rather than the optional default `data/raw/` locations:

```bash
python -m respay.cli.prepare_data \
  --dip-raw-path '/home/liangyuyang/Data/DIP data/df_sample-1.csv' \
  --drg-raw-dir '/home/liangyuyang/Data/DRG data' \
  --output-root results/reproduction_audit/processed
```

Preparation exports compact model inputs without patient names. The retained DIP
CSV is the starting input for the published pipeline; the package consumes its
existing fee/feature scale without fitting a new scaler on DIP.

## Replay the published weights

The audit assembled a local bundle under `results/paper_reference/` with 20 unchanged
checkpoints and the three archived prediction files. Its `manifest.json` records
their SHA-256 hashes. This bundle is ignored by Git; copying only the source code
does not transfer it. Keep it with the authorized local reproduction materials.

The checkpoint tree is:

```text
results/paper_reference/checkpoints/
  {residual,direct_target}/
    {dip_disease,drg_identity_amount}/
      seed_{42,43,44,45,46}/<task>_seed_<seed>_model.pt
```

DIP weights come from
`../Data/experiment_outputs/results_gate_dip0_drg075_20261025/{residual,direct_target}/`.
DRG weights come from
`../Data/experiment_outputs/respay_lr_head_no_recurrence/` and
`../Data/experiment_outputs/respay_lr_head_no_recurrence_ablation/direct_target/`.
The archived baseline predictions come from
`../Data/experiment_outputs/results_gate_dip0_drg075_20261025/baselines/predictions.csv`.

Regenerate model predictions from these trusted weights:

```bash
python -m respay.cli.predict_checkpoints \
  --checkpoint-root results/paper_reference/checkpoints \
  --processed-root results/reproduction_audit/processed \
  --output-dir results/reproduction_audit/checkpoint_predictions_rebuilt \
  --device cpu --threads 1

python -m respay.cli.analyze_results \
  --main-predictions results/reproduction_audit/checkpoint_predictions_rebuilt/residual/predictions.csv \
  --direct-predictions results/reproduction_audit/checkpoint_predictions_rebuilt/direct_target/predictions.csv \
  --baseline-predictions results/paper_reference/archived_predictions/baselines/predictions.csv \
  --processed-root results/reproduction_audit/processed \
  --output-dir results/reproduction_audit/replayed_tables
```

The inference command checks feature order, seed, target variant and label boundary.
It intentionally applies the current task gates, overriding historical DRG checkpoint
metadata that says 0.5. It records checkpoint hashes and applied gates in
`checkpoint_provenance.json`; no weights are modified.

For numeric reproduction directly from the archived prediction files, point
`--main-predictions` and `--direct-predictions` to the corresponding files under
`results/paper_reference/archived_predictions/` and use a separate output directory.

## Refit all models

The `run_all` command documented in README rebuilds data, trains both neural
variants, fits four baselines, and exports all 22 tables. Use a fresh `--output-root`
to preserve the published reference artifacts. `--device cpu --threads 1` reproduces
the verified DIP fits and produces a separate DRG CPU experiment. To retrain tasks
on different devices, use the individual training commands with `--tasks` and
`--device`; both variants require their own invocation.

## Local evidence

`results/reproduction_audit/` contains the actual verification artifacts:

- `preprocessing_verification.json`: exact tensor comparisons against archived inputs.
- `dip_refit_verification.json`: five-seed from-scratch agreement for both variants and baselines.
- `drg_cpu_refit_verification.json`: measured differences between CPU refits and archived CUDA outputs.
- `tables_and_checkpoints_verification.json`: all 44 table comparisons (22 archived and 22 replayed), plus checkpoint prediction differences.
- `training/`, `archived_tables/`, `replayed_tables/`, `cpu_refit_tables/`: separate trained artifacts and numerical tables.

The package exports the numeric evidence for cohort/exposure analysis, forecasting,
paired errors, identification, ablation, and threshold sensitivity. Figure rendering
and LaTeX compilation live outside this code package.

Run `python -m pytest`, `ruff check .`, and `ruff format --check .` to check the
package. If an unrelated globally installed pytest plugin prevents collection, use
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`; the package tests need no plugins.
