# Synthetic example inputs

These public files are generated entirely from scratch by `respay.data.sample_data`.
Every disease code, group name, case date, feature value, weight, and payment amount
is invented. Column
labels, insurance category labels, and required Excel layouts follow
the input readers. Years and version labels are fixed format conventions.

There are 60 synthetic disease groups in each task. DIP has 420 records across
2023 and 2024. DRG has 360 records in each of 2023/2024 and 720 records in 2025,
including both DRG versions, plus two 60-group rate tables.

The files are named by year and task:

- `dip/2023-2024_DIP_synthetic_data.csv` (both years are needed for the target transition)
- `drg/2023_DRG_synthetic_data.xlsx`
- `drg/2024_DRG_synthetic_data.xlsx`
- `drg/2025_DRG_synthetic_data.xlsx`
- `drg/2024_DRG_synthetic_weights_rates_v1.1.xlsx`
- `drg/2025_DRG_synthetic_weights_rates_v2.0.xlsx`

All three sample base rates (9000, 7000, 8000) are invented.

From the ResPay root, run the complete demonstration:

```bash
python -m respay.cli.run_all \
  --dip-raw-path respay/data/samples/dip/2023-2024_DIP_synthetic_data.csv \
  --drg-raw-dir respay/data/samples/drg \
  --output-root results/sample --seeds 42 43 --epochs 3 --device cpu --threads 1
```

This runs preprocessing, both neural variants, all four baselines, and numeric
analysis. Outputs demonstrate the workflow only; they do not reproduce the
paper's results or measure real-world performance. All labels used as numerical
training targets are synthetic too. Rebuild the inputs with
`python -m respay.data.sample_data`; existing example files will be replaced.
