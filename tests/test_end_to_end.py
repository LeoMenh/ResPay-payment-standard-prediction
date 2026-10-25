"""End-to-end CLI orchestration contracts with bounded synthetic stages."""

import sys

import pandas as pd
from respay.cli import run_all


def test_run_all_forwards_options_and_runs_stages_in_order(monkeypatch, tmp_path):
    calls = []
    raw_dip = tmp_path / "dip.csv"
    raw_drg = tmp_path / "drg"
    output = tmp_path / "output"
    processed = output / "processed"

    def export_dip(raw, destination):
        calls.append(("prepare_dip", raw, destination))
        return {}

    def export_drg(destination, raw):
        calls.append(("prepare_drg", raw, destination))
        return {}

    def train(task, seed, epochs, device, folder, use_residual_target=True, processed_root=None):
        calls.append(
            ("train", task, seed, epochs, str(device), use_residual_target, processed_root)
        )
        return (pd.DataFrame({"task": [task], "seed": [seed]}), None)

    def baselines(tasks, seeds, destination, root):
        calls.append(("baselines", tasks, seeds, destination, root))
        return destination / "predictions.csv"

    def analyze(main, direct, baseline, root, destination):
        calls.append(("analyze", main, direct, baseline, root, destination))
        return []

    monkeypatch.setattr(run_all, "export_dip_datasets", export_dip)
    monkeypatch.setattr(run_all, "export_drg_datasets", export_drg)
    monkeypatch.setattr(run_all, "train_one", train)
    monkeypatch.setattr(run_all, "run_baselines", baselines)
    monkeypatch.setattr(run_all, "analyze_results", analyze)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all",
            "--dip-raw-path",
            str(raw_dip),
            "--drg-raw-dir",
            str(raw_drg),
            "--output-root",
            str(output),
            "--device",
            "cpu",
            "--seeds",
            "7",
            "8",
            "--epochs",
            "2",
        ],
    )

    run_all.main()

    assert [call[0] for call in calls] == [
        "prepare_dip",
        "prepare_drg",
        *("train",) * 8,
        "baselines",
        "analyze",
    ]
    assert calls[0] == ("prepare_dip", raw_dip, processed / "dip_processed")
    assert calls[1] == ("prepare_drg", raw_drg, processed / "drg_processed")
    train_calls = [call for call in calls if call[0] == "train"]
    assert [(call[2], call[5]) for call in train_calls] == [
        (7, True),
        (8, True),
        (7, True),
        (8, True),
        (7, False),
        (8, False),
        (7, False),
        (8, False),
    ]
    assert all(call[3] == 2 and call[4] == "cpu" for call in train_calls)
    assert all(call[6] == processed for call in train_calls)
    assert calls[-1][1:] == (
        output / "respay/residual/predictions.csv",
        output / "respay/direct_target/predictions.csv",
        output / "baselines/predictions.csv",
        processed,
        output / "tables",
    )


def test_analysis_cli_help_is_available():
    from respay.cli import analyze_results

    assert callable(analyze_results.main)
