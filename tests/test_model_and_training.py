"""Behavioral tests for the selected ResPay model and prediction workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _bundle(task: str):
    from respay.dataset import DatasetConfig

    if task == "dip_disease":
        codes = list("ABCDEFGHIJKLMNOPQRST")
        prior = [10.0 + index for index in range(len(codes))]
        target = [value + (1.0 if index in {0, 3} else 0.0) for index, value in enumerate(prior)]
        train = pd.DataFrame(
            {
                "code": codes,
                "target": target,
                "prior": prior,
                "x": [float(index) for index in range(len(codes))],
            }
        )
        test = pd.DataFrame(
            {"code": ["E", "F"], "target": [51.0, 60.0], "prior": [50.0, 60.0], "x": [4.0, 5.0]}
        )
        config = DatasetConfig(
            task,
            Path("train"),
            Path("test"),
            "target",
            "code",
            ["x"],
            "prior",
            scale_features_in_loader=False,
        )
    else:
        rows = []
        test_rows = []
        for split_codes, destination in ((["A", "B", "C", "D"], rows), (["E", "F"], test_rows)):
            for index, disease in enumerate(split_codes):
                gap = 1.0 if disease in {"A", "D", "E"} else 0.0
                identities = (("employee", 0.75, index), ("resident", 0.25, index + 0.2))
                for identity, share, feature in identities:
                    current = 10.0 + index
                    destination.append(
                        {
                            "identity_code": f"{disease}_{identity}",
                            "target_identity_amount": current + gap,
                            "current_identity_amount": current,
                            "x": feature,
                            "disease_code": disease,
                            "identity_type": identity,
                            "identity_weight": share,
                            "target_group_amount": current + gap,
                            "current_group_amount": current,
                        }
                    )
        train = pd.DataFrame(rows)
        test = pd.DataFrame(test_rows)
        config = DatasetConfig(
            task,
            Path("train"),
            Path("test"),
            "target_identity_amount",
            "identity_code",
            ["x"],
            "current_identity_amount",
            scale_features_in_loader=False,
        )

    from respay.dataset import build_dataset_bundle_from_frames

    return build_dataset_bundle_from_frames(config, train, test)


def test_attention_weights_and_predictions_are_group_local():
    from respay.model import ResPay

    torch.manual_seed(3)
    model = ResPay(numeric_state_dim=2, hidden_dim=8).eval()
    first = torch.tensor([[0.0, 1.0], [1.0, 0.0], [8.0, 9.0]])
    second = first.clone()
    second[-1] = torch.tensor([-100.0, 100.0])
    groups = torch.tensor([0, 0, 1])
    with torch.no_grad():
        output_a = model(first, groups, num_groups=2)
        output_b = model(second, groups, num_groups=2)
    torch.testing.assert_close(output_a.prediction[:1], output_b.prediction[:1])
    torch.testing.assert_close(output_a.lr_logit[:1], output_b.lr_logit[:1])
    torch.testing.assert_close(output_a.attention_weights[:2].sum(), torch.tensor(1.0))
    torch.testing.assert_close(output_a.attention_weights[2:], torch.tensor([1.0]))


def test_identity_logits_are_weighted_before_sigmoid_and_gate_is_inclusive():
    from respay.trainer import GATE_THRESHOLDS, aggregate_identity_by_disease, apply_gate

    units = pd.DataFrame({"disease_code": ["A", "A"], "identity_weight": [0.75, 0.25]})
    aggregated, _, codes = aggregate_identity_by_disease(
        torch.tensor([4.0, -2.0]), units, torch.device("cpu")
    )
    assert codes == ["A"]
    expected = torch.sigmoid(torch.tensor(2.5)).item()
    assert torch.sigmoid(aggregated).item() == pytest.approx(expected)
    assert torch.sigmoid(aggregated).item() != pytest.approx(
        (0.75 * torch.sigmoid(torch.tensor(4.0)) + 0.25 * torch.sigmoid(torch.tensor(-2.0))).item()
    )
    np.testing.assert_array_equal(
        apply_gate(
            np.array([8.0, 8.0]),
            np.array([3.0, 3.0]),
            np.array([0.75, 0.74999]),
            threshold=GATE_THRESHOLDS["drg_identity_amount"],
        ),
        np.array([8.0, 3.0]),
    )


@pytest.mark.parametrize("task", ["dip_disease", "drg_identity_amount"])
@pytest.mark.parametrize("use_residual_target", [True, False])
def test_one_epoch_training_predicts_and_checkpoint_round_trips(
    tmp_path, task, use_residual_target
):
    from respay.model import ResPay
    from respay.trainer import predict_groups, train_one

    bundle = _bundle(task)
    frame, metadata = train_one(
        task,
        seed=42,
        epochs=1,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        use_residual_target=use_residual_target,
        bundle=bundle,
    )
    assert len(frame) == 2
    assert frame["seed"].eq(42).all()
    assert frame["task"].eq("DIP" if task == "dip_disease" else "DRG").all()
    assert frame["gated_prediction"].notna().all()
    checkpoint = torch.load(metadata["checkpoint_path"], map_location="cpu", weights_only=False)
    restored = ResPay(len(bundle.feature_cols), checkpoint["config"]["hidden_dim"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored_frame = predict_groups(
        restored,
        bundle,
        torch.device("cpu"),
        checkpoint["residual_mean"],
        checkpoint["residual_std"],
        use_residual_target,
    )
    np.testing.assert_allclose(restored_frame["prediction"], frame["prediction"], rtol=1e-6)
    np.testing.assert_allclose(restored_frame["lr_probability"], frame["lr_probability"], rtol=1e-6)
    assert checkpoint["use_residual_target"] is use_residual_target
    expected_threshold = 0.0 if task == "dip_disease" else 0.75
    assert checkpoint["gate_threshold"] == expected_threshold
    assert metadata["decision_probability_threshold"] == expected_threshold
    assert Path(metadata["checkpoint_path"]).exists()


@pytest.mark.parametrize("method", ["Current standard", "Ridge", "MLP", "XGB"])
def test_selected_baselines_return_group_keyed_predictions(method):
    from respay.baselines import baseline_predictions

    bundle = _bundle("dip_disease")
    frame = baseline_predictions(bundle, method, seed=42)
    assert frame["code"].tolist() == ["E", "F"]
    assert frame["method"].eq(method).all()
    assert np.isfinite(frame["prediction"]).all()


def test_zero_score_threshold_applies_all_corrections():
    from respay.trainer import apply_gate

    np.testing.assert_array_equal(
        apply_gate(
            np.array([8.0, 9.0]), np.array([3.0, 4.0]), np.array([0.0, 0.49]), threshold=0.0
        ),
        [8.0, 9.0],
    )


def test_checkpoint_inference_uses_current_policy_and_checks_feature_order(tmp_path):
    from respay.trainer import predict_checkpoint, train_one

    bundle = _bundle("drg_identity_amount")
    expected, metadata = train_one(
        "drg_identity_amount", 42, 1, torch.device("cpu"), tmp_path, bundle=bundle
    )
    path = Path(metadata["checkpoint_path"])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # Historical DRG weights recorded the old inference gate; fitting is unchanged.
    saved["gate_threshold"] = 0.5
    torch.save(saved, path)
    actual = predict_checkpoint(path, bundle, torch.device("cpu"))
    pd.testing.assert_frame_equal(actual, expected)
    saved["feature_cols"] = ["wrong_feature"]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="feature"):
        predict_checkpoint(path, bundle, torch.device("cpu"))
