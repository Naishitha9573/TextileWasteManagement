import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from learning_rate_tuning import (  # noqa: E402
    DEFAULT_FINE_TUNING_DEPTH,
    DEFAULT_DROPOUT,
    DEFAULT_OPTIMIZER,
    BASELINE_METRICS,
    SUPPORTED_DROPOUTS,
    SUPPORTED_OPTIMIZERS,
    build_best_metrics,
    build_optimizer,
    create_model,
    fine_tuning_feature_indices,
    optimizer_parameter_groups,
    set_trainable_layers,
    trainable_parameter_count,
)
from model import CLASS_NAMES
from torch import nn
import torch
from torch.optim import Adam, AdamW, SGD


EXPECTED_INDICES = {
    "classifier_only": (),
    "last_block": (8,),
    "last_2_blocks": (7, 8),
    "last_3_blocks": (6, 7, 8),
}


@pytest.fixture
def model():
    return create_model(len(CLASS_NAMES), pretrained=False)


def test_depth_mapping():
    for depth, indices in EXPECTED_INDICES.items():
        assert fine_tuning_feature_indices(depth) == indices


def test_default_depth_preserves_current_behavior():
    assert DEFAULT_FINE_TUNING_DEPTH == "last_3_blocks"


@pytest.mark.parametrize("depth, expected_indices", EXPECTED_INDICES.items())
def test_trainability_and_parameter_counts(model, depth, expected_indices):
    set_trainable_layers(model, "classifier_only")
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert all(not parameter.requires_grad for parameter in model.features.parameters())

    set_trainable_layers(model, depth)
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    for index, feature in enumerate(model.features):
        expected = index in expected_indices
        assert all(parameter.requires_grad is expected for parameter in feature.parameters())


def test_parameter_counts(model):
    expected_counts = {
        "classifier_only": 11_529,
        "last_block": 423_689,
        "last_2_blocks": 1_140_921,
        "last_3_blocks": 3_167_269,
    }
    for depth, expected_count in expected_counts.items():
        set_trainable_layers(model, depth)
        assert trainable_parameter_count(model) == expected_count


@pytest.mark.parametrize("optimizer_name, optimizer_type", [("adamw", AdamW), ("adam", Adam), ("sgd", SGD)])
def test_build_optimizer_uses_selected_type_and_parameter_groups(model, optimizer_name, optimizer_type):
    feature_indices = set_trainable_layers(model, "last_2_blocks")
    groups = optimizer_parameter_groups(model, feature_indices, 1e-3, 5e-5)
    optimizer = build_optimizer(optimizer_name, groups, 1e-5)

    assert isinstance(optimizer, optimizer_type)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 5e-5]
    assert all(parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"])
    assert all(group["weight_decay"] == 1e-5 for group in optimizer.param_groups)


def test_optimizer_defaults_and_supported_choices():
    assert DEFAULT_OPTIMIZER == "adamw"
    assert SUPPORTED_OPTIMIZERS == ("adamw", "adam", "sgd")


def test_default_dropout_preserves_current_classifier_behavior(model):
    assert DEFAULT_DROPOUT == 0.2
    assert isinstance(model.classifier[0], nn.Dropout)
    assert model.classifier[0].p == DEFAULT_DROPOUT


@pytest.mark.parametrize("dropout", SUPPORTED_DROPOUTS)
def test_supported_dropout_is_applied(dropout):
    configured_model = create_model(len(CLASS_NAMES), pretrained=False, dropout=dropout)

    assert isinstance(configured_model.classifier[0], nn.Dropout)
    assert configured_model.classifier[0].p == dropout
    assert isinstance(configured_model.classifier[1], nn.Linear)


def test_build_best_metrics_uses_actual_baseline_keys(model):
    config = type("Config", (), {"learning_rate": 5e-5, "optimizer": "adamw", "dropout": 0.0, "fine_tuning_depth": "last_2_blocks"})()
    history = [{"epoch": 3, "validation_weighted_f1": 0.9}]
    metrics = build_best_metrics(config, model, torch.device("cpu"), 3, (0.88, 0.89, 0.87), history, (7, 8), 1.234)

    assert metrics["baseline_validation_accuracy"] == BASELINE_METRICS["validation_accuracy"]
    assert metrics["baseline_validation_macro_f1"] == BASELINE_METRICS["validation_macro_f1"]
    assert metrics["baseline_validation_weighted_f1"] == BASELINE_METRICS["validation_weighted_f1"]
    assert metrics["improvement_macro_f1_percentage_points"] == (0.88 - BASELINE_METRICS["validation_macro_f1"]) * 100