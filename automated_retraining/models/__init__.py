# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
from types import SimpleNamespace
from typing import Dict, Type, cast

import torch.optim as optim
import yaml

from automated_retraining.models.base_model import BaseClassifier, BaseModel
import automated_retraining.models.architectures
from automated_retraining.model_calibration.post_hoc.temperature_scaling import (
    TemperatureScaling,
)
from automated_retraining.models.active_model import ActiveModel
from automated_retraining.models.cifar10_model import CIFAR10Model
from automated_retraining.models.mnist_model import MNISTModel


def configure_model(
    model_config: SimpleNamespace,
    dataset_config: SimpleNamespace,
    training_config: SimpleNamespace,
    training_params: SimpleNamespace,
) -> BaseModel:
    """Configure a model based on config params. Adds some extra attributes to the
    namespace and moves model to the specified device.

    Args:
        model_config (SimpleNamespace): Model config/parameters
        dataset_config (SimpleNamespace): Dataset config/parameters
        training_config (SimpleNamespace): Training config
        training_params (SimpleNamespace): Training parameters

    Returns:
        BaseModel: The created model.
    """
    # Model Setup
    model_attr: Type[BaseModel] = globals()[model_config.model_name]
    model = model_attr(**{**vars(model_config)})
    model.load_model(
        model_config.architecture,
        cast(Dict, dataset_config.num_classes),
        device=training_config.device,
    )
    model.optimizer = optim.SGD(
        model.parameters(),
        lr=training_params.lr,
        weight_decay=training_params.weight_decay,
        momentum=training_params.momentum,
        nesterov=False,
    )
    model.scheduler = optim.lr_scheduler.MultiStepLR(
        model.optimizer, milestones=list(range(0, 201, 50)), gamma=training_params.gamma
    )
    if hasattr(model_config, "calibration"):
        calibration_attr = globals()[model_config.calibration]
        model.calibration = calibration_attr(**{**vars(model_config)})
    else:
        model.calibration = None
    with open(os.path.join(training_config.log_dir, "hyper_parameters.yaml"), "w") as f:
        yaml.safe_dump(vars(training_params), f)
    if training_config.device == "cuda":
        model.cuda()

    return model
