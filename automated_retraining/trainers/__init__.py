# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
import time
from types import SimpleNamespace

import torch

from automated_retraining.trainers.base import BaseTrainer
from automated_retraining.trainers.active_trainer import ActiveTrainer
from automated_retraining.trainers.trainer import Trainer


# move this to an __init__.py when trainer module is implemented
def configure_training(
    training_config: SimpleNamespace,
    dataset_config: SimpleNamespace,
    model_config: SimpleNamespace,
) -> SimpleNamespace:
    """Sets up additional parameters for training.

    Args:
        training_config (SimpleNamespace): Training config
        dataset_config (SimpleNamespace): Dataset config/parameters
        model_config (SimpleNamespace): Model config/parameters

    Returns:
        SimpleNamespace: Updated training configuration
    """
    # Training Config Additions
    train_date = time.strftime("%Y-%m-%d", time.localtime())
    log_dir = os.path.join(
        training_config.results_dir,
        model_config.architecture,
        training_config.experiment,
        train_date,
    )
    try:
        version = len(os.listdir(log_dir))
    except FileNotFoundError:
        version = 0
    training_config.log_dir = os.path.join(log_dir, str(version))
    os.makedirs(training_config.log_dir)
    if torch.cuda.is_available() and training_config.device == "cuda":
        training_config.device = "cuda"
    return training_config
