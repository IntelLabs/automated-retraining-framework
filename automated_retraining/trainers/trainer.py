# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from types import SimpleNamespace
from typing import Callable, Optional, Union

from automated_retraining.datasets import LearnSet, QuerySet, TrainSet
from automated_retraining.models.base_model import BaseModel
from automated_retraining.trainers.base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        training_config: SimpleNamespace,
        train_loader: Callable = None,
        val_loader: Optional[Callable] = None,
        **kwargs
    ) -> None:
        """Training module used to implement the train/test/validation loop.

        Args:
            model (BaseModel): Model to train
            training_config (SimpleNamespace): Parameters used for training
            train_loader (Callable): When called, sets up the training dataloader. Defaults to None
            val_loader (Optional[Callable]): When called, sets up the val dataloader. Defaults to None.
        """
        super().__init__(
            model,
            training_config,
            train_loader=train_loader,
            val_loader=val_loader,
            **kwargs
        )
        self.chkpt_prefix = "_".join(
            [
                self.model.architecture,
                training_config.dataset_name,
                training_config.experiment,
                "epoch",
            ]
        ).lower()
