# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Union

import automated_retraining.datasets as datasets
import automated_retraining.models as models
from automated_retraining.submodules.submodule_utils import SubModule
from automated_retraining.trainers import Trainer, configure_training


class TrainingModule(SubModule):
    """Training module used to set up framework for training models.

    Args:
        SubModule (submodules.SubModule): Base submodule
    """

    def __init__(self, **kwargs):
        """Set up the training module. Config file typically passed in with kwargs."""
        self.__dict__.update(kwargs)
        self.parse_config(self.config)

    def run(self) -> None:
        """Sets up datasets, configs, and models, then run the training loop"""
        # Set up and additions to dataset, training config, and model.
        dataset: Union[
            datasets.LearnSet, datasets.QuerySet, datasets.TrainSet
        ] = datasets.configure_dataset(self.dataset_config, dataset_type="TrainSet")
        assert isinstance(
            dataset, datasets.TrainSet
        ), "Incorrect datasets assigned, should be datasets.TrainSet is {}".format(
            type(dataset)
        )
        ## LJW NOTE: assuming dataset labelled
        self.dataset_config.num_classes = dataset.num_classes()
        self.training_config = configure_training(
            self.training_config, self.dataset_config, self.model_config
        )
        model: models.BaseModel = models.configure_model(
            self.model_config,
            self.dataset_config,
            self.training_config,
            self.training_params,
        )

        # Trainer Setup and Start Training
        self.training_config.dataset_name = dataset.data_module.__class__.__name__
        train_loader = dataset.train_dataloader
        val_loader = dataset.val_dataloader
        trainer = Trainer(
            model,
            self.training_config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer.train_model()
        return
