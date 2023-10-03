# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from types import SimpleNamespace
from typing import Type, Union

from automated_retraining.datasets.cifar10_dataset import CIFAR10DataModule
from automated_retraining.datasets.dataset import BaseDataModule
from automated_retraining.datasets.mnist_dataset import MNISTDataModule
from automated_retraining.datasets.sets import LearnSet, QuerySet, TrainSet


def configure_dataset(
    dataset_config: SimpleNamespace, dataset_type: str = "TrainSet"
) -> Union[LearnSet, QuerySet, TrainSet]:
    # Dataset Config Additions and Setup
    data_module: Type[BaseDataModule] = globals()[dataset_config.datamodule]
    dataset: Type[Union[LearnSet, QuerySet, TrainSet]] = globals()[dataset_type]
    return dataset(data_module(**vars(dataset_config)))
