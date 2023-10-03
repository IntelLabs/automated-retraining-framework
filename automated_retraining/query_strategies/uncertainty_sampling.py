# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np  # type: ignore
import torch
from torch.utils.data import DataLoader

from automated_retraining.models.base_model import BaseClassifier

from automated_retraining.query_strategies.base import BaseQuery


class UncertaintySampling(BaseQuery):
    def execute(
        learner: BaseClassifier, dataloader: DataLoader, n_instances: int = 1
    ) -> np.ndarray:
        """
        Query using uncertainty sampling. Selects instances with the highest difference
        between 1 and the max softmax value which correlates with uncertainty.

        Adapted from modAL code.

        Args:
            learner (PyTorch model): The PyTorch model currently active learning.
            dataloader (PyTorch DataLoader): A dataloader containing the pool of datapoints from which to query.
            n_instances (int): The number of intances to query from X. Defaults to 1.

        Returns:
            np.ndarray: Indices into X corresponding to the queried samples
        """
        assert n_instances <= len(dataloader.dataset)

        ## get uncertainty for each sample
        with torch.no_grad():
            outputs = learner.predict_proba(dataloader)
        uncertainty = 1 - np.max(outputs, axis=1)
        ## shuffle values in case of ties
        shuffle_idx = np.random.permutation(len(uncertainty))
        shuffle_uncertainty = uncertainty[shuffle_idx]
        ## sort according to uncertainty, low to high
        sorted_idx = np.argsort(shuffle_uncertainty)
        ## invert shuffle
        query_idx = shuffle_idx[sorted_idx[-n_instances:]]
        return query_idx
