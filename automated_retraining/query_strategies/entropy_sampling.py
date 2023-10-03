# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np  # type: ignore
import torch
from scipy.stats import entropy
from torch.utils.data import DataLoader

from automated_retraining.models.base_model import BaseClassifier

from automated_retraining.query_strategies.base import BaseQuery


class EntropySampling(BaseQuery):
    def execute(
        learner: BaseClassifier, dataloader: DataLoader, n_instances: int = 1
    ) -> np.ndarray:
        """
        Query using entropy sampling. Selects instances with highest entropy in softmax output
        which correlates with least certainty in the decision.

        Adapted from modAL code.

        Args:
            learner (PyTorch model): The PyTorch model currently active learning.
            dataloader (PyTorch DataLoader): A dataloader containing the pool of datapoints from which to query.
            n_instances (int): The number of intances to query from X. Defaults to 1.

        Returns:
            np.ndarray: Indices into X corresponding to the queried samples
        """
        assert n_instances <= len(dataloader.dataset)

        ## get softmax for each sample
        with torch.no_grad():
            outputs = learner.predict_proba(dataloader)
        ## get entropy
        e = entropy(outputs, axis=1)
        ## shuffle values in case of ties
        shuffle_idx = np.random.permutation(len(e))
        shuffle_e = e[shuffle_idx]
        ## sort according to e, low to high
        sorted_idx = np.argsort(shuffle_e)
        ## invert_shuffle
        query_idx = shuffle_idx[sorted_idx[-n_instances:]]
        return query_idx
