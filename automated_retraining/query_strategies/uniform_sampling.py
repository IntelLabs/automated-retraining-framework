# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np  # type: ignore
from torch.utils.data import DataLoader

from automated_retraining.models.base_model import BaseClassifier

from automated_retraining.query_strategies.base import BaseQuery


class UniformSampling(BaseQuery):
    def execute(
        learner: BaseClassifier, dataloader: DataLoader, n_instances: int = 1
    ) -> np.ndarray:
        """
        Randomly query samples using the uniform distribution. Note that this strategy
        does not make use of the learner.

        Args:
            learner (PyTorch model): The PyTorch model currently active learning.
            dataloader (PyTorch DataLoader): A dataloader containing the pool of datapoints from which to query.
            n_instances (int): The number of intances to query from X. Defaults to 1.

        Returns:
            np.ndarray: Indices into X corresponding to the queried samples
        """
        assert n_instances <= len(dataloader.dataset)
        query_idx = np.random.choice(
            range(len(dataloader.dataset)), size=n_instances, replace=False
        )
        return query_idx
