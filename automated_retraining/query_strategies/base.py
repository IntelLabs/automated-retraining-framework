# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np
from torch.utils.data import DataLoader

from automated_retraining.models.base_model import BaseClassifier


class BaseQuery:
    def execute(
        self, learner: BaseClassifier, dataloader: DataLoader, n_instances: int
    ) -> np.ndarray:
        """
        Base module/function for running all active learning query strategies.
        Selects the top n_instances from X based on some criteria which may
        or may not use the model (learner). Returns the indices into X corresponding
        to the queried samples.

        Args:
            learner (PyTorch model): The PyTorch model currently active learning.
            dataloader (PyTorch DataLoader): A dataloader containing the pool of datapoints from which to query.
            n_instances (int): The number of intances to query from X.

        Raises:
            NotImplementedError: Needs to be implemented in inheriting modules/classes.
        """
        raise NotImplementedError
