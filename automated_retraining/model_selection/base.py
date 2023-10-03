# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import List

import numpy as np


class ModelSelection:
    def __init__(self, **kwargs):
        """
        Base class for model selection criteria.
        """
        self.dataloader = kwargs.pop("dataloader")
        kwargs = {}

    def run_model_selection(self, models: List) -> int:
        """
        Select the best model from a list according the specified criteria.

        Args:
            models (List): A list of PyTorch models to be scored/selected from.

        Returns:
            int: The index into the models list specifying the best model.
        """
        metrics = np.zeros(len(models))
        for j, model in enumerate(models):
            metric = self.compute_metric(model)
            metrics[j] = metric
            print(f"{self._name} scores: Model {model.chkpt_name} {metric}")

        select_model_idx = self.selection_func(metrics)
        return select_model_idx

    def compute_metric():
        """
        Compute the model selection metric.

        Raises:
            NotImplementedError: Needs to be implemented by inheriting modules/classes.
        """
        raise NotImplementedError
