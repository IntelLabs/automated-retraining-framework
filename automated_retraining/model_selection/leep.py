# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np

from automated_retraining.model_selection.base import ModelSelection
from automated_retraining.models.base_model import BaseClassifier


## customized for AL setup
class LEEP(ModelSelection):
    def __init__(self, target_classes: int, source_classes: int, **kwargs):
        """
        Computes the Log Expected Empirical Prediction (LEEP) score.

        Source:
            https://arxiv.org/abs/2002.12462


        Args:
            target_classes (int): The number of classes in the target task.
            source_classes (int): The number of classes in the source task.
        """
        self._name: str = "LEEP"
        self.selection_func = np.argmax
        self.n_target_classes: int = target_classes
        self.n_source_classes: int = source_classes
        super().__init__(**kwargs)

    def compute_metric(self, model: BaseClassifier) -> float:
        """
        Compute the LEEP score for a give model and dataset.

        Args:
            model: The PyTorch model being evaluated

        Returns:
            float: The LEEP score. A value between negative infinity and zero.
        """
        # Step 1: Compute dummy label distributions of the inputs in the target data set
        ## model outputs = dummy categorical distribution over Z
        ## y = truth labels
        theta_x, y = model.predict_proba(
            self.dataloader, with_labels=True
        )  ## get model(samples) with softmax
        n = len(y)
        # Step 2: Compute the empirical conditional distribution of the target label given the source label
        ##  Compute empirical joint distribution P(y, z) for label pair y, z
        p_y_z = np.zeros((self.n_target_classes, self.n_source_classes))
        for i_y in range(self.n_target_classes):
            for i_z in range(self.n_source_classes):
                idx_iy = np.argwhere(y == i_y)  ## all examples of target class y
                curr_p = np.sum(theta_x[idx_iy, i_z].flatten()) / n
                p_y_z[i_y, i_z] = curr_p
        ## Compute empirical marginal distribution P(z)
        p_z = np.sum(p_y_z, axis=0)
        ## Compute empirical conditional distribution P(y|z)
        p_yGz = np.divide(p_y_z, np.tile(p_z, (self.n_target_classes, 1)))

        # Step 3: Compute LEEP
        eep = np.zeros(n)
        for i in range(n):
            eep[i] = np.sum(np.multiply(p_yGz[y[i]], theta_x[i]))
        leep = np.sum(np.log(eep)) / n

        return leep
