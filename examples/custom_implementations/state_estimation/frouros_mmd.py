# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader
from sklearn.gaussian_process.kernels import RBF
from frouros.unsupervised.distance_based import MMD
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from automated_retraining.models import BaseModel
from automated_retraining.state_estimation.base import StateEstimation


class FrourosMaximumMeanDiscrepancy(StateEstimation):
    supervision_level = "un-supervised"

    def __init__(
        self,
        num_permutations: int = 1000,
        sigma: float = 500.0,
        random_state: Optional[int] = None,
        device: str = "cuda",
    ) -> None:
        """
        Class wrapping Frouros' implementation of Maximum Mean Discrepancy (MMD)
        using a Radial Basis Function (RBF) kernel.
        Note: best to use sigma = np.median(pdist(X=np.vstack((X_ref, X_test)), metric="euclidean")) / 2

        Source:
        https://github.com/jaime-cespedes-sisniega/frouros
        """
        super().__init__()
        self.device: str = device

        if random_state is not None:
            self.detector = MMD(
                num_permutations=num_permutations,
                kernel=RBF(length_scale=sigma),
                random_state=random_state,
            )
        else:
            self.detector = MMD(
                num_permutations=num_permutations, kernel=RBF(length_scale=sigma)
            )

    def __get__data(self, dataloader: DataLoader) -> np.ndarray:
        """
        Convert dataloader to numpy array.

        Args:
            dataloader (DataLoader): Any PyTorch dataloader

        Returns:
            np.ndarray: Numpy array containing the dataset from the dataloader
        """
        data = []
        for i, batch in enumerate(dataloader):
            batch, _ = batch
            data.append(batch.numpy())
        data = np.concatenate(data)
        if len(data.shape) > 2:
            ## LJW NOTE: Frouros doesn't like 2D or 3D input, concatenating/flattening
            data = np.reshape(data, (data.shape[0], -1))
        return data

    def get_sigma(self, id_dataloader: DataLoader, ood_dataloader: DataLoader) -> float:
        """Optimally calculate the sigma value, given an in-distribution and out-of-distribution dataset
        using sigma = np.median(pdist(X=np.vstack((X_ref, X_test)), metric="euclidean")) / 2

        Args:
            id_dataloader (DataLoader): DataLoader containing in-distribution data
            ood_dataloader (DataLoader): DataLoader containing out-of-distribution data

        Returns:
            float: Optimized sigma value
        """
        id_data = self.__get__data(id_dataloader)
        ood_data = self.__get__data(ood_dataloader)
        sigma = (
            np.median(pdist(X=np.vstack((id_data, ood_data)), metric="euclidean")) / 2
        )
        return sigma

    def reset(self, dataloader: DataLoader) -> None:
        """
        Re-fit MMD to the given dataset

        Args:
            dataloader (DataLoader): A PyTorch dataloader containing the data to fit the MMD algorithm.
        """
        data = self.__get__data(dataloader)
        self.detector.fit(data)

    def check_if_retraining_needed(
        self, model: BaseModel, dataloader: DataLoader, alpha: float = 0.4
    ) -> bool:
        """
        Check whether retraining is currently needed based on MMD.

        Args:
            model (BaseModel): Model under test.
            dataloader (DataLoader): PyTorch dataloader containing data for calibration error metric calculation.
            alpha (float, optional): P-value threshold. Defaults to 0.2.

        Raises:
            NotImplementedError: If not implemented in inheriting class.

        Returns:
            bool: Specifying whether or not retraining is needed.
        """
        data = self.__get__data(dataloader)
        self.detector.transform(X=data)
        mmd, p_value = self.detector.distance
        print("MMD:", mmd, "p-value:", p_value)

        if p_value < alpha:
            return True
        else:
            return False
