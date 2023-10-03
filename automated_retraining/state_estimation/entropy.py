# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from collections import defaultdict
from importlib.util import module_for_loader

import numpy as np
import torch
from scipy.stats import entropy
from torch import nn
from torch.utils.data import DataLoader

from automated_retraining.models import BaseModel
from automated_retraining.state_estimation.base import StateEstimation


class Entropy(StateEstimation):
    supervision_level = "un-supervised"

    def __init__(self, device: str = "cuda") -> None:
        """
        Model state estimation based on the entropy of the softmax outputs.
        """
        super().__init__()
        self.state_metrics = defaultdict(list)
        self.device: str = device

    def get_softmax(self, model: BaseModel, dataloader: DataLoader) -> np.ndarray:
        """
        Get the softmax outputs given a PyTorch model and dataloader.

        Args:
            dataloader (DataLoader): Dataloader to use for prediction

        Returns:
            np.ndarray: Softmax output of model.
        """
        y_pred = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = [batch.to(self.device) for batch in batch]
                if isinstance(list(model.children())[-1], nn.Softmax):
                    outputs = model.forward(batch)
                else:
                    outputs = nn.Softmax(dim=1)(model.forward(batch))
                for y in outputs:
                    y_pred.append([y.cpu().numpy()])

        y_pred = np.concatenate(y_pred, 0)

        return y_pred

    def get_entropy(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Wrapper for the sklearn entropy function.

        Args:
            y_pred (np.ndarray): Softmax outputs

        Returns:
            np.ndarray: Numpy array containing the entropies of each softmax output.
        """
        e = entropy(y_pred, axis=1)
        return e


class AverageEntropy(Entropy):
    supervision_level = "un-supervised"

    def __init__(self, device: str = "cuda") -> None:
        """
        Model state estimation based on the entropy of the softmax outputs.
        """
        super().__init__()

    def check_if_retraining_needed(
        self, model: BaseModel, dataloader: DataLoader, threshold: float = 0.5
    ) -> bool:
        """
        Check whether retraining is currently needed based on the entropy of the
        softmax outputs.

        Args:
            model (BaseModel): Model under test.
            dataloader (DataLoader): PyTorch dataloader containing data for calibration error metric calculation.
            threshold (float, optional): The threshold of tolerated entropy. Defaults to 0.5.

        Raises:
            NotImplementedError: If not implemented in inheriting class.

        Returns:
            bool: Specifying whether or not retraining is needed.
        """
        y_pred = self.get_softmax(model, dataloader)
        e = self.get_entropy(y_pred)
        e_avg = np.mean(e)
        print("Average Entropy: ", e_avg)

        if e_avg > threshold:
            return True
        else:
            return False
