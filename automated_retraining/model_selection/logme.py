# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import List

import numpy as np

from automated_retraining.model_selection.base import ModelSelection
from automated_retraining.models.base_model import BaseClassifier


## Source: https://github.com/thuml/LogME
## customized for AL
class LogME(ModelSelection):
    def __init__(self, regression: bool = False, device: str = "cuda", **kwargs):
        """
        Computes the Logarithm of Maximum Evidence (LogME) score.

        Sources:
            paper: https://arxiv.org/abs/2102.11005
            code: https://github.com/thuml/LogME


        Args:
            regression (bool, optional): True if the model is a regression model. False otherwise. Defaults to False.
            device (str, optional): "cuda" to use GPU. "cpu" to use CPU. Defaults to "cuda".
        """
        self._name: str = "LogME"
        self.selection_func = np.argmax
        self.device: str = device
        self.regression: bool = regression
        super().__init__(**kwargs)

    def compute_metric(self, model: BaseClassifier) -> float:
        """
        Compute the LogME score for a given model and dataset

        Args:
            model: The PyTorch model being evaluated.

        Returns:
            float: The LogME score. A value between -1 and 1.
        """
        ## f: [N, F], feature matrix from pre-trained model
        ## y: target labels.
        ##    For classification, y has shape [N] with element in [0, C_t).
        ##    For regression, y has shape [N, C] with C regression-labels
        f = []
        y = []
        model.eval()
        for batch_idx, batch in enumerate(self.dataloader):
            batch = [batch.to(self.device) for batch in batch]
            _ = model(batch)
            f.append(model.model.penult.detach().cpu().numpy().astype(np.float64))
            y.append(batch[-1].cpu().numpy())
        model.train()
        f = np.concatenate(f, 0)
        y = np.concatenate(y, 0)

        if self.regression:
            y = y.astype(np.float64)

        fh = f  ## features
        f = f.transpose()  ## transpose of features
        D, N = f.shape  ## D = feature dim, N = batch size
        ## v = unitary arrays, s = singular values, vh = unitary arrays
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        if self.regression:
            K = y.shape[1]
            for i in range(K):
                y_ = y[:, i]
                evidence = self.__each__evidence(y_, f, fh, v, s, vh, N, D)
                evidences.append(evidence)
        else:
            K = int(y.max() + 1)  ## max label index + 1
            for i in range(K):
                y_ = (y == i).astype(int)  ## boolean vector indicating curr_label == i
                if np.sum(y_) == 0:
                    pass
                else:
                    evidence = self.__each__evidence(y_, f, fh, v, s, vh, N, D)
                    evidences.append(evidence)
        return np.mean(evidences)

    def __each__evidence(
        self,
        y_: np.ndarray,
        f: np.ndarray,
        fh: np.ndarray,
        v: np.ndarray,
        s: np.ndarray,
        vh: np.ndarray,
        N: int,
        D: int,
    ) -> np.ndarray:
        """
        Compute the maximum evidence for each class.

        Args:
            y_ (np.ndarray): boolean vector indicating whether or not curr_label == i
            f (np.ndarray): transpose of features
            fh (np.ndarray): features
            v (np.ndarray): unitary array(s) of f*fh
            s (np.ndarray): singluar values of f*fh
            vh (np.ndarray): unitary array(s) of f*fh
            N (int): batch size
            D (int): feature dimension

        Returns:
            np.ndarray : maximum evidence for each class
        """
        alpha = 1.0
        beta = 1.0
        lam = alpha / beta
        tmp = vh @ (f @ y_)
        for _ in range(11):
            # should converge after at most 10 steps
            # typically converge after two or three steps
            gamma = (s / (s + lam)).sum()
            # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
            # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
            m = v @ (tmp * beta / (alpha + beta * s))
            alpha_de = (m * m).sum()
            alpha = gamma / alpha_de
            beta_de = ((y_ - fh @ m) ** 2).sum()
            beta = (N - gamma) / beta_de
            new_lam = alpha / beta
            if np.abs(new_lam - lam) / lam < 0.01:
                break
            lam = new_lam
        evidence = (
            D / 2.0 * np.log(alpha)
            + N / 2.0 * np.log(beta)
            - 0.5 * np.sum(np.log(alpha + beta * s))
            - beta / 2.0 * beta_de
            - alpha / 2.0 * alpha_de
            - N / 2.0 * np.log(2 * np.pi)
        )
        temp = evidence / N
        return temp
