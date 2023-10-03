# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from automated_retraining.models import BaseModel
from automated_retraining.state_estimation.base import StateEstimation

## SOURCES:
## https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61
## https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing
## https://github.com/gpleiss/temperature_scaling


class CalibrationError(StateEstimation):
    supervision_level = "supervised"

    def __init__(self, device: str = "cuda") -> None:
        """
        Base class for all calibration error metrics including expected
        calibration error (ECE) and maximum calibration error (MCE).

        Args:
            device (str, optional): "cuda" to use GPU. "cpu" to use CPU. Defaults to "cuda".
        """
        super().__init__()
        self.device: str = device

        ## N = num samples
        ## M = num classes
        ## preds is an NxM numpy ndarray of softmax output from model
        ## labels_oneh is an NxM numpy ndarray of one-hot labels

    def calculate_std(
        self, metric: np.ndarray, n_start: int = -5, n_stop: Optional[int] = None
    ) -> float:
        """
        Calculate the standard deviation of the give metric between specified iterations.

        Args:
            metric (np.ndarray): Array of calibration error metrics for each iteration.
            n_start (int, optional): Starting index for std calculation. May use negative values. Defaults to -5.
            n_stop (Optional[int], optional): Stopping index for std calculation. If None, std calculation is performed up to current iteration. Defaults to None.

        Returns:
            float: Standard deviation value
        """
        if len(metric[n_start:n_stop]) > 0:
            return torch.std(torch.tensor(metric[n_start:n_stop]).float()).item()
        else:
            return 0.0

    def get_logits_and_labels(
        self, model: BaseModel, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect and return the model outputs (prior to softmax) along with the associated ground truth labels.

        Args:
            model (BaseModel): Model under test
            dataloader (DataLoader): PyTorch dataloader containing dataset to be run through model

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model outputs (prior to softmax) and associated ground truth labels
        """
        logits = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = [batch.to(self.device) for batch in batch]
                logits.append(model(batch))
                labels.append(batch[1])

            logits = torch.cat(logits).to(self.device)
            labels = torch.cat(labels).to(self.device)

        return logits, labels

    def update_calibration_metric(
        self, model: BaseModel, dataloader: DataLoader, metric: str, n_samples: int = -1
    ) -> None:
        """
        Calculate and record specified calibration metric.

        Args:
            model (BaseModel): Model under test
            dataloader (DataLoader): Pytorch dataloader containing data for performing calculation
            metric (str): Desired calibration metric
            n_samples (int, optional): Number of samples to use for calculation. Can be negative value. Defaults to -1.
        """
        logits, labels = self.get_logits_and_labels(model, dataloader)
        value = self.get_metric_torch(logits[:n_samples], labels[:n_samples])
        self.state_metrics[metric].append(value.item())

    def update_std_metric(
        self, n_start: int = -5, n_stop: Optional[int] = None
    ) -> None:
        """
        Calculate and record the standard deviation of all metrics for specified iterations.

        Args:
            n_start (int, optional): Starting index for standard deviation calcuations. May be negative. Defaults to -5.
            n_stop (Optional[int], optional): Stopping index for standard deviation calcuations. If None, calcuation is performed up to current iteration. Defaults to None.
        """
        tmp_metrics = defaultdict(list)
        for metric_name in self.state_metrics.keys():
            if metric_name[-4:] == "_std":
                continue
            tmp_metrics[metric_name + "_std"] = self.calculate_std(
                self.state_metrics[metric_name]
            )
        self.state_metrics.update(tmp_metrics)

    def get_metric_torch(
        self, logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 10
    ) -> float:
        """
        Get value of calibration error metric. Must be implemented in inheriting class.

        Args:
            logits (torch.Tensor): Output of model pre-softmax layer.
            labels (torch.Tensor): Ground truth labels associated with logits.
            num_bins (int, optional): Number of bins for calibration error calculation. Defaults to 10.

        Raises:
            NotImplementedError: If not implemented in inheriting class.

        Returns:
            float: Calibration error metric value.
        """
        raise NotImplementedError

    def check_if_retraining_needed(
        self, model: BaseModel, dataloader: DataLoader, std_threshold: float = 1
    ) -> bool:
        """
        Check whether retraining is currently needed based on how many standard deviations
        the specified Calibration Error metric has shifted.

        Args:
            model (BaseModel): Model under test.
            dataloader (DataLoader): PyTorch dataloader containing data for calibration error metric calculation.
            std_threshold (float, optional): The number of std deviations of calibration shift tolerated before retraining is initiated. Defaults to 1.

        Raises:
            NotImplementedError: If not implemented in inheriting class.

        Returns:
            bool: Specifying whether or not retraining is needed.
        """
        raise NotImplementedError


class ExpectedCalibrationError(CalibrationError):
    supervision_level = "supervised"

    def __init__(self, device: str = "cuda"):
        """_summary_

        Args:
            device (str, optional): "cuda" to use GPU. "cpu" to use CPU. Defaults to "cuda".
        """
        super().__init__()
        self.device = device

    def get_metric_torch(
        self, logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 10
    ) -> float:
        """
        Calculate ECE.

        Args:
            logits (torch.Tensor): Output of model pre-softmax layer.
            labels (torch.Tensor): Ground truth labels associated with logits.
            num_bins (int, optional): Number of bins for calibration error calculation. Defaults to 10.

        Returns:
            float: ECE value
        """
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                abs_conf_dif = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * abs_conf_dif

        return ece

    def check_if_retraining_needed(
        self, model: BaseModel, dataloader: DataLoader, std_threshold: float = 1
    ) -> bool:
        """
        Check whether retraining is currently needed based on how many standard deviations
        the ECE has shifted. The function first calculates the current ECE, the standard deviation of the past
        five iterations, and thresholds the standard deviation.

        Args:
            model (BaseModel): Model under test.
            dataloader (DataLoader): PyTorch dataloader containing data for calibration error metric calculation.
            std_threshold (float, optional): The number of std deviations of calibration shift tolerated before retraining is initiated. Defaults to 1.

        Returns:
            bool: Specifying whether or not retraining is needed.
        """
        self.update_calibration_metric(model, dataloader, "ece")
        self.update_std_metric()

        if self.state_metrics["ece_std"] != self.state_metrics["ece_std"]:
            return False
        current_ece = self.state_metrics["ece"][-1]
        average_ece = torch.mean(torch.tensor(self.state_metrics["ece"][-5:]).float())
        print(f"Avg ECE : {average_ece:0.3f}\tCur ECE: {current_ece:0.3f}")
        retraining_needed = (
            current_ece - average_ece
        ) > std_threshold * self.state_metrics["ece_std"]
        return retraining_needed


class MaximumCalibrationError(CalibrationError):
    supervision_level = "supervised"

    def __init__(self, device: str = "cuda"):
        """
        Module for calculating Maximum Calibration Error.

        Source:
            https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf

        Args:
            device (str, optional): "cuda" to use GPU. "cpu" to use CPU. Defaults to "cuda".
        """
        super().__init__()
        self.device = device

    def get_metric_torch(
        self, logits: torch.Tensor, labels: torch.Tensor, num_bins: int = 10
    ) -> float:
        """
        Calculate MCE.

        Args:
            logits (torch.Tensor): Output of model pre-softmax layer.
            labels (torch.Tensor): Ground truth labels associated with logits.
            num_bins (int, optional): Number of bins for calibration error calculation. Defaults to 10.

        Returns:
            float: MCE value
        """
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        mce = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                abs_conf_dif = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                mce = torch.maximum(mce, abs_conf_dif)

        return mce

    def check_if_retraining_needed(
        self, model: BaseModel, dataloader: DataLoader, std_threshold: float = 1
    ) -> bool:
        """
        Check whether retraining is currently needed based on how many standard deviations
        the MCE has shifted. The function first calculates the current MCE, the standard deviation of the past
        five iterations, and thresholds the standard deviation.

        Args:
            model (BaseModel): Model under test.
            dataloader (DataLoader): PyTorch dataloader containing data for calibration error metric calculation.
            std_threshold (float, optional): The number of std deviations of calibration shift tolerated before retraining is initiated. Defaults to 1.

        Returns:
            bool: Specifying whether or not retraining is needed.
        """
        self.update_calibration_metric(model, dataloader, "mce")
        self.update_std_metric()

        if self.state_metrics["mce_std"] != self.state_metrics["mce_std"]:
            return False
        current_mce = self.state_metrics["mce"][-1]
        average_mce = torch.mean(torch.tensor(self.state_metrics["mce"][-5:]).float())
        print(f"Avg MCE : {average_mce:0.3f}\tCur MCE: {current_mce:0.3f}")
        retraining_needed = (
            abs(current_mce - average_mce)
            > std_threshold * self.state_metrics["mce_std"]
        )
        return retraining_needed
