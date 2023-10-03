# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from automated_retraining.state_estimation import (
    ExpectedCalibrationError,
    MaximumCalibrationError,
)


class TemperatureScaling:
    def __init__(
        self,
        temperature: Optional[float] = None,
        metric: Optional[Callable] = ExpectedCalibrationError(),
        device: Optional[str] = "cuda",
        **kwargs
    ) -> None:
        """Model calibration via temperature scaling.

        References:
            - https://arxiv.org/pdf/1706.04599.pdf

        Args:
            temperature (float, optional): Pre-specified temperature value. Defaults to None.
            metric (Callable, optional): Calibration error metric used to determine temperature (ECE or MCE). Defaults to ExpectedCalibrationError().
            device (str, optional): "cuda" or "cpu". Defaults to "cuda".
        """
        self.device = device
        self.metric = metric

        if temperature is None:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrate incoming logits using temperature scaling.

        Args:
            logits (torch.Tensor): Output of NN, pre-softmax.

        Returns:
            torch.Tensor: Calibrated/temperature-scaled NN outputs.
        """
        return torch.div(logits, self.temperature.to(self.device))

    def reset(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Identify and reset temperature using new batch of logits and labels.

        Args:
            logits (torch.Tensor): Output of NN, pre-softmax.
            labels (torch.Tensor): Integer labels for each set of logits.
        """
        self.set_temperature(logits, labels)

    def set_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.02,
        max_iter: int = 1000,
        line_search_fn: Callable = None,
    ) -> None:
        """Identify optimal temperature value with respect to the cross entropy loss
        using given logits and labels.


        Args:
            logits (torch.Tensor): Output of NN, pre-softmax
            labels (torch.Tensor): Integer labels for each set of logits.
            lr (float, optional): Learning rate for optimizer. Defaults to 0.02.
            max_iter (int, optional): Max iterations for optimizer. Defaults to 1000.
            line_search_fn (Callable, optional): Optional line search algorithm for . Defaults to None.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter, line_search_fn=line_search_fn
        )

        before_ece = self.metric.get_metric_torch(logits, labels)
        print("Metric before temperature scaling: ", before_ece)

        def eval():
            optimizer.zero_grad()
            loss = criterion(self.calibrate(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        print("Temperature set to ", self.temperature.item())

        scaled_logits = self.calibrate(logits)
        after_ece = self.metric.get_metric_torch(scaled_logits, labels)
        print("Metric after temperature scaling: ", after_ece)
