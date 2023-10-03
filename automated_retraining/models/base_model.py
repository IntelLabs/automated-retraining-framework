# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as torchmodels

from automated_retraining.models import architectures


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        architecture: str,
        pretrain: bool = True,
        chkpt: str = "",
        **kwargs,
    ) -> None:
        """Base model which all other models are build from.

        Args:
            model_name (str): Name of model.
            architecture (str): Architecture used.
            pretrain (bool, optional): Pretrain model or not. Defaults to True.
            chkpt (str, optional): Load from checkpoint. Defaults to "".

        Raises:
            AttributeError: If not pretraining, must specify a model checkpoint to use.
        """
        super().__init__()
        self.model_name: str = model_name
        self.architecture: str = architecture
        self.droprate: float = 0.0
        self.pretrain: bool = pretrain
        self.chkpt: str = chkpt
        if self.pretrain == False and self.chkpt == "":
            raise AttributeError(
                "Must specify chkpt in model_config if pretrain is False"
            )
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler._LRScheduler

    def loss(self, outputs: torch.Tensor, batch: List[torch.Tensor]):
        pass

    def forward(self, batch: List[torch.Tensor]):
        pass

    def validate(self, batch: List[torch.Tensor]):
        pass

    def metrics(self):
        # Additional metrics to evaluate model, ECE, etc.
        pass

    def compute_metrics(self, logits: torch.Tensor, batch: List[torch.Tensor]):
        raise NotImplementedError

    def get_logits_and_labels(self, dataloader, device="cuda"):
        self.model.eval()
        logits = []
        labels = []
        with torch.no_grad():
            for data, target in dataloader:
                outputs = self.model(data.to(device))
                logits.extend(outputs)
                labels.extend(target.to(device))
        logits = torch.stack(logits)
        labels = torch.stack(labels)
        return logits, labels

    def load_model(
        self, architecture: str, num_classes: Dict[str, int], device: str = "cuda"
    ) -> None:
        """Load model from pre trained checkpoint.

        Args:
            architecture (str): Architecture used to train the model.
            num_classes (Dict[str, int]): Class category and the number of classes
            device (str, optional): Defaults to "cuda".
        """
        model_kwargs = {"num_classes": next(iter(num_classes.values()))}
        model_func = getattr(torchmodels, architecture, None)
        if model_func is None:
            model_func = getattr(architectures, architecture)
        model = model_func(**model_kwargs)
        model.device = device
        # Load checkpoint if provided
        if not self.pretrain and self.chkpt != "":
            model_path = self.chkpt
            assert os.path.exists(model_path) and os.path.isfile(
                model_path
            ), f"Path to model does not exist: {model_path}"
            print("Loading weights from: ", model_path)
            state_dict = torch.load(model_path, map_location=torch.device(device))
            # strict=True, state_dict must match model spec
            # strict=False, any difference in state_dict will defer to model spec - use for transfer learning
            model.load_state_dict(state_dict, strict=False)
            model.using_model_chkpt = model_path
        assert not isinstance(list(model.children())[-1], nn.Softmax)
        self.model = model

    def new_output_layer(self, new_num_classes: int) -> None:
        """Output layer can be modified to accommodate a new number of classes.

        Args:
            new_num_classes (int): New number of classes.
        """
        last_name, last_module = list(self.model.named_modules())[-1]
        if isinstance(last_module, torch.nn.modules.linear.Linear):
            in_features = last_module.in_features
            last_module = getattr(torch.nn.modules, type(last_module).__name__)
            setattr(self.model, last_name, last_module(in_features, new_num_classes))
        else:
            raise NotImplementedError(
                f"Cannot update layer with type {type(last_module)}. Only torch.nn.modules.linear.Linear supported."
            )


class BaseClassifier(BaseModel):
    def __init__(self, **kwargs):
        """Base classifier using accuracy metric and cross entropy loss."""
        super().__init__(**kwargs)
        self.train_acc = torchmetrics.Accuracy()
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy()
        self.val_loss = torch.nn.CrossEntropyLoss()

    def loss(self, outputs: torch.Tensor, batch: List[torch.Tensor]) -> torch.Tensor:
        """Compute loss with cross entropy.

        Args:
            outputs (torch.Tensor): Model outputs.
            batch (List[torch.Tensor]): Batch of data.

        Returns:
            torch.Tensor: Loss.
        """
        _, y = batch
        return self.train_loss(outputs, y)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of model.

        Args:
            batch (List[torch.Tensor]): Batch of data.

        Returns:
            torch.Tensor: Model outputs.
        """
        x, _ = batch
        logits = self.model(x)
        return logits

    def validate(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform validation.

        Args:
            batch (List[torch.Tensor]): Batch of data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits and targets.
        """
        _, targets = batch
        logits = self.forward(batch)
        return logits, targets

    def metrics(self):
        """Get metrics used to compute accuracy and loss.

        Returns:
            torchmetrics.Accuracy, torch.nn.CrossEntropyLoss: Accuracy and loss.
        """
        if self.training:
            return [self.train_acc]
        else:
            return [self.val_acc, self.val_loss]

    def compute_metrics(self, logits: torch.Tensor, batch: List[torch.Tensor]) -> List:
        """Computes accuracy and loss metrics.

        Args:
            logits (torch.Tensor): Logits.
            batch (List[torch.Tensor]): Batch of data.

        Returns:
            List: Calculated metrics.
        """
        _, labels = batch
        output: List = []
        for metric in self.metrics():
            if isinstance(metric, torchmetrics.classification.accuracy.Accuracy):
                if self.calibration is not None and not self.training:
                    calibrated_logits = self.calibration.calibrate(logits)
                    out = metric(F.softmax(calibrated_logits.detach(), dim=1), labels)
                else:
                    out = metric(F.softmax(logits.detach(), dim=1), labels)
                out = (metric.compute().cpu() * 100).item()
            elif isinstance(metric, torch.nn.modules.loss.CrossEntropyLoss):
                out = metric(logits, labels)
            output.append(out)
        return output
