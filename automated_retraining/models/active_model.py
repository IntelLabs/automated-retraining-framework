# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import automated_retraining.query_strategies as strategies
from automated_retraining.datasets.sets import BaseDataset
from automated_retraining.models.base_model import BaseClassifier


class ActiveModel(BaseClassifier):
    def __init__(self, **kwargs) -> None:
        """Model used for active learning."""
        super().__init__(**kwargs)
        self.pretrain = False

    def set_strategy(self, strategy: str) -> None:
        """Set the strategy used for model calibration checks.

        Args:
            strategy (str): Strategy name from `automated_retraining.query_strategies`
        """
        self.sampling_strategy = getattr(strategies, strategy)

    def query(
        self, dataset: BaseDataset, num_examples: int = 10, weight_classes: bool = None
    ) -> Tuple[List[int], BaseDataset]:
        """Query samples from a dataset.

        Args:
            dataset (BaseDataset): Dataset to query from.
            num_examples (int, optional): Num samples to query. Defaults to 10.
            weight_classes (bool, optional): Weight querying by class distribution. Defaults to None.

        Returns:
            Tuple[List[int], BaseDataset]: List of indices queried and the queried dataset.
        """
        if num_examples > len(dataset):
            return None, None
        else:
            if weight_classes is None:
                query_idx = self.sampling_strategy.execute(
                    learner=self,
                    dataloader=dataset.query_dataloader(),
                    n_instances=num_examples,
                )
                queried = deepcopy(dataset.get_query_dataset())
                queried.subset(query_idx)
                return query_idx, queried
            else:
                num_classes = len(weight_classes)
                max_cnts = [0] * num_classes
                cnts = [0] * num_classes
                for idx, weight in enumerate(weight_classes):
                    max_cnts[idx] = np.ceil(weight * num_examples)
                query_idx = self.sampling_strategy.execute(
                    learner=self,
                    dataloader=dataset.query_dataloader(),
                    n_instances=num_examples,
                )
                queried = deepcopy(dataset.get_query_dataset())
                fin_idx = []
                for idx in query_idx:
                    sample, label = queried[idx]
                    if cnts[label] < max_cnts[label]:
                        fin_idx.append(idx)
                        cnts[label] += 1
                    if sum(cnts) >= num_examples:
                        break
                queried.subset(fin_idx)
                return fin_idx, queried

    def predict_proba(
        self,
        dataloader: DataLoader,
        with_labels: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[int]]]:
        """_summary_

        Args:
            dataloader (DataLoader): Dataloader to use for prediction
            with_labels (bool, optional): Defaults to False.

        Returns:
            Tuple[np.ndarray, Optional[List[int]]]: Softmax output of model, and optionally labels.
        """
        if with_labels:
            labels = []
        y_probas = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = [batch.to(self.model.device) for batch in batch]
                logits = self.forward(batch)
                if self.calibration is not None:
                    logits = self.calibration.calibrate(logits)
                outputs = nn.Softmax(dim=1)(logits)
                for yp in outputs:
                    y_probas.append([yp.cpu().numpy()])
                if with_labels:
                    labels.append(batch[-1].cpu().numpy())
        y_proba = np.concatenate(y_probas, 0)
        if with_labels:
            labels = np.concatenate(labels, 0)
            return y_proba, labels
        else:
            return y_proba
