# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from collections import defaultdict

from torch.utils.data import DataLoader


class StateEstimation:
    def __init__(self) -> None:
        """
        Base class for all model state estimation methods and metrics
        such as calibration-based metrics (i.e. ECE and MCE).
        """
        self.state_metrics = defaultdict(list)

    def check_if_retraining_needed(self) -> bool:
        """
        Check if retraining is needed using the specified model state
        estimation method or metric.

        Raises:
            NotImplementedError: If not implemented in inheriting class.

        Returns:
            bool: Specifying whether or not retraining is needed.
        """
        raise NotImplementedError

    def reset(self, dataloader: DataLoader = None) -> None:
        pass
