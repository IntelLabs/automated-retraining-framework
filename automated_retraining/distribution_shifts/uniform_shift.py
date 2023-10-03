# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np

from automated_retraining.distribution_shifts.base import DistributionShift


class UniformShift(DistributionShift):
    def __init__(self, **kwargs):
        """
        This class implements a gradual uniform shift between in/out
        distribution data.

        """
        self.n_shift = kwargs["n_shift"]
        self.counter = 0
        self.interval = np.linspace(1.0, 0.0, self.n_shift + 1)
        super().__init__(**kwargs)

    def shift(self):
        """Select in/out distribution split.

        Returns:
            float: Float values indicating percentage of data that should come from
            in-distribution file.
        """
        percent_in_distribution = self.interval[self.counter]
        if self.counter != self.n_shift:
            self.counter += 1
        else:
            self.counter = 1
            self.interval = np.flip(self.interval)
        return percent_in_distribution
