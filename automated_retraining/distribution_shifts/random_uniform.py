# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np

from automated_retraining.distribution_shifts.base import DistributionShift


class RandomUniform(DistributionShift):
    def __init__(self, **kwargs):
        """
        This class allows for a uniformly random selected in/out distribution split
        every iteration for the entire active learning session.
        """
        super().__init__(**kwargs)

    def shift(self):
        """Selects in/out distribution split uniformly at random.

        Returns:
            float: Float values indicating percentage of data that should come from
            in-distribution file.
        """
        return np.random.uniform()
