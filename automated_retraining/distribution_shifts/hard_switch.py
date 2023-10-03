# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import numpy as np

from automated_retraining.distribution_shifts.base import DistributionShift


class HardSwitch(DistributionShift):
    def __init__(self, **kwargs):
        """
        This class implements a hard switch between in/out distribution
        data every n_shift iterations (i.e. between a [1.0, 0.0] split
        and [0.0, 1.0] split).
        """
        self.n_shift = kwargs["n_shift"]
        self.counter = 0
        self.percent_in_distribution = 1.0
        super().__init__(**kwargs)

    def shift(self):
        """Identifies whether data should come from in/out distribution
        based on current iteration.

        Returns:
            float: Float values indicating percentage of data that should come from
            in-distribution file. (In this setting returns either 1.0 or 0.0.)
        """
        if self.counter != self.n_shift:
            ## keep with current distribution
            self.counter += 1
        else:
            ## shift distribution
            self.counter = 0
            self.percent_in_distribution = np.abs(self.percent_in_distribution - 1.0)
        return self.percent_in_distribution
