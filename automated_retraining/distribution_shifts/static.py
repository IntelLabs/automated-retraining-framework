# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from automated_retraining.distribution_shifts.base import DistributionShift


class Static(DistributionShift):
    def __init__(self, **kwargs):
        """
        This class creates an even distribution split between in/out distribution
        for the entire active learning session, dictated in simulator_config.
        """
        self.distribution = kwargs["distribution"]
        assert sum(self.distribution) == 1.0
        super().__init__(**kwargs)

    def shift(self):
        """Returns a constant in/out distribution split.

        Returns:
            float: Float values indicating percentage of data that should come from
            in-distribution file.
        """
        return self.distribution[0]
