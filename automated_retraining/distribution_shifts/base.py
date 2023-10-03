# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License


class DistributionShift:
    def __init__(self, **kwargs) -> None:
        """
        Base function for creating distribution shifts in the simulator.
        """
        pass

    def shift(self) -> float:
        """
        Identify what percentage of the data should come from the in-distribution
        file based on the current iteration index.

        Raises:
            NotImplementedError: Needs to be implemented in inheriting modules/classes

        Returns:
            float: Float values indicating percentage of data that should come from
            in-distribution file.
        """
        raise NotImplementedError
