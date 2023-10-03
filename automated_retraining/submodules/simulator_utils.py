# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
import pickle

import pandas as pd

import automated_retraining.distribution_shifts as distribution_shifts
from automated_retraining.submodules.submodule_utils import SubModule


class SimulatorModule(SubModule):
    def __init__(self, config: str, **kwargs) -> None:
        """Simulator module used to facilitate testing different scenarios of data distribution shift.

        Args:
            config (str, optional): Config to use on the simulator module.
            Defaults to "./configs/active_config.yaml".
        """
        self.__dict__.update(kwargs)
        self.parse_config(config)
        self.n_samples: int = self.simulator_config.n_samples
        self.__setup()

    def __setup(self) -> None:
        """Setup the simulator module"""
        try:
            if self.simulator_config.in_distribution_file.split(".")[-1] == "pkl":
                with open(
                    os.path.join(self.simulator_config.in_distribution_file), "rb"
                ) as f:
                    in_data_dict = pickle.load(f)
                self.in_distribution_df = pd.DataFrame.from_dict(in_data_dict)
                with open(
                    os.path.join(self.simulator_config.out_distribution_file), "rb"
                ) as f:
                    out_data_dict = pickle.load(f)
                self.out_distribution_df = pd.DataFrame.from_dict(out_data_dict)
            else:
                self.in_distribution_df = pd.read_csv(
                    os.path.join(self.simulator_config.in_distribution_file),
                    sep=None,
                    engine="python",
                    index_col=0,
                    header=None,
                )
                self.out_distribution_df = pd.read_csv(
                    os.path.join(self.simulator_config.out_distribution_file),
                    sep=None,
                    engine="python",
                    index_col=0,
                    header=None,
                )
        except:
            raise RuntimeError("Unsupported file type.")

        self.in_distribution_df = self.in_distribution_df.reset_index(drop=True)

        self.percent_in_distribution = getattr(
            distribution_shifts, self.simulator_config.distribution_shift
        )(**vars(self.simulator_config))

        try:
            for k, v in self.simulator_config.in_transform.items():
                self.in_distribution_df[k] = len(self.in_distribution_df) * [v]

            for k, v in self.simulator_config.out_transform.items():
                self.out_distribution_df[k] = len(self.out_distribution_df) * [v]
        except AttributeError:
            pass

    def retrieve_data(self, idx: int) -> pd.DataFrame:
        """Retrieve in/out of distribution samples for edge

        Args:
            idx (int): iteration idx

        Returns:
            pd.DataFrame: dataframe of data to be
            sent to the edge
        """

        n_in = int(self.percent_in_distribution.shift() * float(self.n_samples))
        n_out = self.n_samples - n_in

        if n_in > len(self.in_distribution_df.index):
            raise RuntimeError("Out of In-Distribution Samples")
        in_samples = self.in_distribution_df.sample(n=n_in)
        self.in_distribution_df.drop(list(in_samples.index.values), inplace=True)

        if n_out > len(self.out_distribution_df.index):
            raise RuntimeError("Out of Out-of-Distribution Samples")
        out_samples = self.out_distribution_df.sample(n=n_out)
        self.out_distribution_df.drop(list(out_samples.index.values), inplace=True)

        new_samples = pd.concat([in_samples, out_samples], ignore_index=True, axis=0)

        return new_samples
