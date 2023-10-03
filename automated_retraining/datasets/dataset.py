# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset


class BaseDataModule:
    def __init__(self, dataset_dir: str, datamodule: str, **kwargs):
        """Base data module which sets the framework for creating the various sets used
        throughout the codebase.

        Args:
            dataset_dir (str): Directory where the dataset is stored.
            datamodule (str): Data module that is being created.
        """
        self.data_split: List[float] = [0.7, 0.1, 0.2]
        self.train_dataset: Union[BaseDataset, Subset]
        self.val_dataset: Union[BaseDataset, Subset]
        self.test_dataset: Union[BaseDataset, Subset]
        self.query_dataset: Union[BaseDataset, Subset]
        self.learn_dataset: Union[BaseDataset, Subset]
        self.num_classes: dict = {}
        self.dataset_dir: str = dataset_dir
        self.datamodule: str = datamodule

    def create_learn_set(self):
        raise NotImplementedError

    def create_query_set(self):
        raise NotImplementedError

    def create_training_set(self):
        """
        for training split dataset into train, val, and test
        and create separate dataloaders for each
        """
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def query_dataloader(self):
        raise NotImplementedError

    def learn_dataloader(self):
        raise NotImplementedError


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def subset(self, indices: Iterable[int]) -> None:
        """Create a subset of data given a set of indices.

        Args:
            indices (Iterable[int]): Indices of data to create subset.
        """
        assert hasattr(self, "info_df")

        non_indices = np.delete(np.arange(len(self.info_df)), indices)
        self.info_df.drop(non_indices, inplace=True)
        self.info_df.reset_index(inplace=True, drop=True)

    def concat_datasets(self, incoming_df: pd.DataFrame) -> pd.DataFrame:
        """Combine two dataframes, typically adding new data to a full set of data seen.

        Args:
            incoming_df (pd.DataFrame): Data frame to append to.

        Returns:
            pd.DataFrame: New data frame.
        """
        if hasattr(self, "info_df"):
            new_df = pd.concat([self.info_df, incoming_df])
            new_df.reset_index(inplace=True, drop=True)
        else:
            self.info_df = incoming_df
            new_df = self.info_df
        return new_df
