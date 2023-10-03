# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from copy import deepcopy
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from automated_retraining.datasets.dataset import BaseDataModule, BaseDataset


class BaseSet:
    def __init__(self, data_module: BaseDataModule):
        """Base dataset which the learn, train, and query sets are built from.

        Args:
            data_module (BaseDataModule): Data module to use in combination with the
            base set.
        """
        self.data_module: BaseDataModule = data_module
        self.with_filenames: bool = False
        self.set_generator()

    def set_generator(self) -> None:
        self.generator = torch.Generator()
        self.generator.manual_seed(0)
        torch.random.manual_seed(0)

    def update_dataset(
        self,
        dataset: BaseDataset,
        new_df: pd.DataFrame,
        only_new: bool = False,
        random_samples: bool = False,
    ) -> None:
        """Add new dataframe to a dataset using new samples, random samples, or all
        samples as well as new samples.

        Args:
            dataset (BaseDataset): Dataset to update
            new_df (pd.DataFrame): New data
            only_new (bool, optional): Use only new data. Defaults to False.
            random_samples (bool, optional): Use random samples. Defaults to False.

        Returns:
            _type_: Returns self
        """
        dataset = getattr(self.data_module, dataset)
        if only_new and not random_samples:
            new_info_df = new_df
        elif random_samples and not only_new:
            new_info_df = dataset.concat_datasets(new_df)
            new_info_df = dataset.subset(
                np.random.choice(
                    np.arange(len(new_info_df)), size=len(new_df), replace=False
                )
            )
        else:
            new_info_df = dataset.concat_datasets(new_df)

        dataset.info_df = new_info_df
        return self

    def create_train_test_val_split(self):
        raise NotImplementedError

    def create_query_set(self):
        raise NotImplementedError

    def create_learn_set(self):
        raise NotImplementedError

    def update_query_set(self):
        raise NotImplementedError

    def update_learn_set(self):
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

    def num_classes(self):
        return self.data_module.num_classes


class LearnSet(BaseSet):
    # On Datacenter for active learning/training
    def __init__(self, data_module: BaseDataModule, **kwargs) -> None:
        """Create learn set on initialization.

        Args:
            data_module (BaseDataModule): Data module used to create learn set.
        """
        super().__init__(data_module, **kwargs)
        self.with_filenames: bool = False
        self.create_learn_set()

    def __len__(self) -> int:
        return len(self.data_module.learn_dataset)

    def create_learn_set(self) -> None:
        self.data_module.create_learn_set()

    def update_learn_set(self):
        pass

    def learn_dataloader(self) -> DataLoader:
        return self.data_module.learn_dataloader()

    def train_dataloader(self) -> DataLoader:
        return self.data_module.train_dataloader()


class QuerySet(BaseSet):
    # On Edge for active learning query
    def __init__(self, data_module: BaseDataModule, **kwargs):
        """Create query set on initialization.

        Args:
            data_module (BaseDataModule): Data module to create query set with.
        """
        super().__init__(data_module, **kwargs)
        self.with_filenames: bool = False
        self.create_query_set()

    def __len__(self) -> int:
        return len(self.data_module.query_dataset)

    def create_query_set(self) -> None:
        """Generate both query and random sets to use for querying."""
        self.data_module.create_query_set()
        self.data_module.create_random_set()

    def update_query_set(self, delete_idx: List[int]) -> None:
        """Delete specific indices in the query dataset.

        Args:
            delete_idx (int): indices to delete
        """
        keep_idx = np.arange(len(self.data_module.query_dataset))
        keep_idx = np.delete(keep_idx, delete_idx)
        self.data_module.query_dataset.subset(keep_idx)

    def get_query_dataset(self) -> BaseSet:
        return self.data_module.query_dataset

    def query_dataloader(self) -> DataLoader:
        return self.data_module.query_dataloader()

    def get_random_dataset(self) -> BaseSet:
        return self.data_module.random_dataset

    def random_dataloader(self) -> DataLoader:
        return self.data_module.random_dataloader()


class TrainSet(BaseSet):
    # On Datacenter for training model from scratch
    def __init__(self, data_module: BaseDataModule, **kwargs):
        """Create query set on initialization.

        Args:
            data_module (BaseDataModule): Data module to create train set with.
        """
        super().__init__(data_module, **kwargs)

        self.create_train_test_val_split()

    def __len__(self) -> int:
        return len(self.data_module.train_dataset)

    def create_train_test_val_split(self) -> None:
        self.data_module.create_training_set()

    def train_dataloader(self) -> DataLoader:
        return self.data_module.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.data_module.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.data_module.test_dataloader()
