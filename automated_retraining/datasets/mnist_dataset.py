# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from automated_retraining.datasets.dataset import BaseDataModule, BaseDataset
from automated_retraining.datasets.utils import ColorAugmentation, DistributionCompose


class MNISTDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        n_samples: int = 0,
        dataset_name: str = "MNISTDataset",
        distribution_transforms: Optional[Dict] = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset = getattr(sys.modules[__name__], dataset_name)
        self.batch_size: int = batch_size
        self.n_samples: int = n_samples
        self.dims = (3, 28, 28)
        # Define image transformations
        transform_list = [
            transforms.Resize(size=self.dims[1:]),
            transforms.Grayscale(num_output_channels=3),
            ColorAugmentation(),
        ]
        self.distribution_transforms = distribution_transforms
        if distribution_transforms:
            self.train_transform = DistributionCompose(transform_list)
            self.val_transform = DistributionCompose(transform_list)
        else:
            self.train_transform = transforms.Compose(transform_list)
            self.val_transform = transforms.Compose(transform_list)

    def get_dataset(
        self,
        dataset_dir: str,
        transform: transforms,
        n_samples: int,
        train_data: bool = True,
    ) -> BaseDataset:
        """Helper method to get a dataset and subsample if necessary.

        Args:
            dataset_dir (str): Directory where data is stored
            transform (transforms): Data transforms to apply
            n_samples (int): Number of data points to select
            train_data (bool, optional): Flag indicating if data is for training or validation. Defaults to True.
        Returns:
            BaseDataset: The dataset
        """
        dataset = self.dataset(
            dataset_dir, transform, self.distribution_transforms, train_data=train_data
        )
        if n_samples != 0 and len(dataset) > 0:
            inds = np.random.choice(np.arange(len(dataset)), n_samples, replace=False)
            dataset.subset(inds)
        elif n_samples == 0 and len(dataset) > 0:
            print("Using full dataset")
        else:
            print(
                f"Warning: dataset contains no samples and a subset has not been selected"
            )
        self.num_classes = dataset.num_classes
        return dataset

    def create_training_set(self) -> None:
        """Training set creation."""
        train = True
        self.train_dataset = self.get_dataset(
            self.dataset_dir,
            self.train_transform,
            int(self.n_samples * self.data_split[0]),
            train_data=train,
        )
        train = False
        self.test_dataset = self.get_dataset(
            self.dataset_dir,
            self.train_transform,
            int(self.n_samples * sum(self.data_split[1:])),
            train_data=train,
        )
        self.num_classes = self.train_dataset.num_classes

    def create_query_set(self) -> None:
        """Query set creation."""
        self.query_dataset = self.get_dataset(
            self.dataset_dir,
            self.val_transform,
            int(self.n_samples * self.data_split[0]),
        )
        self.num_classes = self.query_dataset.num_classes

    def create_random_set(self) -> None:
        """Random set creation."""
        self.random_dataset = self.get_dataset(
            self.dataset_dir,
            self.val_transform,
            int(self.n_samples * self.data_split[0]),
        )
        self.num_classes = self.random_dataset.num_classes

    def create_learn_set(self) -> None:
        """Learn set creation."""
        self.learn_dataset = self.get_dataset(
            self.dataset_dir,
            self.val_transform,
            int(self.n_samples * self.data_split[0]),
        )
        self.num_classes = self.learn_dataset.num_classes

    def get_dataloader(self, dataset: BaseDataset) -> DataLoader:
        """Template for getting a standard pytorch dataloader.

        Args:
            dataset (BaseDataset): Dataset to use with dataloader

        Returns:
            DataLoader: The dataloader
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Get the train dataloader.
        Returns:
            DataLoader: Train dataloader
        """
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        """Get the val dataloader.
        Returns:
            DataLoader: Val dataloader.
        """
        print("Using test dataset in val_dataloader")
        return self.get_dataloader(self.test_dataset)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        Returns:
            DataLoader: Test dataloader.
        """
        return self.get_dataloader(self.test_dataset)

    def random_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        Returns:
            DataLoader: Test dataloader.
        """
        return self.get_dataloader(self.random_dataset)

    def learn_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        Returns:
            DataLoader: Test dataloader.
        """
        return self.get_dataloader(self.learn_dataset)

    def query_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        Returns:
            DataLoader: Test dataloader.
        """
        return self.get_dataloader(self.query_dataset)


class MNISTDataset(BaseDataset, MNIST):
    def __init__(
        self,
        dataset_dir: str,
        transform,
        distribution_transforms,
        train_data=True,
        label: str = "digit",
    ) -> None:
        """Set up the MNIST Dataset

        Args:
            dataset_dir (str): Directory where data is located.
            transform (_type_): Transforms to apply to data.
            train_data (bool, optional): Flag indicating whether to load train or test data. Defaults to True.
            label (str): general type of the classes
        """
        super(MNIST, self).__init__(dataset_dir)
        if dataset_dir != "":
            MNIST.__init__(self, self.root, download=True, train=train_data)
        self.train = train_data
        self.dataset_dir: str = dataset_dir
        self.transform = transform
        self.distribution_transforms = distribution_transforms
        self.label = label
        self.info_df_headers = {"index": int, self.label: int, "for_training": bool}
        for k, v in self.distribution_transforms.items():
            self.info_df_headers[k] = type(v)

        self._info_df: pd.DataFrame = pd.DataFrame(columns=self.info_df_headers)
        self.load_data()

    def load_data(
        self,
        loadfiles: pd.DataFrame = None,
    ) -> None:
        """Load data, on init or from a particular file. Useful when assigning new data
        to the dataset.

        Args:
            loadfiles (pd.DataFrame, optional): Dataframe of data filenames. Defaults to none.
        """
        try:
            if loadfiles is None:
                try:
                    self.data, self.targets = self._load_data()
                except FileNotFoundError:
                    for self.train in [False, True]:
                        self.download()
                    self.data, self.targets = self._load_data()

                info_df_data = {
                    "index": np.arange(len(self.targets)),
                    self.label: self.targets,
                    "for_training": [self.train] * len(self.targets),
                }
                if self.distribution_transforms:
                    for k, v in self.distribution_transforms.items():
                        info_df_data[k] = len(self.targets) * [v]

                self.info_df = pd.DataFrame(info_df_data)
            else:
                self.set_data(loadfiles)

        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Must load in info_df manually")

        self.get_num_classes()

    def get_num_classes(self):
        try:
            self.num_classes = {self.label: len(self.info_df[self.label].unique())}
        except Exception:
            print("Setting num classes to default: 10")
            self.num_classes = {self.label: 10}

    def set_data(self, new_info_df: pd.DataFrame):
        """Assign a dataframe to the `info_df` attribute.

        Args:
            new_info_df (pd.DataFrame): New dataframe.
        """
        self.info_df = new_info_df
        self.info_df.columns = self.info_df_headers
        self.num_classes = {self.label: len(self.info_df[self.label].unique())}

    def __getitem__(self, index: int, *args, **kwargs) -> Tuple[Any, Any]:
        _index = index
        target = self.info_df[self.label].iloc[index]
        index = self.info_df["index"].iloc[index]
        img = self.data[index]
        img = Image.fromarray(img.numpy(), mode="L")
        if self.distribution_transforms:
            img = self.transform(
                img, self.distribution_transforms, self.info_df, _index
            )
        else:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.info_df)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "raw")
