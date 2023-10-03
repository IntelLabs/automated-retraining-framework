# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
import sys
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from automated_retraining.datasets.dataset import BaseDataModule, BaseDataset


class CIFAR10DataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        n_samples: int = 0,
        dataset_name: str = "CIFAR10Dataset",
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
            transforms.ToTensor(),
        ]
        self.train_transform = transforms.Compose(transform_list)
        self.val_transform = transforms.Compose(transform_list[-2:])

    def get_dataset(
        self,
        dataset_dir: str,
        transform: transforms,
        n_samples: int,
        train_data=True,
    ) -> BaseDataset:
        """Helper method to get a dataset and subsample if necessary.
        Args:
            dataset_dir (str): Directory where data is stored
            transform (transforms): Data transforms to apply
            n_samples (int): Number of data points to select
        Returns:
            BaseDataset: The dataset
        """
        dataset = self.dataset(dataset_dir, transform, train_data=train_data)
        if n_samples != 0 and len(dataset) > 0:
            inds = np.random.choice(np.arange(len(dataset)), n_samples, replace=False)
            dataset.subset(inds)
        elif n_samples == 0 and len(dataset) > 0:
            print("Using full dataset")
        else:
            print(
                f"Warning: dataset contains no samples and a subset has not been selected"
            )
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


class CIFAR10Dataset(BaseDataset, CIFAR10):
    def __init__(
        self, dataset_dir: str, transform, train_data=True, label: str = "target"
    ) -> None:
        """Set up the CIFAR10 dataset

        Args:
            dataset_dir (str): Directory where data is located.
            transform (_type_): Transforms to apply to data.
            train_data (bool, optional): Flag indicating whether to load train or test data. Defaults to True.
            label (str): general type of the classes
        """
        super(CIFAR10, self).__init__(dataset_dir)
        if dataset_dir != "":
            CIFAR10.__init__(self, self.root, download=True, train=train_data)
        self.train = train_data
        self.dataset_dir: str = dataset_dir
        self.label = label
        self.transform = transform
        self.info_df_headers = ["index", self.label, "for_training"]
        self.info_df_dtypes = {"index": int, self.label: int, "for_training": bool}
        self._info_df: pd.DataFrame = pd.DataFrame(columns=self.info_df_headers)
        self.load_data(from_init=True)

    def load_data(
        self,
        loadfile: str = None,
        loadfiles: pd.DataFrame = None,
        from_init: bool = False,
    ) -> None:
        """Load data, on init or from a particular file. Useful when assigning new data
        to the dataset.
        Args:
            loadfile (str, optional): File to load from. Defaults to None.
            loadfiles (pd.DataFrame, optional): Dataframe of data filenames. Defaults to none.
            from_init (bool, optional): Load on initialization. Defaults to False.
        """
        try:
            if loadfiles is None:
                self.info_df = pd.DataFrame(columns=self.info_df_headers)
                try:
                    self._load_meta()
                except RuntimeError:
                    for self.train in [False, True]:
                        self.download()
                    self._load_meta()
                info_df_data = [
                    np.arange(len(self.targets)),
                    self.targets,
                    [self.train] * len(self.targets),
                ]
                self.info_df = self.concat_datasets(
                    pd.DataFrame(dict(zip(self.info_df_headers, info_df_data)))
                )
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

    def set_data(self, new_info_df: Union[None, pd.DataFrame]):
        """Assign a dataframe to the `info_df` attribute.
        Args:
            new_info_df (pd.DataFrame): New dataframe.
        """
        if new_info_df is None:
            self.info_df = self._info_df
        else:
            self.info_df = new_info_df
        self.info_df.columns = self.info_df_headers

    def __getitem__(self, index: int, *args, **kwargs) -> Tuple[Any, Any]:
        """The full dataset will be stored in `self.data`, so we can grab samples from
        here based on the index, making sure the target at the index matches up with the
        target from the full dataset.

        Args:
            index (int): Index to grab a sample from

        Returns:
            Tuple[Any, Any]: image, target
        """
        target = self.info_df["target"].iloc[index]
        index = self.info_df["index"].iloc[index]
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.info_df)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "raw")
