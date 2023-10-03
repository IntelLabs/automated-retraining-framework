# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
import yaml

import automated_retraining.models as models
import automated_retraining.state_estimation as state_estimation
from automated_retraining.datasets import BaseDataModule, configure_dataset
from automated_retraining.trainers import configure_training


class SubModule:
    def __init__(self, config: str) -> None:
        """Set up a submodule by parsing a config file.

        Args:
            config (str, optional): Config to parse.
        """
        self.parse_config(config)

    def parse_config(self, config: str) -> None:
        """Extract parameter from config file.

        Args:
            config (str, optional): File to parse.
        """
        args_dict = yaml.safe_load(open(config))

        # Update with edge and datacenter specific configs
        if self.mode == "edge":
            mode_key = "edge_config"
        elif self.mode == "datacenter":
            mode_key = "datacenter_config"
        else:
            mode_key = None
        if mode_key in args_dict.keys():
            module_specific_conf = args_dict[mode_key]
            for conf, conf_dict in module_specific_conf.items():
                for conf_key, conf_val in conf_dict.items():
                    print(f"modifying {conf_key} to {conf_val}")
                    args_dict[conf][conf_key] = conf_val

        for key in args_dict.keys():
            try:
                args_dict[key] = SimpleNamespace(**args_dict[key])
            except TypeError:
                # print('Error with key: %s' % key)
                pass

        args = SimpleNamespace(**args_dict)
        self.model_config: SimpleNamespace = args.model_config
        self.training_params: SimpleNamespace = args.training_params
        self.training_config: SimpleNamespace = args.training_config
        self.dataset_config: SimpleNamespace = args.dataset_config
        if "simulator_config" in vars(args).keys():
            self.simulator_config: SimpleNamespace = args.simulator_config
        if not args.config == "training":
            self.active_params: SimpleNamespace = args.active_params

    def load_model_from_checkpoint(
        self, architecture: str, classes: Dict[str, int], checkpoint: str
    ) -> None:
        """Load a model from a presaved checkpoint

        Args:
            architecture (str): Architecture used when training model
            classes (Dict[str, int]): Class name and number of distinct classes
            checkpoint (str): Path to checkpoint
        """
        self.model.chkpt = checkpoint
        self.model.load_model(architecture, classes, device=self.training_config.device)
        self.model.to(device=self.training_config.device)

    def configure_datasets(self, dataset_keys: List[str]) -> None:
        """Setup datasets and their parameters

        Args:
            dataset_keys (List[str]): Datasets to setup
        """
        key_set_map = {
            "query_dataset": "QuerySet",
            "random_dataset": "QuerySet",
            "learn_dataset": "LearnSet",
        }
        for dataset_key in dataset_keys:
            self.datasets[dataset_key] = configure_dataset(
                self.dataset_config, key_set_map[dataset_key]
            )
        ## LJW NOTE: assuming dataset is labelled
        self.dataset_config.num_classes = self.datasets[dataset_keys[0]].num_classes()

    def configure_training(self) -> None:
        """Setup training params."""
        self.training_config = configure_training(
            self.training_config, self.dataset_config, self.model_config
        )

    def configure_model(self) -> None:
        """Setup model and its params."""
        self.model: models.BaseModel = models.configure_model(
            self.model_config,
            self.dataset_config,
            self.training_config,
            self.training_params,
        )

    def load_state_estimation_utils(
        self, state_estimator: str = "ExpectedCalibrationError"
    ) -> None:
        """Loads the state estimation function specified.

        Args:
            state_estimator (str, optional): State estimation function to used.
            Defaults to "ExpectedCalibrationError".
        """
        state_estimation_module = getattr(state_estimation, state_estimator)
        self.state_estimation = state_estimation_module(
            device=self.training_config.device
        )

    def check_model_state(self):
        random_dataset = self.datasets["random_dataset"]
        retraining_needed = self.state_estimation.check_if_retraining_needed(
            model=self.active_trainer.model,
            dataloader=random_dataset.random_dataloader(),
        )
        return retraining_needed

    def set_sampling_strategy(self, strategy: str) -> None:
        """Set model to use the correct sampling strategy.

        Args:
            strategy (str): Name of strategy to use.
        """
        self.model.set_strategy(strategy)

    def set_dataset_dir(self, dataset_key: str, dir: str) -> None:
        """Set the directory of the dataset specified by dataset_key.

        Args:
            dataset_key (str): Dataset to set directory for.
            dir (str): Directory to set to dataset.
        """
        setattr(
            self.get_dataset(dataset_key),
            "dataset_dir",
            dir,
        )
        print(f"Set dataset directory: {dir}")

    def get_dataset_dir(self, dataset_key: str) -> str:
        """Get the directory of the dataset used.

        Args:
            dataset_key (str): Dataset to get dataset directory for.

        Returns:
            str: Directory of dataset.
        """
        return getattr(self.get_dataset(dataset_key), "dataset_dir")

    def set_info_df(
        self,
        dataset_key: str,
        loadfile: str = None,
        loadfiles: pd.DataFrame = None,
        info_df: pd.DataFrame = None,
    ) -> None:
        """Assigns a new `info_df` to the dataset either from a loaded file or from an
        existing dataframe.

        Args:
            dataset_key (str): Dataset to update.
            loadfile (str, optional): File to load data from. Defaults to None.
            loadfiles (pd.DataFrame, optional): DF containing all filenames to load. Defaults to None.
            info_df (pd.DataFrame, optional): Existing dataframe. Defaults to None.

        Raises:
            ValueError: Must provide either loadfile or info_df
        """
        dataset = self.get_dataset(dataset_key)
        if loadfile is not None and loadfiles is None and info_df is None:
            # load a single file from loadfile
            dataset.load_data(loadfile=loadfile)
            location = loadfile
        elif loadfile is None and loadfiles is not None and info_df is None:
            # load multiple files from loadfiles
            dataset.load_data(loadfiles=loadfiles)
            location = "loadfiles"
        elif loadfile is None and loadfiles is None and info_df is not None:
            # use passed dataframe from info_df
            info_df.reset_index(drop=True, inplace=True)
            dataset.set_data(info_df)
            location = "reference df"
        else:
            raise ValueError(
                f"Must provide one and only one of the following: loadfile path, loadfiles, or info_df."
            )
        print(f"Set info df from: {location}")

    def get_dataset(self, dataset_key: str) -> BaseDataModule:
        """Get dataset specified by `dataset_key`.

        Args:
            dataset_key (str): Dataset to retrieve.

        Returns:
            BaseDataModule: The dataset
        """
        return getattr(self.datasets[dataset_key].data_module, dataset_key)

    def concat_datasets(
        self, dataset_key: str, new_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Add new data to dataframe existing in the specified dataset.

        Args:
            dataset_key (str): Dataset to add new data to.
            new_df (pd.DataFrame, optional): New data to add. Defaults to None.

        Returns:
            pd.DataFrame: New dataframe.
        """
        if new_df is not None:
            return self.get_dataset(dataset_key).concat_datasets(new_df)
        else:
            return self.get_dataset(dataset_key).concat_datasets(self.full_df)

    def get_info_df(self, dataset_key: str) -> None:
        """Get the `info_df` from a dataset.

        Args:
            dataset_key (str): Dataset to get `info_df` from.
        """
        getattr(self.datasets[dataset_key], "info_df")

    def update_transforms(self, dataset_key: str, transforms: dict) -> None:
        """Update dataset transforms by selecting the dataset to update and passing in
        the class name of the transform with its __init__ method args.

        Args:
            dataset_key (str): Dataset to update transfroms for.
            transforms (dict): Transforms to update along with the __init__ args.
        """
        dataset = self.get_dataset(dataset_key)
        if hasattr(dataset, "transform"):
            for transform_name, init_args in transforms.items():
                for transform in dataset.transform.transforms:
                    if transform.__class__.__name__ == transform_name:
                        transform.__init__(init_args)
                        break
        self.dataset = dataset
