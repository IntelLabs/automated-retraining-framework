# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import glob
import os
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

import yaml

import automated_retraining.datasets as datasets


class DatasetTester(unittest.TestCase):
    def test_create_learn_set(self):
        datamodule.create_learn_set()

    def test_create_train_set(self):
        datamodule.create_training_set()

    def test_create_query_set(self):
        datamodule.create_query_set()

    def test_train_dataloader(self):
        datamodule.train_dataloader()

    def test_val_dataloader(self):
        datamodule.val_dataloader()

    def test_test_dataloader(self):
        datamodule.test_dataloader()

    def test_query_dataloader(self):
        datamodule.query_dataloader()

    def test_learn_dataloader(self):
        datamodule.learn_dataloader()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config-dir", type=str, default="./configs/")
    args = parser.parse_args()

    configs = glob.glob(os.path.join(args.config_dir, "*.yaml"))
    print("configs: {}\n".format(configs))
    dataset_configs = {}
    for config in configs:
        args_dict = yaml.safe_load(open(config))
        dataset_config = SimpleNamespace(**args_dict["dataset_config"])
        if repr(dataset_config) not in dataset_configs.keys():
            dataset_configs[repr(dataset_config)] = dataset_config

    for dataset_config in dataset_configs.values():
        print("Running tests for DataModule: {}\n".format(dataset_config.datamodule))
        datamodule = getattr(datasets, dataset_config.datamodule)
        datamodule = datamodule(**vars(dataset_config))
        suite = unittest.makeSuite(DatasetTester)
        unittest.TextTestRunner(verbosity=3).run(suite)
