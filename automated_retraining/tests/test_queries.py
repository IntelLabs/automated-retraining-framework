# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

import yaml

import automated_retraining.datasets as datasets
import automated_retraining.models as models
import automated_retraining.query_strategies as strategies
from automated_retraining.trainers import configure_training


class QueryTester(unittest.TestCase):
    def test_execute(self):
        strategy.execute(learner, query_set.query_dataloader())


def main(
    args=None, config_file="../configs/base_config.yaml", test_all=True, verbosity=3
):
    if not args:
        parser = ArgumentParser()
        parser.add_argument("--config-file", type=str, default=config_file)
        args = parser.parse_args()
        args = yaml.safe_load(open(args.config_file))

    assert "active_params" in args
    al_config = SimpleNamespace(**args["active_params"])
    assert "training_config" in args
    training_config = SimpleNamespace(**args["training_config"])
    assert "training_params" in args
    training_params = SimpleNamespace(**args["training_params"])
    assert "model_config" in args
    model_config = SimpleNamespace(**args["model_config"])
    assert "dataset_config" in args
    dataset_config = SimpleNamespace(**args["dataset_config"])

    global strategy, learner, query_set

    query_set = datasets.configure_dataset(dataset_config, dataset_type="QuerySet")
    dataset_config.num_classes = query_set.num_classes()
    training_config = configure_training(training_config, dataset_config, model_config)
    learner = models.configure_model(
        model_config, dataset_config, training_config, training_params
    )

    if test_all:
        for s in [
            "EntropySampling",
            "MarginSampling",
            "UncertaintySampling",
            "UniformSampling",
        ]:
            strategy = getattr(strategies, s)
            print("Running tests for Query Strategy: {}\n".format(s))
            suite = unittest.makeSuite(QueryTester)
            unittest.TextTestRunner(verbosity=verbosity).run(suite)
    else:
        strategy = getattr(strategies, al_config["strategy"])
        print("Running tests for Query Strategy: {}\n".format(al_config["strategy"]))
        suite = unittest.makeSuite(QueryTester)
        unittest.TextTestRunner(verbosity=verbosity).run(suite)


if __name__ == "__main__":
    main()
