# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

# codings: mypy
# type: ignore
import os
import sys
import unittest
from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, Optional, Union

import torchvision.models as torchmodels
import yaml

import automated_retraining.distribution_shifts as distribution_shifts
import automated_retraining.models as models
import automated_retraining.models.architectures as architectures
import automated_retraining.query_strategies as strategies
import automated_retraining.state_estimation as state_estimators
from automated_retraining.tests.utils import validate_schema
import automated_retraining.model_calibration as model_calibration


class ConfigTester(unittest.TestCase):
    def test_config(self) -> None:
        self.assertTrue(
            test_params.config
            in [
                "training",
                "active_learning_distributed",
                "active_learning_standalone",
            ],
            'Expected "training", "active_learning_standalone", or "active_learning_distributed"',
        )


class ModelConfigTester(unittest.TestCase):
    def test_schema(self) -> None:
        if not test_schema:
            return
        valid_schema = validate_schema(vars(test_params), test_key)
        self.assertTrue(valid_schema == None)

    def test_model_name(self) -> None:
        if not hasattr(test_params, "model_name"):
            return

        config = test_args["config"]
        if config == "training":
            self.assertTrue(
                test_params.model_name != "ActiveModel", "Did not expect ActiveModel"
            )
        elif config == "active_learning":
            self.assertTrue(
                test_params.model_name in ["ActiveModel"], "Expected ActiveModel"
            )
        else:
            pass
        self.assertTrue(
            hasattr(models, test_params.model_name),
            "model_name not defined in models folder",
        )

    def test_architecture_name(self) -> None:
        if not hasattr(test_params, "architecture"):
            return
        self.assertTrue(
            hasattr(architectures, test_params.architecture)
            or hasattr(torchmodels, test_params.architecture),
            "architectue not in utils/architectures folder",
        )

    def test_calibration_name(self) -> None:
        if not hasattr(test_params, "calibration"):
            return
        self.assertTrue(
            hasattr(model_calibration, test_params.calibration),
            "unknown calibration method",
        )


class TrainingParamsTester(unittest.TestCase):
    def test_schema(self) -> None:
        if not test_schema:
            return
        valid_schema = validate_schema(vars(test_params), test_key)
        self.assertTrue(valid_schema == None)

    def test_batch_size(self) -> None:
        if not hasattr(test_params, "batch_size"):
            return
        self.assertTrue(0 < test_params.batch_size, "Expected batch_size > 0")

    def test_weight_decay(self) -> None:
        if not hasattr(test_params, "weight_decay"):
            return
        self.assertTrue(0 < test_params.weight_decay, "Expected weight_decay > 0")

    def test_learning_rate(self) -> None:
        if not hasattr(test_params, "lr"):
            return
        self.assertTrue(0 < test_params.lr, "Expected lr > 0")

    def test_gamma(self) -> None:
        if not hasattr(test_params, "gamma"):
            return
        self.assertTrue(0 < test_params.gamma, "Expected gamma > 0")

    def test_momentum(self) -> None:
        if not hasattr(test_params, "momentum"):
            return
        self.assertTrue(0 <= test_params.momentum, "Expected momentum > 0")


class TrainingConfigTester(unittest.TestCase):
    def test_schema(self) -> None:
        if not test_schema:
            return
        valid_schema = validate_schema(vars(test_params), test_key)
        self.assertTrue(valid_schema == None)

    def test_with_validation(self) -> None:
        if not hasattr(test_params, "with_validation"):
            return
        config = test_args["config"]
        if config == "active_learning":
            self.assertTrue(
                test_params.with_validation == False,
                "Expected with_validation to be false when active learning",
            )

    def test_results_dir(self) -> None:
        if not hasattr(test_params, "results_dir"):
            return
        path = test_params.results_dir
        self.assertTrue(
            os.path.isdir(path),
            f"Path {path} does not exist. Please specify full path",
        )

    def test_experiment(self) -> None:
        pass

    def test_max_epochs(self) -> None:
        if not hasattr(test_params, "max_epochs"):
            return
        self.assertTrue(0 < test_params.max_epochs, "Expected max_epochs > 0")

    def test_device(self) -> None:
        if not hasattr(test_params, "device"):
            return
        self.assertTrue(
            test_params.device in ["cuda", "cpu"], 'Expected "cpu" or "cuda"'
        )

    def test_logger(self) -> None:
        if not hasattr(test_params, "logger"):
            return
        self.assertTrue(
            test_params.logger in ["tensorboard", "csvfile"],
            'Expected "tensorboard" or "csvfile"',
        )


class DatasetConfigTester(unittest.TestCase):
    def test_schema(self) -> None:
        if not test_schema:
            return
        valid_schema = validate_schema(vars(test_params), test_key)
        self.assertTrue(valid_schema == None)

    def test_datamodule(self) -> None:
        pass

    def test_dataset_dir(self) -> None:
        if not hasattr(test_params, "dataset_dir"):
            return
        self.assertTrue(
            os.path.isdir(test_params.dataset_dir),
            f"Path {test_params.dataset_dir} does not exits. Please specify full path.",
        )

    def test_n_samples(self) -> None:
        pass


class ActiveParamsTester(unittest.TestCase):
    def test_schema(self) -> None:
        if not test_schema:
            return
        valid_schema = validate_schema(vars(test_params), test_key)
        self.assertTrue(valid_schema == None)

    def test_n_query(self) -> None:
        if not hasattr(test_params, "n_query"):
            return
        self.assertTrue(0 < test_params.n_query, "Expected n_query > 0")

    def test_n_iter(self) -> None:
        if not hasattr(test_params, "n_iter"):
            return

    def test_strategy(self) -> None:
        if not hasattr(test_params, "strategy"):
            return
        self.assertTrue(
            hasattr(strategies, test_params.strategy),
            "strategy not in query_strategies",
        )

    def test_state_estimator(self) -> None:
        if not hasattr(test_params, "state_estimation_method"):
            return
        self.assertTrue(
            hasattr(state_estimators, test_params.state_estimation_method),
            "state estimation method not in state_estimation",
        )

    def test_host(self) -> None:
        if not hasattr(test_params, "state_estimation_host"):
            return
        self.assertTrue(test_params.state_estimation_host in ["edge", "datacenter"])
        state_estimator = getattr(state_estimators, test_params.state_estimation_method)
        if test_params.state_estimation_host == "edge":
            self.assertTrue(state_estimator.supervision_level == "un-supervised")

    def test_chkpt_dir(self) -> None:
        if not hasattr(test_params, "chkpt_dir"):
            return
        path = test_params.chkpt_dir
        self.assertTrue(
            os.path.isdir(path),
            f"Path {path} does not exist. Please specify full path.",
        )


class SimulatorConfigTester(unittest.TestCase):
    def test_schema(self) -> None:
        if not test_schema:
            return
        valid_schema = validate_schema(vars(test_params), test_key)
        self.assertTrue(valid_schema == None)

    def test_n_samples(self) -> None:
        self.assertTrue(0 < test_params.n_samples, "Expected n_samples > 0")

    def test_in_distribution_file(self) -> None:
        path = test_params.in_distribution_file
        self.assertTrue(
            os.path.exists(path),
            f"File {path} does not exist. Please specify full path.",
        )

    def test_out_distribution_file(self) -> None:
        path = test_params.out_distribution_file
        self.assertTrue(
            os.path.exists(path),
            f"File {path} does not exist. Please specify full path.",
        )

    def test_method(self) -> None:
        self.assertTrue(
            hasattr(distribution_shifts, test_params.distribution_shift),
            "method not in distribution_shifts",
        )

    def test_attributes(self) -> None:
        if test_params.distribution_shift == "HardSwitch":
            self.assertTrue("n_shift" in vars(test_params).keys())

        if test_params.distribution_shift == "UniformSwitch":
            self.assertTrue("n_shift" in vars(test_params).keys())

        if test_params.distribution_shift == "Static":
            self.assertTrue("distribution" in vars(test_params).keys())


def main(config: Optional[Union[str, Dict]]):
    param_map = {
        "config": ConfigTester,
        "model_config": ModelConfigTester,
        "training_params": TrainingParamsTester,
        "training_config": TrainingConfigTester,
        "dataset_config": DatasetConfigTester,
        "active_params": ActiveParamsTester,
        "simulator_config": SimulatorConfigTester,
    }

    global test_params, test_args, test_key, test_schema

    if isinstance(config, str):
        args_dict = yaml.safe_load(open(config))
    else:
        args_dict = deepcopy(config)

    args_dict["config"] = {"config": args_dict["config"]}
    test_args = args_dict
    tests = []
    for test_key in args_dict.keys():
        test_schema = True
        test_params = SimpleNamespace(**args_dict[test_key])
        print("\n")
        print(test_params)
        if test_key in ["edge_config", "datacenter_config"]:
            test_schema = False
            for sub_test_key in args_dict[test_key].keys():
                test_params = SimpleNamespace(**args_dict[test_key][sub_test_key])
                print("\n")
                print(test_params)
                suite = unittest.makeSuite(param_map[sub_test_key])
                tests.append(unittest.TextTestRunner(verbosity=1).run(suite))
        else:
            suite = unittest.makeSuite(param_map[test_key])
            tests.append(unittest.TextTestRunner(verbosity=1).run(suite))

    for test in tests:
        if not test.wasSuccessful():
            print(
                f"Errors in config_file. Please run config file tester: tests/test_configs.py"
            )
            sys.exit()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="automated_retraining/tests/debug_configs/debug_mnist_al_distributed.yaml",
    )
    args = parser.parse_args()
    main(args.config_file)
