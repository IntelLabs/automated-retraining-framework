# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import yaml
from argparse import ArgumentParser
from types import SimpleNamespace

import automated_retraining.tests.test_configs as config_test
from automated_retraining.submodules import (
    ActiveLearning,
    ActiveLearningStandalone,
    TrainingModule,
)


def main(task, config, mode):
    if task == "training":
        trainer = TrainingModule(config=config, mode=mode)
        trainer.run()
    if task == "active_learning_standalone":
        al_standalone = ActiveLearningStandalone(config=config, mode=mode)
        al_standalone.run()
    if task == "active_learning_distributed":
        al = ActiveLearning(config=config, mode=mode)
        al.run()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base_config.yaml")
    parser.add_argument("--mode", type=str, default="demo")
    cli_args = parser.parse_args()  # command line arguments
    args_dict = yaml.safe_load(open(cli_args.config))
    config_tests = config_test.main(args_dict)
    config_file = cli_args.config
    for key in args_dict.keys():
        try:
            args_dict[key] = SimpleNamespace(**args_dict[key])
        except TypeError:
            pass

    args = SimpleNamespace(**args_dict)
    task = args.config
    main(task, config_file, cli_args.mode)
