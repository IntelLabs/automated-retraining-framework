# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from distutils.core import setup

setup(
    name="automated_retraining",
    version="0.1dev",
    packages=[
        "automated_retraining",
        "automated_retraining.utils",
        "automated_retraining.datasets",
        "automated_retraining.models",
        "automated_retraining.models.architectures",
        "automated_retraining.model_selection",
        "automated_retraining.state_estimation",
        "automated_retraining.trainers",
        "automated_retraining.tests",
        "automated_retraining.submodules",
        "automated_retraining.query_strategies",
        "automated_retraining.utils.communication",
        "automated_retraining.distribution_shifts",
        "automated_retraining.datasets.utils",
    ],
    long_description=open("README.md").read(),
)
