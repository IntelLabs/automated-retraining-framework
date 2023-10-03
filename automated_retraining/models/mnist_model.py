# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from automated_retraining.models.base_model import BaseClassifier


class MNISTModel(BaseClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
