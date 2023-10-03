# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
from typing import Dict

import yaml
from jsonschema import validate


def join_path(base: str, *dir):
    return os.path.join(base, *dir)


def is_file_type_in_directory(base: str, dir: str, extension=""):
    dir = join_path(base, dir)
    files = os.listdir(dir)
    return any(file.endswith(extension) for file in files)


def validate_schema(instance: Dict, instance_type: str) -> bool:
    base = "automated_retraining/tests/schema_files"
    instance_map = {
        "model_config": "model_schema.yaml",
        "training_params": "training_params_schema.yaml",
        "training_config": "training_config_schema.yaml",
        "dataset_config": "dataset_schema.yaml",
        "active_params": "active_standalone_schema.yaml",
        "simulator_config": "simulator_schema.yaml",
    }
    instance = {instance_type: instance}
    file = os.path.join(base, instance_map[instance_type])
    schema = yaml.safe_load(open(file))
    validate(instance=instance, schema=schema)
