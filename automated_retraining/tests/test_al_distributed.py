# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import subprocess

dc = subprocess.Popen(
    [
        "python",
        "main.py",
        "--config",
        "./automated_retraining/tests/debug_configs/debug_mnist_al_distributed.yaml",
        "--mode",
        "datacenter",
    ],
    # capture_output=True,
)
edge = subprocess.Popen(
    [
        "python",
        "main.py",
        "--config",
        "./automated_retraining/tests/debug_configs/debug_mnist_al_distributed.yaml",
        "--mode",
        "edge",
    ],
    # capture_output=True,
)

while edge.poll() is None:
    continue

dc.kill()
print("All processes finished successfully")
