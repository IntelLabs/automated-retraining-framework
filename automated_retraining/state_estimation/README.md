# State Estimation

The directory contains model state estimation methods used during active learning to identify when to initiate retraining.

## Currently Implemented Model State Estimation Methods

This module currently contains the following methods for estimating model calibration:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
When the standard deviation of the specified calibration metric falls outside of a tolerable range, retraining is initiated.

## Creating a Model State Estimation Method

New model state estimation method should inherit from the `StateEstimation` class in `./automated_retraining/state_estimation/base.py`, and need to implement the `check_if_retraining_needed` function which should return a boolean. New methods will also likely need to modify the `__init__` function to include any necessary arguments.

## Enable the Model State Estimation Method in the Trainer Module

In order to make the custom model state estimation method accessible in the `Trainer` module edit the file `./automated_retraining/state_estimation/__init__.py` and add an import statement, for example: 
```python 
from automated_retraining.state_estimation.calibration import (
    ExpectedCalibrationError,
    MaximumCalibrationError,
)
```

To use the model state estimation method during active learning, edit the config file (`./configs/active_config.yaml`). The `active_params` dictionary in the config (shown below) should be modified to specify the desired `state_estimation_method`. 

```yaml 
    active_params:
        n_query: 20
        n_iter: 3
        strategy: EntropySampling
        model_selection_method: LEEP
        state_estimation_method: ExpectedCalibrationError
        state_estimation_host: datacenter
        chkpt_dir: "./checkpoints/al_checkpoints/"
        starting_chkpt: "./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt" 
```
