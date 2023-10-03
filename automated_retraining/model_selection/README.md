# Model Selection

The directory contains model selection methods used during active learning.

## Currently Implemented Model Selection Methods

This module currently contains model selection methods using transferability metrics: Log Expected Empirical Prediction (LEEP) and the Logarithm of Maximum Evidence (LogME). Further information about these metrics can be found in the publications and open-source code:
- LEEP
    - Paper: https://arxiv.org/abs/2002.12462
- LogME
    - Paper: https://arxiv.org/abs/2102.11005
    - Code: https://github.com/thuml/LogME

## Creating a Model Selection Method

New model selection method should inherit from the `Model Selection` class in `./automated_retraining/model_selection/base.py`, and need to implement the `compute_metric` function which should return a single floating point value. New methods will also likely need to modify the `__init__` function to include any necessary arguments.

## Enable the Model Selection Method in the Trainer Module

In order to make the custom model selection method accessible in the `Trainer` module edit the file `./automated_retraining/model_selection/__init__.py` and add an import statement, for example: 
```python 
 from automated_retraining.model_selection.leep import LEEP
```

To use the model selection method during active learning, edit the config file (`./configs/active_config.yaml`). The `active_params` dictionary in the config (shown below) should be modified to specify the desired `model_selection_method`. 

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
