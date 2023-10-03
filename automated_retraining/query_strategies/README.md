# Query Strategies

The directory contains query strategies used during active learning.

## Currently Implemented Query Strategies

This module currently contains the following query strategies:

- Uniform Sampling
- Uncertainty Sampling
- Entropy Sampling
- Margin Sampling

## Creating a Query Strategy

New query strategies should inherit from the `BaseQuery` class in `./automated_retraining/query_strategies/base.py`, and need to implement the `execute` function. The `execute` function should return the indices of queried instances.

## Enable the Query Strategy in the Trainer Module

In order to make the custom model selection method accessible in the `Trainer` module edit the file `./automated_retraining/query_strategies/__init__.py` and add an import statement, for example: 
```python 
 from .base import *
```

To use the model selection method during active learning, edit the config file (`./configs/active_config.yaml`). The `active_params` dictionary in the config (shown below) should be modified to specify the desired `strategy`. 

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
