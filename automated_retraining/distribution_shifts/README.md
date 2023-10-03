# Distribution Shifts

The directory contains classes/functions for creating distribution shifts in the simulator.

## Currently Implemented Distribution Shifts

- `HardSwitch` - Switches back and forth between all in-distribution and all out-distribution every `n_shift` iterations.
- `RandomUniform` - Selects in/out-distribution split uniformly at random every iteration.
- `Static` - Uses a static in/out-distribution split for every iteration.
- `UniformShift` - A gradual uniform shift between in/out-distribution every `n_shift` iterations. For example, letting `n_shift`=4, [1.0, 0.0] -> [0.75, 0.5] -> [0.5, 0.5] -> [0.25, 0.75] -> [0.0, 1.0] -> [0.25, 0.75] -> ...

## Creating a Distribution Shift

New distribution shifts should inherit from the `DistributionShift` class in `./automated_retraining/distribution_shift/base.py`, and need to implement the `shift` function which should return a single floating point value. New methods will also likely need to modify the `__init__` function to include any necessary arguments.

## Enable the Distribution Shift in the Trainer Module

In order to make the custom distribution shift accessible in the `ActiveLearning` submodule edit the file `./automated_retraining/distribution_shift/__init__.py` and add an import statement, for example: 
```python 
 from automated_retraining.distribution_shifts.hard_switch import HardSwitch
```

To use the distribution shift during active learning, edit the config file (`./configs/active_config.yaml`). The `simulator_config` dictionary in the config (shown below) should be modified to specify the desired `distribution_shfit`. 

```yaml 
    simulator_config:
        n_samples: 100
        in_distribution_file: "in_distribution.txt"
        out_distribution_file: "out_distribution.txt"
        distribution_shift: HardSwitch
        n_shift: 5
        distribution:
            - 0.5
            - 0.5
```
