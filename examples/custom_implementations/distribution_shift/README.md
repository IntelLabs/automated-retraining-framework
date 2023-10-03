# Custom Implementation Example - Distribution Shift

The following example will walk through how to implement a custom distribution shift in the simulator: a random hard switch. More specifically, this example will walk through the required steps of:

- Creating a new distribution shift class in `automated_retraining/distribution_shifts/`
- Implementing the `shift` function
- Using the new distribution shift

## Creating a New Distribution Shift

New distribution shifts are expected to inherit from the `DistributionShift` class in `automated_retraining/distribution_shifts/base.py`.
To set up the new distribution shift, a random hard switch, first create a new python file, `automated_retraining/distribution_shifts/random_hard_switch.py` and import `DistributionShift` and `numpy`.
Create a new class `RandomHardSwitch` which inherits from `DistributionShift`.
The `kwargs` are any parameters specified in the config file.


```python
import numpy as np

from automated_retraining.distribution_shifts.base import DistributionShift


class RandomHardSwitch(DistributionShift):
    def __init__(self, **kwargs):
        """
        This class implements a hard switch between in/out distribution
        data at random intervals.
        """
        super().__init__(**kwargs)
```

Also, add the following line to `automated_retraining/distribution_shifts/__init__.py` to allow the new distribution shift to be accessed in the other sub-modules.

```python
from automated_retraining.distribution_shifts.random_hard_switch import RandomHardSwitch
```

## Implementing the `shift` Function

The only function that must be implemented in any distribution shift class is the `shift` function which should return a floating point value denoting the fraction of in-distribution data to be seen during the current iteration.
Because we are implementing a hard switch between all in-distribution data and all out-of-distribution data, our `shift` function will return either `0.0` or `1.0`, selected randomly using numpy's `random.choice` function.

```python
import numpy as np

from automated_retraining.distribution_shifts.base import DistributionShift


class RandomHardSwitch(DistributionShift):
    def __init__(self, **kwargs):
        """
        This class implements a hard switch between in/out distribution
        data at random intervals.
        """
        super().__init__(**kwargs)

    def shift(self) -> float:
        """Identifies whether data should come from in/out distribution.

        Returns:
        float: Float values indicating percentage of data that should come from in-distribution file. (In this setting returns either 1.0 or 0.0.)
        """
        return np.random.choice([0.0, 1.0])
```

## Using the New Distribution Shift

Finally, to use the new `RandomHardSwitch` distribution shift during active learning, modify the `simulator_config` portion of the active config file (`configs/active_distributed_config.yaml`) as follows, swapping the `distribution_shift` field for the new custom distribution shift:

```yaml
simulator_config:
  n_samples: 100
  in_distribution_file: "in_distribution.txt"
  out_distribution_file: "out_distribution.txt"
  distribution_shift: RandomHardSwitch
```

If needed, modify the `results` and `dataset_dirs` to existing paths, and make any additional changes to the configuration file, if desired. 

To run the new custom `HardDistributionShift` during active learning, simply run:

```python 
python main.py --config ./configs/active_distributed_config.yaml
```