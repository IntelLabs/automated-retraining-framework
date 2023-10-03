# Custom Implementation Example - Model Selection

The following example will walk through how to implement a custom model selection method that will select a source model uniformly at random. 
More specifically, this example will walk through:

- Creating a new model selection class in `automated_retraining/model_selection`
- Implementing the model selection method two different ways
- Using the new model selection method

## Creating a New Model Selection Class

New model selection classes are expected to inerit from the base `ModelSelection` class in `efficient_trianing/model_selection/base.py`. 
To set up the new random model selection method, first create a new python file, `automated_retraining/model_selection/random.py` and import `ModelSelection`, `BaseClassifier`, and numpy. 
Create a new class `RandomModelSelection` which inherits from `ModelSelection`.
The `kwargs` are any parameters specified in the config file.

```python
from typing import List

import numpy as np

from automated_retraining.model_selection.base import ModelSelection
from automated_retraining.models.base_model import BaseClassifier


class RandomModelSelection(ModelSelection):
    def __init__(self, **kwargs) -> None:
        self._name: str = "RandomModelSelection"
        super().__init__(**kwargs)
```

Next, add the following line to `automated_retraining/model_selection/__init__.py` to allow the new model selection method to be accessed in the `Trainer` module:

```python
from automated_retraining.model_selection.random import RandomModelSelection
```

## Implementing the Model Selection Method: Two Ways

When using any model selection method, the `Trainer` module calls the `run_model_selection` function in the `ModelSelection` class.
The default implementation of `run_model_selection` computes a model selection metric for each source model available, calling the `compute_metric` function, and selects a model index using a selection function (`selection_func`) such as numpy's `argmax` or `argmin` (defined as an attribute of the `ModelSelection` class).

When implementing a custom model selection method, either: 
1. Overwrite the default `run_model_selection` function, or
2. Implement the `compute_metric` function and add a `selection_func` attribute to the `__init__` function.

While the former is preferred and results in a cleaner, more efficient implementation, both options are described for completeness.

### Option 1

To implement a random model selection method by overwriting the default `run_model_selection` function, we can simply use numpy to generate a random index into the list of models provided by adding the following function to the new `RandomModelSelection` class.

```python
def run_model_selection(self, models: List) -> int:
        """
        Select the best model from a list according the specified criteria.

        Args:
            models (List): A list of PyTorch models to be scored/selected from.

        Returns:
            int: The index into the models list specifying the best model.
        """
        select_model_idx = np.random.randint(len(models))
        return select_model_idx
```

### Option 2

Alternatively, we can implement the following `compute_metric` function, 

```python
def compute_metric(self, model: BaseClassifier) -> float:
        """
        A dummy compute_metric function that computes a random float between [0,1).

        Args:
            model: The PyTorch model being evaluated

        Returns:
            float: a random float
	"""
        return np.random.random()
``` 

Then, add the following line to the `__init__` function.

```python
self.selection_func = np.argmax
```

## Using the New Model Selection Method

Finally, to use the new `RandomModelSelection` method during active learning, modify the `active_params` portion of the active config file (`configs/active_distributed_config.yaml`) as follows, swapping the `model_selection_method` field for the new custom query strategy:

```yaml
active_params:
    n_query: 20
    n_iter: 10
    strategy: EntropySampling
    model_selection_method: RandomModelSelection
    state_estimation_method: ExpectedCalibrationError
    state_estimation_host: datacenter
    chkpt_dir: "./checkpoints/al_checkpoints/"
    starting_chkpt: "./checkpoints/al_checkpoints/resnet110_chkpt200.pt"
```

If needed, modify the `results` and `dataset_dirs` to existing paths, and make any additional changes to the configuration file, if desired. 

To run the new custon `RandomModelSelection` method during active learning, simply run:

```python 
python main.py --config ./configs/active_distributed_config.yaml
```