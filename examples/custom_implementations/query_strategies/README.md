# Custom Implementation Example - Query Strategies

The following example will walk through how to implement a custom query strategy, FIFO Sampling, which will return the first `n` samples. More specifically, this example will walk through the required steps of:

- Creating a new query strategy class in `automated_retraining/query_strategies/`
- Implementing the `execute` function
- Using the new query strategy



## Creating a New Query Strategy Class

New query strategies are expected inherit from the `BaseQuery` class in `automated_retraining/query_strategies/base.py`.
To set up the new FIFO Sampling strategy, first create a new python file, `automated_retraining/query_strategies/fifo_sampling.py` and import `BaseQuery` and `BaseClassifier`, as well as `numpy` and PyTorch's `DataLoader`.
Create a new class `FIFOSampling` which inherits from `BaseQuery`.
Note that there is no `__init__` function required for query strategy classes.

```python
import numpy as np  # type: ignore
from torch.utils.data import DataLoader

from automated_retraining.models.base_model import BaseClassifier

from .base import BaseQuery

class FIFOSampling(BaseQuery):
	pass

```

Next, add the following line to `automated_retraining/query_strategies/__init__.py` to allow the new query strategy to be accessed in the `Trainer` module:
 ```python 
 from .fifo_sampling import *
```

## Implementing the `execute` Function

The only function that must be implemented in any query strategy class is the `execute` function which should return the indicies of the queried samples.
For consistency, all `execute` functions take in a model, a PyTorch DataLoader, and the number of instances to query from the dataloader (`n_instances`).
However, the model will not be used in logic of this function for our FIFO sampling example.

First, to prevent errors, assert that `n_instances` does not exceed the length of the DataLoader.
Then, identify the indicies of the first `n_instances` samples from the DataLoader which will simply be 1 through `n_instances`.
Finally, we'll return the indicies of the first `n_instances` samples.
The `FIFOSampling` class should now be as follows:

```python
class FIFOSampling(BaseQuery):
	def execute(
		learner: BaseClassifier, dataloader: DataLoader, n_instances: int = 1
	) -> np.ndarray:
		"""
		Query the first n_instances. Note that this strategy
		does not make use of the learner.

		Args:
		learner (PyTorch model): The PyTorch model currently active learning.
		dataloader (PyTorch DataLoader): A dataloader containing the pool of datapoints from which to query.
		n_instances (int): The number of intances to query from X. Defaults to 1.

		Returns:
		np.ndarray: Indices into X corresponding to the queried samples
		"""
		assert n_instances <= len(dataloader.dataset)
		query_idx = np.arange(1, n_instances + 1, dtype=int)
		return query_idx

```

## Using the New Query Strategy

Finally, to use the new `FIFOSampling` strategy during active learning, modify the `active_params` portion of the active config file (`configs/active_distributed_config.yaml`) as follows, swapping the `strategy` field for the new custom query strategy:

```yaml
active_params:
  n_query: 20
  n_iter: 10
  strategy: FIFOSampling
  model_selection_method: LEEP
  state_estimation_method: ExpectedCalibrationError
  state_estimation_host: datacenter
  chkpt_dir: "./checkpoints/al_checkpoints/"
  starting_chkpt: "./checkpoints/al_checkpoints/resnet110_chkpt200.pt"
```

If needed, modify the `results` and `dataset_dirs` to existing paths, and make any additional changes to the configuration file, if desired. 

To run the new custom `FIFOSampling` strategy during active learning, simply run:

```python 
python main.py --config ./configs/active_distributed_config.yaml
```