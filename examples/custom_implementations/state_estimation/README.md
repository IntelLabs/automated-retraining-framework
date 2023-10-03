# Wrapping External Libraries - State Estimation Example

The following example will walk through how to wrap an external libarary within the `automated_retraining` codebase.
More specifically, this example will walk through how to create a new state estimator utilizing Frouros' implementation of [Maximum Mean Discrepancy (MMD)](https://github.com/jaime-cespedes-sisniega/frouros).
The required steps are as follows:

- Install Frouros
- Creating a new state estimation class in `automated_retraining/state_estimation`
- Implementing the `__init__`, `reset`, and `check_if_retraining_needed` functions
- Using the wrapped state estimator

## Install Frouros

To install the Frouros library via `pip`, run the following command: 

```python
pip install frouros
```

## Creating a New State Estimation Class
New state estimators are expected to inherit from the `StateEstimation` class in `automated_retraining/state_estimation/base.py`.
To set up the new wrapped state estimator, first create a new python file,  `automated_retraining/state_estimation/frouros_mmd.py` and import the following:

```python 
import numpy as np
from torch.utils.data import DataLoader
from sklearn.gaussian_process.kernels import RBF
from frouros.unsupervised.distance_based import MMD
from scipy.spatial.distance import pdist

from automated_retraining.models import BaseModel
from automated_retraining.state_estimation.base import StateEstimation
```

Create a new class `FrourosMaximumMeanDiscrepancy` which inherits from `StateEstimation`, and create a `supervision_level` attribute:

```python
class FrourosMaximumMeanDiscrepancy(StateEstimation):
    supervision_level = "un-supervised"
```

Next, add the following line to `automated_retraining/state_estimation/__init__.py` to allow the new state estimator to be accessed in the `Trainer` module:
 ```python 
from automated_retraining.state_estimation.frouros_mmd import (
    FrourosMaximumMeanDiscrepancy,
)
```

## Function Implementations

First, create an `__init__` function that will create an instance of Frouros' MMD algorithm. This `__init__` function should also take in parameters `num_permuations`, `sigma`, `device`, and (optionally) a `random_state`, which are needed to initialize Frouros MMD.

```python
	def __init__(
		self,
		num_permutations: int = 1000,
		sigma: float = 500.0,
		random_state: Optional[int] = None,
		device: str = "cuda",
	) -> None:
		"""
		Class wrapping Frouros' implementation of Maximum Mean Discrepancy (MMD)
		using a Radial Basis Function (RBF) kernel.
		Note: best to use sigma = np.median(pdist(X=np.vstack((X_ref, X_test)), metric="euclidean")) / 2

		Source:
		https://github.com/jaime-cespedes-sisniega/frouros
		"""
		super().__init__()
		self.device: str = device

		if random_state is not None:
		self.detector = MMD(
			num_permutations=num_permutations,
			kernel=RBF(length_scale=sigma),
			random_state=random_state,
		)
		else:
		self.detector = MMD(
			num_permutations=num_permutations, kernel=RBF(length_scale=sigma)
		)
```

Before we implement the `reset` and `check_if_retraining_needed` functions, we'll write a simple helper function to convert dataloaders to 2D numpy arrays, as Frouros requires all inputs to be 2D numpy arrays.

```python
	def __get__data(self, dataloader: DataLoader) -> np.ndarray:
		"""
		Convert dataloader to numpy array.

		Args:
		dataloader (DataLoader): Any PyTorch dataloader

		Returns:
		np.ndarray: Numpy array containing the dataset from the dataloader
		"""
		data = []
		for i, batch in enumerate(dataloader):
		batch, _ = batch
		data.append(batch.numpy())
		data = np.concatenate(data)
		if len(data.shape) > 2:
		## LJW NOTE: Frouros doesn't like 2D or 3D input, concatenating/flattening
		data = np.reshape(data, (data.shape[0], -1))
		return data
```

The `reset` function will fit the Frouros MMD algorithm with a new batch of data. 
The `check_if_retraining_needed` function will use the Frouros MMD algorithm to estimate the distance between the in-distribution and current data distributions and compare this estimate to a threshold, `alpha`.

```python
	def reset(self, dataloader: DataLoader) -> None:
		"""
		Re-fit MMD to the given dataset

		Args:
		dataloader (DataLoader): A PyTorch dataloader containing the data to fit the MMD algorithm.
		"""
		data = self.__get__data(dataloader)
		self.detector.fit(data)

	def check_if_retraining_needed(
		self, model: BaseModel, dataloader: DataLoader, alpha: float = 0.4
	) -> bool:
		"""
		Check whether retraining is currently needed based on MMD.

		Args:
		model (BaseModel): Model under test.
		dataloader (DataLoader): PyTorch dataloader containing data for calibration error metric calculation.
		alpha (float, optional): P-value threshold. Defaults to 0.2.

		Raises:
		NotImplementedError: If not implemented in inheriting class.

		Returns:
		bool: Specifying whether or not retraining is needed.
		"""
		data = self.__get__data(dataloader)
		self.detector.transform(X=data)
		mmd, p_value = self.detector.distance
		print("MMD:", mmd, "p-value:", p_value)

		if p_value < alpha:
			return True
		else:
			return False
```

## Using the New State Estimator

Finally, to use the new `FrourosMaximumMeanDiscrepancy` state estimator during active learning, modify the `active_params` portion of the active config file (`configs/active_distributed_config.yaml`) as follows, swapping the `state_estimation_method` field for our new state estimator:

```yaml
active_params:
  n_query: 20
  n_iter: 10
  strategy: EntropySampling
  model_selection_method: LEEP
  state_estimation_method: FrourosMaximumMeanDiscrepancy
  state_estimation_host: datacenter
  chkpt_dir: "./checkpoints/al_checkpoints/"
  starting_chkpt: "./checkpoints/al_checkpoints/resnet110_chkpt200.pt"
```

If needed, modify the `results` and `dataset_dirs` to existing paths, and make any additional changes to the configuration file, if desired. 

To run the new custom `FrourosMaximumMeanDiscrepancy` state estimator during active learning, simply run:

```python 
python main.py --config ./configs/active_distributed_config.yaml
```