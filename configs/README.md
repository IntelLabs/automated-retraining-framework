Config files consist of a set of parameters which will initialize and overwrite defaults in the various submodules. The config files in this folder may be used as a starting point and highlight all of the required parameters, as well as optional parameters that may be used to further customize the training or active learning setup.

When integrating custom modules, it is recommended to add any new parameters to the config files directory to maintain consistency and organization.

## Required config parameters description

```yaml
config: active_learning_distributed   ## required,
```
The `config` parameter selects which mode to run in: training, active_learning_standalone, or active_learning_distributed
```yaml
model_config:   
  model_name: ActiveModel   ## required
  architecture: resnet18   ## required
```
The `model_name` should be available in `./automated_retraining/models` and is useful for selecting custom models to train, and use for active learning. Similarily, the architecture should be available in `./automated_retraining/models/architectures` or in the `torchvision` library. A common selection is `resnet18`. Note that when using the `torchvision` models, additional arguments may be required to initialize the model.
```yaml
training_params:
  weight_decay: 0.0002    ## required
  lr: 0.1   ## required
  gamma: 1    ## required
  momentum: 0   ## required
```
Standard parameters used for model training.
```yaml
training_config:
  with_validation: false    ## required
  results_dir: "./logs/"    ## required
  experiment: ActiveLearningDistributed   ## required
  max_epochs: 10    ## required
  device: cpu    ## required
```
The parameter `with_validation` will enable the validation loop to be run every epoch of training. This is usually desired, however wiht active learning there may not be validaiton data, and the flag can be set to `False`. The `results_dir` and `experiment` help in organizing experiment runs. `max_epochs` sets the max number of training epochs, and `device` selects the hardware to use (cpu/gpu).

```yaml
dataset_config:
  datamodule: MNISTDataModule    ## required
  dataset_dir: "./datasets/MNIST/"    ## required
  n_samples: 0    ## required, 0 uses full dataset
```
The `datamodule` must be available in `./automated_retraining/datasets`, and the `dataset_dir` must already be created. `n_samples` allows for selecting the full dataset to train with (`n_samples: 0`), or any smaller number than the total length of the dataset. 
```yaml
active_params:
  n_query: 20   ## required
  n_iter: 3   ## required
  strategy: EntropySampling   ## required
  model_selection_method: LEEP    ## required
  state_estimation_method: ExpectedCalibrationError   ## required
  state_estimation_host: datacenter   ## required
  chkpt_dir: "./checkpoints/al_checkpoints/mnist/"    ## required
  starting_chkpt: "./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt"    ## required
```
The active learning loop is run `n_iter` times, and on each iteration, `n_query` samples are selected. The query sampling strategy is set with `stragegy` which must be available from `./automated_retraining/query_strategies`. The `model_selection_method` argument is used to determin which model is best suited for transfer learning, and must be available in `./automated_retraining/model_selection`. The metric used to initiate retraining is set by `state_estimation_method` which must be available in `./automated_retraining/state_estimation`. When running distributed active learning, the `state_estimation_host` must be specified. Finally, a directory containing checkpoints for active learning must be specified by `chkpt_dir` and a starting checkpoint must be selected by `starting_chkpt`. 
```yaml
simulator_config:
  n_samples: 100    ## required
  in_distribution_file: "./in_distribution.txt"    ## required
  out_distribution_file: "./out_distribution.txt"    ## required
  distribution_shift: HardSwitch    ## required
  n_shift: 5    ## required only for HardSwitch and UniformShift distribution shifts
```
The simulator helps simulate active learning by modifying dataset distributions, and specifying how the distribution should be shaped. `n_samples` sets the number of samples per distribution to use for each query. the `distribution_file` parameters specify which data should belong to each distribution. The `distribution_shift` should be available in `./automated_retraining/distribution_shifts`, and `n_shift` will specify how many queries of each distribution type should be made before switching.
```yaml
edge_config:
  dataset_config:
    dataset_dir: "./datasets/MNIST/"    ## required
  active_params:
    chkpt_dir: "./datasets/MNIST/"    ## required
    starting_chkpt: "./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt"    ## required
datacenter_config:
  dataset_config:
    dataset_dir: "./datasets/MNIST/"    ## required
  active_params:
    chkpt_dir: "./datasets/MNIST/"    ## required
    starting_chkpt: "./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt"    ## required

```
`edge_config` and `datacenter_config` are used when running in active learning distributed mode and are used to make edge or datacenter specific updates to any of the config parameters above. For example, the edge and datacenter may not have the same file directory structure and store data nad checkpoints in different locations. 

To implement specific updates create a nested dictionary with the config and parameter above that should be updated. An example of changing the dataset directory for the edge
is shown below.

```yaml
edge_config: 
    dataset_config:
        dataset_dir: "./datasets/MNIST"

```
