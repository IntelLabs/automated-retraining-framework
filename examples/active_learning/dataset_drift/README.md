# Active Learning Example - Dataset Drift and Model Selection

A common problem in active learning is identifying dataset drift, and selecting a model to fine-tune to the new data. In this example, the MNIST dataset is used with different color channel augmentations to simulate dataset drift, and the built in model selection metrics will be utilized to pick a model best suited for the fine-tuning update. This example will walk through the following steps:

- Dataset setup (modifications to MNIST Dataset)
- Color channel transform setup
- Training two models on MNIST-Red, and MNIST-Green
- Using the distributed active learning mode to test dataset drift detection and model selection

## Dataset setup
A few modifications need to be made to the existing [MNIST dataset](https://github.com/intel-restricted/applications.ai.active-learning.edge-datacenter-retraining/blob/main/automated_retraining/datasets/mnist_dataset.py) to use it with the active learning workflow. Specifically, there are new dataloaders and datasets that are expected during active learning. The `query`, `learn` and `random` dataloaders and datasets must be added in addition to what is already in  `mnist_dataset.py`.

``` python
    def create_query_set(self) -> None:
        """Query set creation."""
        self.query_dataset = self.get_dataset(
            self.dataset_dir,
            self.val_transform,
            int(self.n_samples * self.data_split[0]),
        )
        self.num_classes = self.query_dataset.num_classes

    def create_random_set(self) -> None:
        """Random set creation."""
        self.random_dataset = self.get_dataset(
            self.dataset_dir,
            self.val_transform,
            int(self.n_samples * self.data_split[0]),
        )
        self.num_classes = self.random_dataset.num_classes

    def create_learn_set(self) -> None:
        """Learn set creation."""
        self.learn_dataset = self.get_dataset(
            self.dataset_dir,
            self.val_transform,
            int(self.n_samples * self.data_split[0]),
        )
        self.num_classes = self.learn_dataset.num_classes

    def random_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        Returns:
            DataLoader: Test dataloader.
        """
        return self.get_dataloader(self.random_dataset)

    def learn_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        Returns:
            DataLoader: Test dataloader.
        """
        return self.get_dataloader(self.learn_dataset)

    def query_dataloader(self) -> DataLoader:
        """Get the test dataloader.
        Returns:
            DataLoader: Test dataloader.
        """
        return self.get_dataloader(self.query_dataset)
```



## Color channnel transform setup
To introduce a new type of data, color augmentations will be done to the MNIST dataset by implementing a custom transform with the `torchvision` library in `./automated_retraining/datasets/utils/custom_transforms.py`.

An images color channel is modified by specifying the red, green, and blue multiplier to be applied to their respective color channel. To make a fully green image, the transform would be `ColorAugmentation([0, 1, 0])`.

```python
class ColorAugmentation(object):
    """Custom data transform used to alter the color channel (R, G, B) of an image.
    The augmentation used a color tuple specifying the respective multiplier to apply to
    each color channel. To use an unmodified version of the image, the color multiplier
    values would be (1, 1, 1). A red image would be created by (1, 0, 0). Note that
    (0, 0, 0) will return an all black image. Mixtures of color multipliers may also be
    used.
    """

    def __init__(self, color: List =[1, 1, 1]):
        self.name = "ColorAugmentation"
        self.color = color

    def __call__(self, img):
        red, green, blue = self.color
        color_transform = (red, 0, 0, 0, 0, green, 0, 0, 0, 0, blue, 0)
        img = img.convert("RGB", color_transform)
        return img

    def __repr__(self):
        red, green, blue = self.color
        return f"{self.name}(red={red}, green={green}, blue={blue})"
```

The MNIST data module is modified to use a color augmentation by default, which keeps the image in grayscale. 

An additional `distribution_transforms` parameter will be added to the `MNISTDataModule` that will allow changing color channel from the config file itself during active learning. 

```python
class MNISTDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        n_samples: int = 0,
        dataset_name: str = "MNISTDataset",
        distribution_transforms: Optional[Dict] = {},
        **kwargs,
    ) -> None:
```

If these distribution transforms are included, a custom subclass of `torchvision.Compose` called `DistributionCompose` will be used to ensure the transforms will be included (`./automated_retraining/datasets/utils/custom_transforms.py`).


The transform can then be added to the transforms list:
```python
transform_list = [
    transforms.Resize(size=self.dims[1:]),
    transforms.Grayscale(num_output_channels=3),
    ColorAugmentation(color=color),
    transforms.ToTensor(),
]       
```

## Model Training
To test dataset drift and model selection methods, a few model checkpoints need to be available. When dataset drift is detected, all available models will be evaluated to determine which is best suited for fine-tuning. In this example, two models will be trained on MNIST-Red and MNIST-Green. To see the training steps in detail, refer to the [MNIST training example](https://github.com/intel-restricted/applications.ai.active-learning.edge-datacenter-retraining/tree/main/examples/model_training/mnist_training).

Most importantly, the color values in the MNIST dataset transform will be changed for each model training, and the checkpoints saved so they can be used in active learning. 


## Dataset drift detection and model selection
A separate config file with active learning specific parameters should be used. An example is given for MNIST below. Note that the `active_learning_distributed` mode is specified which allows for simulating dataset shift. 

```yaml
---
config: active_learning_distributed
model_config:
  model_name: ActiveModel
  architecture: resnet18
training_params:
  batch_size: 64
  weight_decay: 0.0002
  lr: 0.1
  gamma: 0.1
  momentum: 0.9
training_config:
  with_validation: false
  results_dir: "./logs/"
  experiment: MNISTAL_debug
  max_epochs: 10
  device: cuda
dataset_config:
  datamodule: MNISTDataModule
  dataset_dir: "./datasets/MNIST"
  n_samples: 1000
  distribution_transforms:
    ColorAugmentation:
    - 1
    - 0
    - 0
active_params:
  n_query: 20
  n_iter: 20
  strategy: MarginSampling
  model_selection_method: LEEP
  state_estimation_method: ExpectedCalibrationError
  state_estimation_host: datacenter
  chkpt_dir: "./checkpoints/al_checkpoints/mnist"
  starting_chkpt: "./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt"
simulator_config:
  n_samples: 100
  distribution_shift: HardSwitch
  in_distribution_file: in_distribution.txt
  out_distribution_file: out_distribution.txt
  n_shift: 5
  distribution:
  - 0.5
  - 0.5
  in_transform:
    ColorAugmentation:
    - 1
    - 0
    - 0
  out_transform:
    ColorAugmentation:
    - 0
    - 1
    - 0

```
In this example, the dataset shift simulator is set to switch the dataset every 5 iterations. To create the in and out of distribution text files, the utility script below may be used. 
```python
from automated_retraining.datasets.mnist_dataset import MNISTDataset

data = MNISTDataset(
    dataset_dir="../../../data/MNIST", transform=None, train_data=True, label="digit"
)

in_distribution = data.info_df.sample(30000, replace=False)
out_distribution = data.info_df.drop(in_distribution.index)

in_distribution.to_csv("in_distribution.txt", header=None, sep=",")
out_distribution.to_csv("out_distribution.txt", header=None, sep=",")
```

The color transform corresponding to in distribution and out of distribution data are specified and are applied by the shift simulator.


The example can now be run by
```python
python main.py --config PATH_TO_CONFIG.yaml
```

As noted in the config file, the starting checkpoint will be `mnist_resnet18_red.pt` and the color transform will be red as well. After 5 iterations of querying new data, the shift simulator will switch and a green color transform will be applied. This shift will be detected through the `state_estimation_method` and the best suited model will be updated. In this config, the `ExpectedCalibrationError` (ECE) is monitored, and retraining is needed whenever this metric exceeds one standard deviation of the mean of the last 5 ECE values.

An example output is shown below.

```console
Running iteration 0
Loss: 0.00       Accuracy: 100.00

Running iteration 1
Loss: 0.00       Accuracy: 100.00
Avg ECE : 0.000 Cur ECE: 0.000

Running iteration 2
Loss: 0.00       Accuracy: 100.00
Avg ECE : 0.000 Cur ECE: 0.000

Running iteration 3
Loss: 0.00       Accuracy: 100.00
Avg ECE : 0.000 Cur ECE: 0.000

Running iteration 4
Loss: 0.00       Accuracy: 100.00
Avg ECE : 0.000 Cur ECE: 0.000

Running iteration 5
Loss: 3.21       Accuracy: 88.29
Avg ECE : 0.017 Cur ECE: 0.083


LEEP scores: Model ./checkpoints/al_checkpoints/mnist/mnist_resnet18_green.pt -1.6120014453957026
LEEP scores: Model ./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt -0.6073212121690683
Selecting model from checkpoint ./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt for active learning.


Active Models Available:
        ./checkpoints/al_checkpoints/mnist/mnist_resnet18_green.pt
        ./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt


Selecting active model: mnist_resnet18_red.pt
Loading weights from:  ./checkpoints/al_checkpoints/mnist/mnist_resnet18_red.pt
Using model copy for active learning ./checkpoints/al_checkpoints/mnist/al_chkpt_updates/mnist_resnet18_red_al_update_0.pt



Length of training set: 20
Training Epoch 0: : 2batch [00:00,  2.18batch/s, Batch_Acc=90, Batch_Loss=0.141, Epoch_Avg_Acc=90, Epoch_Avg_Loss=0.141]
...
Training Epoch 9: : 2batch [00:00,  2.21batch/s, Batch_Acc=99, Batch_Loss=0.00162, Epoch_Avg_Acc=99, Epoch_Avg_Loss=0.00162]

Loading weights from:  ./checkpoints/al_checkpoints/mnist/al_chkpt_updates/mnist_resnet18_red_al_update_0.pt

Avg ECE : 0.096 Cur ECE: 0.397

Retraining needed: True


Length of training set: 40
Training Epoch 0: : 2batch [00:00,  2.16batch/s, Batch_Acc=97.1, Batch_Loss=0.533, Epoch_Avg_Acc=97.1, Epoch_Avg_Loss=0.533]
...
Training Epoch 9: : 2batch [00:00,  2.21batch/s, Batch_Acc=98.8, Batch_Loss=0.00298, Epoch_Avg_Acc=98.8, Epoch_Avg_Loss=0.00298]

Loading weights from:  ./checkpoints/al_checkpoints/mnist/al_chkpt_updates/mnist_resnet18_red_al_update_0.pt
Avg ECE : 0.129 Cur ECE: 0.165
Retraining needed: False

```

In the above example, the model starts withe the red version of MNIST, and also the pretrained red MNIST checkpoint. At iteration 5, green samples are introduced, causing a drop in model accuracy and an increase in Expected Calibration Error (ECE). The transfer learning metric LEEP is used to select the model checkpoint best suited for an active learning update, and once the update is finished the normal iteration process resumes.


Using this framework, different dataset drift identification algorithms can be tested, as well as testing how different types of dataset drift impact model performance, model selection, and transfer learning ability. 