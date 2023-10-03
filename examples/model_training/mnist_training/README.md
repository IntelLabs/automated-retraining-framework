# Model Training Example - MNIST

Training models from scratch involves a handful of spearate steps, depending on if the datsets and models have been set up already or not. The following example will work through a simple end-to-end model training example including:

- Model setup
- Datamodule and dataset setup
- Config file setup
- Model Training

By the end of this example, there will be an additional datamodule used for managing experiments using the MNIST dataset, which may be resued in subsequent examples.


## Model Setup

New models are expected to inherit from the predefined base model class in `automated_retraining.models.base_model`. For convenience, a `BaseClassifier` module as been configured, which has predefined training and validation accuracy and loss metrics. Note that if implementing a custom model type, the training methods (`forward`, `loss`, etc.) method must be implemented. See the models [README](https://github.com/intel-restricted/applications.ai.active-learning.edge-datacenter-retraining/blob/main/automated_retraining/models/README.md) for more details.

In this example, the `BaseClassifier` module will be used as it already has the necessary methods and metrics needed for training a model with the MNIST dataset. 

To set up the MNIST model, first create a new python file, `automated_retraining/models/mnist_model.py` and import `BaseClassifier`, and create a new class, `MNISTModel`. The `kwargs` here will typically come from parameters specified in the config file.


```python
from automated_retraining.models.base_model import BaseClassifier

class MNISTModel(BaseClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

```

Additionally, add the following line to the imports in `automated_retraining/models/__init__.py`.
```python
from automated_retraining.models.mnist_model import MNISTModel
```

After that, the `MNISTModel` is set up and ready to use.

## Dataset and Datamodule Setup

Similar to the model setup, new datasets and datamodules should inherit from the predefined base classes in `automated_retraining.datasets`, namely: `BaseSet` and `BaseDataModule`. For more information on the dataset and datamodule structure, see the associated [README](https://github.com/intel-restricted/applications.ai.active-learning.edge-datacenter-retraining/blob/main/automated_retraining/datasets/README.md).

In this example, the `BaseDataModule` and `BaseDataset` will be used. To start, create a new python file `automated_retraining/datasets/mnist_dataset.py`. First, the new datamodule, `MNISTDataModule`, will start by subclassing the `BaseDataModule`, and adding in the necessary input arguments. For this dataset, it is necessary to include: `batch_size`, `num_classes`, `n_samples`, and `dataset_name`. Note that these required inputs may change depending on the dataset being used.

Two methods, `get_dataset()` and `get_dataloder()` should be implemented which help to set up and datasets and dataloders that may be needed. The `BaseDataModule` has a template of methods which may be implemented. Note that in the MNIST dataset from torchvision, a flag `train`, help determine whether to load in training or validation data. Additionally, data transforms may be defined in the `__init__` of the `MNISTDataModule`. 

```python
class MNISTDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_classes: Dict = {"digit": 10},
        n_samples: int = 0,
        dataset_name: str = "MNISTDataset",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset = getattr(sys.modules[__name__], dataset_name)
        self.batch_size: int = batch_size
        self.num_classes: Dict = num_classes
        self.n_samples: int = n_samples
        self.dims = (3, 28, 28)
        # Define image transformations
        transform_list = [
            transforms.Resize(size=self.dims[1:]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
        self.train_transform = transforms.Compose(transform_list)
        self.val_transform = transforms.Compose(transform_list[-2:])

    def get_dataset(
        self,
        dataset_dir: str,
        num_classes: Dict,
        transform: transforms,
        n_samples: int,
        train_data: bool = True,
    ) -> BaseDataset:
        """Helper method to get a dataset and subsample if necessary.

        Args:
            dataset_dir (str): Directory where data is stored
            num_classes (Dict): Num classes as dict, {"digits", 10}
            transform (transforms): Data transforms to apply
            n_samples (int): Number of data points to select
            train_data (bool, optional): Flag indicating if data is for training or validation. Defaults to True.

        Returns:
            BaseDataset: The dataset
        """

        dataset = self.dataset(dataset_dir, num_classes, transform)
        if n_samples != 0 and len(dataset) > 0:
            inds = np.random.choice(np.arange(len(dataset)), n_samples, replace=False)
            dataset.subset(inds)
        else:
            print(
                f"Warning: dataset contains no samples and a subset has not been selected"
            )
        return dataset

    def create_training_set(self) -> None:
        """Training set creation."""
        self.train = True
        self.train_dataset = self.get_dataset(
            self.dataset_dir,
            self.num_classes,
            self.train_transform,
            int(self.n_samples * self.data_split[0]),
        )
        self.train = False
        self.val_dataset = self.get_dataset(
            self.dataset_dir,
            self.num_classes,
            self.train_transform,
            int(self.n_samples * sum(self.data_split[1:])),
        )

    def get_dataloader(self, dataset: BaseDataset) -> DataLoader:
        """Template for getting a standard pytorch dataloader.

        Args:
            dataset (BaseDataset): Dataset to use with dataloader

        Returns:
            DataLoader: The dataloader
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Get the train dataloader.

        Returns:
            DataLoader: Train dataloader
        """
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        """Get the val dataloader.

        Returns:
            DataLoader: Val dataloader.
        """
        return self.get_dataloader(self.val_dataset)

```

The `MNISTDataset` class will hold all of the information about the dataset and provide the utility to load and set new data. The torchvision class for MNIST will be subclassed into a new DataModule: `MNISTDataModule` which has useful methods for downloading the original dataset and managing which data is being used. Additionally, `BaseDataset` will be inherited to make use of its structure and methods for creating subsets and concatenating datasets together. To initialize an `MNISTDataset`, the inputs `datasetdir`, `num_classes` and `transforms` are required. Other useful and reused information in also populated in the in `__init__` which specifies the headers and types of data that will go in to the data attribute, `self.info_df`. The `info_df` attributed is used throughout this repository as a way of selecting and storing data to use in training and active learning. 

A method `load_data()` is defined which takes care of loading data (or downloading the data if necessary), and building the required `info_df`. 

The methods `__getitem__` and `__len__` are already defined in the `MNIST` torchvision dataset class, however they are explicity shown here for demonstration. 
```python
class MNISTDataset(BaseDataset, MNIST):
    def __init__(
        self,
        dataset_dir: str,
        num_classes: dict,
        transform,
        train_data=True,
    ) -> None:
        """Set up the MNIST Dataset

        Args:
            data_dir (str): Directory where data is located.
            num_classes (dict): Class category and number of classes
            transform (_type_): Transforms to apply to data.
            train_data (bool, optional): Flag indicating whether to load train or test data. Defaults to True.
        """
        super(MNIST, self).__init__(dataset_dir)
        MNIST.__init__(self, self.root, train=train_data)
        self.train = train_data
        self.dataset_dir: str = dataset_dir
        self.num_classes: dict = num_classes
        self.transform = transform
        self.info_df_headers = ["digit", "for_training"]
        self.info_df_dtypes = {"digit": int, "for_training": bool}
        self.info_df: pd.DataFrame = pd.DataFrame(columns=self.info_df_headers)
        self.load_data(from_init=True)

    def load_data(
        self,
        loadfiles: pd.DataFrame = None,
    ) -> None:
        """Load data, on init or from a particular file. Useful when assigning new data
        to the dataset.

        Args:
            loadfiles (pd.DataFrame, optional): Dataframe of data filenames. Defaults to none.
        """
        try:
            if loadfiles is None:
                self.info_df = pd.DataFrame(columns=self.info_df_headers)
                try:
                    self.data, self.targets = self._load_data()
                except FileNotFoundError:
                    for self.train in [False, True]:
                        self.download()
                    self.data, self.targets = self._load_data()
                info_df_data = [
                    np.arange(len(self.targets)),
                    self.targets,
                    [self.train] * len(self.targets),
                ]
                self.info_df = self.concat_datasets(
                    pd.DataFrame(dict(zip(self.info_df_headers, info_df_data)))
                )
            else:
                self.set_data(loadfiles)

        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Must load in info_df manually")

    def set_data(self, new_info_df: pd.DataFrame):
        """Assign a dataframe to the `info_df` attribute.

        Args:
            new_info_df (pd.DataFrame): New dataframe.
        """
        self.info_df = new_info_df
        self.info_df.columns = self.info_df_headers

    def __getitem__(self, *args, **kwargs):
        """Overload MNIST class __getitem__ just for demonstration."""
        img, target = MNIST.__getitem__(self, *args, **kwargs)
        return img, target

    def __len__(self):
        return len(self.info_df)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "raw")
        
```

Finally, import the `MNISTDataModule` in the `__init__.py` of the datasets folder.
```python
from automated_retraining.datasets.mnist_dataset import MNISTDataModule
```

## Config File Setup

The base config file (`./configs/base_config.yaml`) for training divides up the inputs arguments into five sections, helping split out what is required for the model, dataset, and trainer. The fields for `model_name` and `datamodule` can be replaced with the new classes, `MNISTModel` and `MNISTDataModule` respectively. An appropriate architecture can be selected, and the `results` and `dataset_dirs` can be changed to existing paths.
Save the config into the configs folder as `mnist_training.yaml`. 
```yaml
config: training
model_config:
  model_name: MNISTModel
  architecture: CifarResnet20
training_params:
  batch_size: 64
  weight_decay: 0.0002
  lr: 0.1
  gamma: 0.1
  momentum: 0.9
training_config:
  with_validation: true
  results_dir: "./logs"
  experiment: debug
  max_epochs: 3
  device: cpu
dataset_config:
  datamodule: MNISTDataModule
  dataset_dir: "./data/MNIST_data"
  n_samples: 1000
  num_classes:
    digit: 10
```
After setting up the config, all that needs to be done is to run training.

```python 
python main.py --config ./configs/mnist_training.yaml
```