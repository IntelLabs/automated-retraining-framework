# Datasets

The directory contains the datasets for use with the trainer module

Datasets have three components:
 - `Dataset` - a standard Pytorch dataset
 - `Datamodule` - wrapper class around the `Dataset` with functions to create the needed data splits, and to return the dataloaders for each data split
 - `Set` - wrapper class around `Datamodule`, the framework uses data in three ways: i) for training a model from scratch, ii) to query samples for active learning, and iii) to implement active learning

Of the three components, a user will have to implement a `Dataset` and a `Datamodule` for their data. Instructions for implementing those is below. 

## Creating a Dataset

New datasets should inherit from the `BaseDataset` class in `./datasets/dataset.py`, and, like other datasets in PyTorch, need to implement the `__getitem__` function to return a data sample and label. Additionally, datasets should implement the `__len__` function that returns the length of the dataset. 

## Creating a Datamodule

One layer of abstraction higher is the datamodule, which new datasets should inherit from the `BaseDataModule` class in `./datasets/dataset.py`. The `BaseDataModule` is structured as such:

```python
 class BaseDataModule:

    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.query_dataset = None
        self.learn_dataset = None
        self.num_classes: dict = {}

    def create_learn_set(self):
        raise NotImplementedError

    def create_query_set(self):
        raise NotImplementedError

    def create_training_set(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError
    
    def val_dataloader(self):
        raise NotImplementedError
    
    def test_dataloader(self):
        raise NotImplementedError
    
    def query_dataloader(self):
        raise NotImplementedError
    
    def learn_dataloader(self):
        raise NotImplementedError
```

In `BaseDataModule` there are three components that are implemented multiple times for different dataset uses. The components are:

 - `self.{train,val,test,query,learn}_dataset` - the datamodule should have a dataset for each potential use, e.g., train, query, or learn. The dataset is a version of the user implemented dataset class derived from `BaseDataset`, and typically will consist of different partitions of the dataset or new data gathered during the active learning process. 
 - `def create_{learn,query,training}_set` - this function is used to partition the entire dataset into subsets. Depending on the use different datasets will be constructed, i.e, for `create_training_set` used in training a model from scratch, the `train_dataset`, `val_dataset`, and `test_dataset` datasets will be created. 
 - `{train,val,test,query,learn}_dataloader` - each dataset implemented within the `DataModule` requires a dataloader to wrap the dataset. This function will return a dataloader for the dataset specified.

## Enable Dataset in the Trainer Module

 In order to make the custom dataset accessible in the `Trainer` module edit the file `./datasets/__init__.py` and add an import statement, for example: 
 ```python 
 from datasets.mnist_dataset import MNISTDataModule
```

The import statement makes the dataset available, to use the dataset to train a model edit the config file (`./configs/base_config.yaml`) to use the custom dataset. The `dataset_config` dictionary in the config (shown below) should be modified with necessary parameters for the dataset. The necessary parameters to specify are: `datamodule`, `dataset_name` (string name for dataset, used for logs), `dataset_dir` (directory where data is located). 

```yaml 
    dataset_config:
        datamodule: MNISTDataModule
        dataset_dir: "./datasets/MNIST/"
        n_samples: 10000
```
