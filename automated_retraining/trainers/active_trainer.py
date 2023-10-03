# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

import automated_retraining.datasets as datasets
import automated_retraining.model_selection as model_selection
from automated_retraining.datasets import LearnSet, QuerySet, TrainSet
from automated_retraining.models.base_model import BaseModel
from automated_retraining.trainers import BaseTrainer


class ActiveTrainer(BaseTrainer):
    """Trainer class specifically used for active learning. Most notably different from the
    regular `Trainer` by holding multiple models in the `al_models` attribute.

    Args:
        BaseTrainer (models.BaseModel): Base PyTorch model.
    """

    def __init__(
        self,
        al_models: Union[BaseModel, List[BaseModel]],
        training_config: SimpleNamespace,
        train_loader: Optional[Callable] = None,
        val_loader: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Sets the active model, and creates checkpoint prefix based on model architecture,
        `data_module`, experiment name, and epoch number.

        Args:
            al_models (Union[BaseModel, List[BaseModel]]): List of models to used in active learning model selection
            training_config (SimpleNamespace): Parameters used for training specified in the main config file.
            train_loader (Optional[Callable]): When called, sets up the training dataloader. Defaults to None
            val_loader (Optional[Callable]): When called, sets up the val dataloader. Defaults to None.
        """
        self.al_models = al_models
        self.set_model()
        super().__init__(
            self.model,
            training_config,
            train_loader=train_loader,
            val_loader=val_loader,
            **kwargs,
        )
        self.chkpt_prefix = "_".join(
            [
                self.model.architecture,
                training_config.dataset_name,
                training_config.experiment,
                "epoch",
            ]
        ).lower()
        self.__dict__.update(kwargs)

    def set_model(self, model_idx: int = -1) -> None:
        """Sets the model attribute by selecting a model from `al_models` by its index
        in the list.

        Args:
            model_idx (int): Index of model to be selected. Defaults to last model added.
        """
        if isinstance(self.al_models, list) and self.al_models != []:
            self.model = self.al_models[model_idx]
        else:
            self.model = self.al_models

    def select_active_model(
        self, dataset: datasets.QuerySet, al_models: List[BaseModel]
    ) -> Tuple[BaseModel, int, str]:
        """Selects best model candidate for active learning based on
        `model_selection_method` (LEEP, LogME), specified in the base config file.

        Args:
            dataset (datasets.QuerySet): Query Dataset used to get samples for model selection
            al_models (List[BaseModel]): A list of models to select from.

        Returns:
            Tuple[BaseModel, int, str]: Model selected, Index of model in model list,
            name of model checkpoint.
        """

        assert isinstance(
            dataset, datasets.QuerySet
        ), "Incorrect datasets assigned, should be datasets.QuerySet is {}".format(
            type(dataset)
        )
        al_model = al_models[0]
        queried_dataloader = dataset.random_dataloader()

        num_classes = next(iter(dataset.num_classes().items()))[1]
        kargs = {
            "dataloader": queried_dataloader,
            "device": self.training_config.device,
            "target_classes": num_classes,
            "source_classes": num_classes,
        }
        model_selector = getattr(model_selection, al_model.model_selection_method)(
            **kargs
        )

        select_model_idx = model_selector.run_model_selection(al_models)

        print(
            f"Selecting model from checkpoint {al_models[select_model_idx].chkpt_name} for active learning."
        )
        al_models[select_model_idx].set_strategy(
            al_models[select_model_idx].strategy
        )  ## ensure correct strategy is used
        chkpt_name = os.path.basename(al_models[select_model_idx].chkpt_name)
        return al_models[select_model_idx], select_model_idx, chkpt_name

    def get_initial_accuracy(self):
        model_accuracies = {}
        for model_idx, al_model in enumerate(self.al_models):
            self.set_model(model_idx)
            with torch.no_grad():
                model_accuracies[al_model.chkpt] = self.evaluate(
                    self.test_set.test_dataloader()
                )[-1]
        print("Initial model accuracies:")
        for k, v in model_accuracies.items():
            print("\t" + f"{os.path.basename(k)}".ljust(50) + f"{round(v,3)}".ljust(10))

    def active_learning(self) -> None:
        acc = np.zeros(self.n_iter)
        loss = np.zeros(self.n_iter)
        self.query_subset_idx = []
        self.get_initial_accuracy()
        # TODO (CG): Now that query dataloader has all samples - how to select AL model?
        active_model, active_model_idx, active_chkpt_name = self.select_active_model(
            self.query_set, self.al_models
        )
        self.set_model(active_model_idx)
        for i in range(self.n_iter):
            print("Running Query: {}".format(i))
            query_idx, queried = self.model.query(self.query_set, self.n_query)
            if query_idx is None:
                print(f"Not enough samples to query. Not training for iteration {i}.")
            else:
                self.query_subset_idx.append(query_idx)
                self.learn_set = self.learn_set.update_dataset(
                    "learn_dataset",
                    queried.info_df,
                    only_new=False,
                    random_samples=False,
                )
                self.train_model()
                # Early stopping is done per active learning query.
                # Reset metrics between queries.
                if self.training_config.early_stopping:
                    self._reset_early_stopping()
                # Training loss
                loss[i] = self.history["train/epoch_loss"][-1]
                # Test accuracy
                with torch.no_grad():
                    acc[i] = self.evaluate(self.test_set.test_dataloader())[-1]
                print("Loss after query " + str(i) + ": " + str(loss[i]))
                print("Accuracy after query " + str(i) + ": " + str(acc[i]))
                ## delete query samples from pool so they are not selected again
                self.query_set.update_query_set(query_idx)
            active_chkpt_dir = os.path.join(
                self.training_config.results_dir, "al_model_" + active_chkpt_name
            )
            self.save_model(active_chkpt_dir)
            # Load saved model
            self.model.model.load_state_dict(torch.load(active_chkpt_dir))
