# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import copy
import os
from types import SimpleNamespace
from typing import Tuple, Union

import torch

import automated_retraining.datasets as datasets
import automated_retraining.models as models
from automated_retraining.datasets import LearnSet, QuerySet
from automated_retraining.submodules.submodule_utils import SubModule
from automated_retraining.trainers import ActiveTrainer, configure_training


class ActiveLearningStandalone(SubModule):
    """
    Module used to run active learning in standalone mode.
    All training, evaluation, model updates, and querying happen on the same device.

    Args:
        SubModule (submodules.SubModule): Base submodule.
    """

    def __init__(self, **kwargs):
        """
        Set up submodule using any kwargs, typically specifying a config file to load
        parameters from.
        """
        self.__dict__.update(kwargs)
        self.parse_config(self.config)

    def __setup(self):
        """Setup the sets used in active learning standalone mode.

        Returns:
            None: _description_
        """
        # Set up and additions to dataset, training config
        query_set: Union[
            datasets.LearnSet, datasets.QuerySet, datasets.TrainSet
        ] = datasets.configure_dataset(self.dataset_config, dataset_type="QuerySet")
        assert isinstance(
            query_set, datasets.QuerySet
        ), "Incorrect datasets assigned, should be datasets.QuerySet is {}".format(
            type(query_set)
        )
        test_set: Union[
            datasets.LearnSet, datasets.QuerySet, datasets.TrainSet
        ] = datasets.configure_dataset(self.dataset_config, dataset_type="TrainSet")
        assert isinstance(
            test_set, datasets.TrainSet
        ), "Incorrect datasets assigned, should be datasets.TrainSet is {}".format(
            type(test_set)
        )
        learn_set_config = copy.deepcopy(self.dataset_config)
        learn_set: datasets.LearnSet = datasets.configure_dataset(
            learn_set_config, dataset_type="LearnSet"
        )
        assert isinstance(
            learn_set, datasets.LearnSet
        ), "Incorrect datasets assigned, should be datasets.LearnSet is {}".format(
            type(learn_set)
        )
        ## LJW NOTE: assuming query dataset labelled
        self.dataset_config.num_classes = query_set.num_classes()
        self.training_config: SimpleNamespace = configure_training(
            self.training_config, self.dataset_config, self.model_config
        )
        return (query_set, learn_set, test_set)

    def gather_checkpoints(self):
        checkpoints = [
            chkpt
            for chkpt in os.listdir(self.active_params.chkpt_dir)
            if ".pt" in chkpt
        ]
        return checkpoints

    def create_initial_checkpoint(self):
        active_model: models.BaseModel = models.configure_model(
            self.model_config,
            self.dataset_config,
            self.training_config,
            self.training_params,
        )
        torch.save(
            active_model.model.state_dict(),
            self.active_params.chkpt_dir
            + f"/{self.model_config.architecture}_initial_chkpt.pt",
        )

    def build_al_models(self, checkpoints):
        al_models = []
        model_chkpts = []
        for checkpoint in checkpoints:
            model_chkpts.append(os.path.join(self.active_params.chkpt_dir, checkpoint))
            self.model_config.chkpt = model_chkpts[-1]
            active_model: models.BaseModel = models.configure_model(
                self.model_config,
                self.dataset_config,
                self.training_config,
                self.training_params,
            )
            active_model.chkpt_name = checkpoint
            active_model.__dict__.update(**vars(self.active_params))
            # import ipdb; ipdb.set_trace();
            al_models.append(active_model)
        return al_models, model_chkpts

    def active_learning_standalone(self):
        """
        Run active learning in standalone mode. A `query_dataset` and `learn_dataset`
        are used along with a set of predefined checkpoints, specified in the config
        file under `active_parmas.chkpt_dir`.

        A model using each checkpoint is created and wrapped with a trainer.

        The active learning loop takes place in the `ActiveTrainer` class.
        """
        # TODO (CG): Test set is using all samples right now. Might not be desired.
        query_set, learn_set, test_set = self.__setup()
        model_chkpts = []
        checkpoints = self.gather_checkpoints()
        if checkpoints == []:
            self.create_initial_checkpoint()
            checkpoints = self.gather_checkpoints()

        al_models, model_chkpts = self.build_al_models(checkpoints)
        self.active_params.model_chkpts = model_chkpts

        self.training_config.dataset_name = learn_set.data_module.__class__.__name__
        train_loader = learn_set.learn_dataloader
        active_trainer = ActiveTrainer(
            al_models,
            self.training_config,
            train_loader=train_loader,
            **vars(self.active_params),
        )
        # TODO (CG): Clean up - blank out the info_df of learn set.
        learn_set.data_module.learn_dataset.set_data(
            learn_set.data_module.learn_dataset._info_df
        )
        active_trainer.query_set = query_set
        active_trainer.learn_set = learn_set
        active_trainer.test_set = test_set
        active_trainer.active_learning()
        return

    def run(self):
        """
        Run active learning in standalone mode.
        """
        self.active_learning_standalone()
