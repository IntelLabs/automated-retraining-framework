# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import asyncio
import copy
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

import automated_retraining.datasets as datasets
import automated_retraining.models as models
from automated_retraining.datasets import BaseDataModule, QuerySet, configure_dataset
from automated_retraining.models import active_model
from automated_retraining.submodules.submodule_utils import SubModule
from automated_retraining.trainers import ActiveTrainer, configure_training
from automated_retraining.utils.communication import AsyncServer


class DataCenterModule(SubModule):
    def __init__(self, config: str, **kwargs) -> None:
        """Datacenter module used for model retraining, checking calibration metrics,
        and managing models.

        Args:
            config (str): Config to use.
        """
        self.__dict__.update(kwargs)
        self.datasets: Dict = {}
        # self.retraining_needed: bool = False
        self.parse_config(config)
        self.__setup()

    def __setup(self) -> None:
        """Set up the data center module and create active trainer attribute."""
        self.chkpt_dir = self.active_params.chkpt_dir
        self.al_update_dir = os.path.join(self.chkpt_dir, "al_chkpt_updates")
        if not os.path.exists(self.al_update_dir):
            os.mkdir(self.al_update_dir)
        self.configure_datasets(["learn_dataset", "random_dataset"])

        self.create_active_models()
        if self.active_params.state_estimation_host == "datacenter":
            self.load_state_estimation_utils(
                state_estimator=self.active_params.state_estimation_method
            )
        self.training_config.dataset_name = next(
            iter(self.datasets.values())
        ).data_module.__class__.__name__
        self.active_trainer = ActiveTrainer(
            self.al_models,
            self.training_config,
            train_loader=self.datasets["learn_dataset"].learn_dataloader,
            **vars(self.active_params),
        )
        self.select_active_model(chkpt=self.chkpt)

    def get_checkpoint_files(self) -> List[str]:
        """Get list of checkpoints to use for active learning.

        Returns:
            List[str]: List of paths to checkpoints.
        """
        files = os.listdir(self.chkpt_dir)
        self.model_checkpoints = [
            os.path.join(self.chkpt_dir, chkpt) for chkpt in files if ".pt" in chkpt
        ]
        return self.model_checkpoints

    def create_active_models(self) -> None:
        """Create a list of active learning models."""
        self.get_checkpoint_files()
        learn_dataset: datasets.LearnSet = configure_dataset(
            self.dataset_config, "LearnSet"
        )

        self.training_config: SimpleNamespace = configure_training(
            self.training_config, self.dataset_config, self.model_config
        )
        al_models = []
        for checkpoint in self.model_checkpoints:
            active_model = self.create_model(checkpoint)
            al_models.append(active_model)

        self.al_models = al_models

    def select_active_model(
        self,
        dataset: BaseDataModule = None,
        al_models: List[str] = None,
        chkpt: str = None,
    ) -> None:
        """Select best candidate for an active learning models based on transfer
        learning metrics. If a checkpoint is supplied, that is used as the AL model. If
        a list of `al_models` is supplied, those are used + transfer learning metrics
        to select the best model candidate.

        Args:
            dataset (BaseDataModule, optional): Dataset to use in model selection. Defaults to None.
            al_models (List[str], optional): List of model checkpoints. Defaults to None.
            chkpt (str, optional): Checkpoint to select. Defaults to None.
        """
        if chkpt is not None:
            for model_idx, model in enumerate(self.al_models):
                if chkpt in model.chkpt_name:
                    self.active_trainer.set_model(model_idx)
        else:
            model, model_idx, chkpt_name = self.active_trainer.select_active_model(
                dataset, al_models
            )
            update_num = len(
                [f for f in os.listdir(self.al_update_dir) if "al_update" in f]
            )
            new_chkpt = os.path.join(
                self.al_update_dir,
                chkpt_name.split(".")[0] + "_al_update_" + str(update_num) + ".pt",
            )
            print("\n")
            print(f"Active Models Available:")
            [print(f"\t{mod.chkpt_name}") for mod in al_models]
            print("\n")
            print(f"Selecting active model: {chkpt_name}")
            new_model: models.BaseModel = self.create_model(
                al_models[model_idx].chkpt_name
            )
            new_model.chkpt_name = new_chkpt
            self.al_models.append(new_model)

            # Without parameters self.active_trainer.set_model()
            # set the last model in list as active
            self.active_trainer.set_model()
            self.save_model(self.active_trainer.model.chkpt_name)
            print(
                f"Using model copy for active learning {self.active_trainer.model.chkpt_name}\n"
            )

    def create_model(self, chkpt_name: str) -> models.BaseModel:
        """Load model from specified checkpoint.

        Args:
            chkpt_name (str): Path to checkpoint to load model from.

        Returns:
            models.BaseModel: Returns loaded model.
        """
        self.model_config.chkpt = chkpt_name
        active_model: models.BaseModel = models.configure_model(
            self.model_config,
            self.dataset_config,
            self.training_config,
            self.training_params,
        )
        active_model.chkpt_name = chkpt_name
        active_model.__dict__.update(**vars(self.active_params))
        return active_model

    def save_model(self, chkpt_name: str) -> None:
        """Save model to new checkpoint.

        Args:
            chkpt_name (str): New checkpoint name.
        """
        obj = self.active_trainer.model
        while hasattr(obj, "model"):
            obj = obj.model
        torch.save(obj.state_dict(), chkpt_name)

    def load_model(self, chkpt_name: str) -> None:
        """Load model from checkpoint.

        Args:
            chkpt_name (str): Checkpoint to load from.
        """
        obj = self.active_trainer.model
        while hasattr(obj, "model"):
            obj = obj.model
        obj.load_state_dict(
            torch.load(chkpt_name, map_location=self.training_config.device)
        )

    def normal_operation(self, random_dataframe: pd.DataFrame) -> bool:
        """Datacenter module operating normally to check if retraining needed

        During normal operation the edge will periodically send randomly sampled
        data to the datacenter. The datacenter will use state_estimation metrics
        to check if retraining of the edge model should be done. A True/False flag
        is returned to edge indicated whether or not to start retraining

        Args:
            random_dataframe (pd.DataFrame): dataframe with information necessary
            to process the randomly sampled data sent from the edge.

        Returns:
            bool: retraining flag is returned indicating, True: retraining is needed,
            False: retraining is not needed at this point.
        """
        self.set_info_df("random_dataset", info_df=random_dataframe)

    def prepare_for_retraining(
        self, random_dataframe: pd.DataFrame, retraining_needed: bool = True
    ) -> None:
        """Prepares datacenter module to begin new round of retraining.

        The learn_dataset is emptied so it can accept new data samples
        during retraining. Using randomly sampled data the datacenter
        selects which model to use for retraining, based on which one
        is likely to reproduce the best post-transfer accuracy.

        Args:
            retraining_needed (bool): Boolean flag sent indicating if
            retraining is necessary
            random_dataframe (pd.DataFrame): dataframe with information necessary
            to process the randomly sampled data sent from the edge.
        """
        self.full_queried_df = None
        self.set_info_df(
            "learn_dataset",
            info_df=pd.DataFrame(
                columns=self.get_dataset(next(iter(self.datasets))).info_df_headers
            ),
        )
        self.set_info_df("random_dataset", info_df=random_dataframe)
        random_dataset = self.datasets["random_dataset"]
        if retraining_needed:
            self.al_iters_complete = 0
            self.select_active_model(random_dataset, self.al_models)

    def retraining(
        self,
        query_dataframe: pd.DataFrame,
        random_dataframe: pd.DataFrame,
        retraining_needed: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """Datacenter performs retraining on new data actively queried by edge

        Queried data is sent to datacenter from the edge. The datacenter
        trains the model currently selected with new data. After training
        random data is used to check if another round of retraining is needed.
        The retraining flag and new model checkpoint are sent to the edge.

        Args:
            retraining_needed (str): TODO: remove, unnecessary, if edge sent
            retraining message then retraining should be done.
            query_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the samples from the edge selected with active learning query. Defaults to None.
            random_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the randomly sampled data sent from the edge. Defaults to None.

        Returns:
            Tuple[bool, Optional[str]]: retraining_needed is updated flag indicating if further retraining is required.
            new_chkpt is string with path to the new model checkpoint after retraining.
        """
        # add queried samples to full dataframe of all samples queried so far
        self.full_queried_df = self.concat_datasets(
            "learn_dataset", new_df=query_dataframe
        )
        self.set_info_df("learn_dataset", info_df=self.full_queried_df)
        self.set_info_df("random_dataset", info_df=random_dataframe)
        random_dataset = self.datasets["random_dataset"]
        # Make sure the trainer is able to use the dataset we want by setting it to the model object
        self.active_trainer.dataset = self.datasets["learn_dataset"]
        print(f"Length of training set: {len(self.active_trainer.dataset)}")
        self.active_trainer.train_model()
        self.save_model(self.active_trainer.model.chkpt_name)
        self.load_model(self.active_trainer.model.chkpt_name)
        self.al_iters_complete += 1

        return self.active_trainer.model.chkpt_name

    def serve(
        self,
        message_type: str,
        query_dataframe: Optional[pd.DataFrame] = None,
        random_dataframe: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Callback function used by datacenter server.

        The serve callback receives messages from the edge.
        Depending on the message type the datacenter processes
        the content of the messages in different ways. Valid messages are:
            - handshake: initial handshake message to initiate communication
            between edge and datacenter.
            - normal: normal operating mode where edge periodically sends
            randomly sampled data to datacenter, and datacenter uses this data
            to check if retraining is necessary.
            - prep_retraining: to prepare for retraining, the datacenter
            clears out previous dataset used to train model, and selects
            model to begin retraining from
            - retraining: retraining phase, datacenter trains model with new
            actively queried data from edge and checks if retraining is still
            necessary.

        Args:
            message_type (str): sent by edge to indicate what type of message was sent, and direct
            the datacenter how to process the request.
            query_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the samples from the edge selected with active learning query. Defaults to None.
            random_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the randomly sampled data sent from the edge. Defaults to None.

        Returns:
            Tuple[Optional[bool], Optional[str]]: retraining_needed is updated flag indicating if further retraining is required.
            new_chkpt is string with path to the new model checkpoint after retraining.
        """
        if message_type == "handshake":
            # TODO: figure out what to send for handshake
            return "Hello", None
        elif message_type == "normal":
            self.normal_operation(random_dataframe)
            # self.retraining_needed = retraining_flag
            return None, None
        elif message_type == "check_state":
            retraining_flag = self.check_model_state()
            return retraining_flag, None
        elif message_type == "reset_state":
            self.state_estimation.reset(
                self.datasets["random_dataset"].random_dataloader()
            )
            return None, None
        elif message_type == "prep_retraining":
            # TODO: should return model checkpoint rather than
            # getting it elsewhere
            self.prepare_for_retraining(random_dataframe)
            return None, self.active_trainer.model.chkpt_name
        elif message_type == "retraining":
            checkpoint = self.retraining(query_dataframe, random_dataframe)
            # self.retraining_needed = retraining_flag
            return None, checkpoint

    def run(self) -> None:
        """Function used to run datacenter server in listening mode

        For communicating between edge and datacenter submodules
        the asyncio package is used to send messages asyncronously.
        The datacenter starts a server that listens on localhost for
        messages from the edge. The server uses a callback function to
        process any messages it recieves
        """

        self.comm_server = AsyncServer(callback=self.serve)
        asyncio.run(self.comm_server.start_server())
