# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import asyncio
import os
from typing import Tuple

import pandas as pd

from automated_retraining.datasets import QuerySet
from automated_retraining.submodules.simulator_utils import SimulatorModule
from automated_retraining.submodules.submodule_utils import SubModule
from automated_retraining.trainers import ActiveTrainer
from automated_retraining.utils.communication import AsyncClient


class EdgeModule(SubModule):
    def __init__(self, config: str, **kwargs) -> None:
        """Edge module used to facilitate model inference and querying.

        Args:
            config (str, optional): Config to use on the edge module.
            Defaults to "./configs/active_config.yaml".
        """
        self.__dict__.update(kwargs)
        self.datasets = {}
        self.parse_config(config)
        self.__setup()

    def __setup(self) -> None:
        """Setup the edge module"""
        self.configure_training()
        self.configure_datasets(["query_dataset", "random_dataset"])
        self.configure_model()
        self.load_model_from_checkpoint(
            self.model_config.architecture,
            self.dataset_config.num_classes,
            self.chkpt,
        )
        if self.active_params.state_estimation_host == "edge":
            self.load_state_estimation_utils(
                state_estimator=self.active_params.state_estimation_method
            )
        self.set_sampling_strategy(self.active_params.strategy)

        self.full_df = pd.DataFrame(
            columns=self.get_dataset(next(iter(self.datasets))).info_df_headers
        )
        self.training_config.dataset_name = next(
            iter(self.datasets.values())
        ).data_module.__class__.__name__
        self.active_trainer = ActiveTrainer(
            self.model,
            self.training_config,
            **vars(self.active_params),
        )

    def run_query(
        self, dataset_key: str = "query_dataset", with_update: bool = False
    ) -> Tuple[int, QuerySet]:
        """Run query on the QuerySet to get samples for model calibration checks.

        Args:
            dataset_key (str, optional): Dataset to query from. Defaults to "query_dataset".
            with_update (bool, optional): Update dataset or not. Defaults to False.

        Returns:
            Tuple[int, QuerySet]: Returns the data samples indices that were chosen and
            the queried dataset.
        """
        query_samples_idx, query_dataset = self.model.query(
            self.datasets[dataset_key], self.active_params.n_query
        )
        if with_update:
            self.datasets[dataset_key].update_query_set(query_samples_idx)
        return query_samples_idx, query_dataset

    def get_random_samples(self) -> pd.DataFrame:
        """Get at most 100 random samples from the full set of data seen.

        Returns:
            pd.DataFrame: Random samples in a dataframe.
        """
        return self.full_df.sample(
            n=min([100, len(self.full_df)]), replace=False, random_state=42
        )

    def normal_operation(self, idx: int, new_samples: pd.DataFrame) -> pd.DataFrame:
        """Edge module operating normally and sending randomly sampled data
        to datacenter

        During normal operation the edge will process incoming data, and
        will periodically send randomly sampled data to the datacenter.
        The datacenter will use state_estimation metrics to check if
        retraining of the edge model should be done. A True/False flag
        is returned to edge indicated whether or not to start retraining

        Args:
            idx (int): idx is an index used to select with
            folder of data to use at current round. Data folders
            are used to simulate data being received in real time
            by the edge.
            new_samples (pd.DataFrame): new samples from simulator

        Returns:
            pd.DataFrame: dataframe of randomly sampled data to be
            sent to the datacenter
        """
        # set query set to new samples
        self.set_info_df("query_dataset", loadfiles=new_samples)
        # concat query set to full_df
        self.full_df = self.concat_datasets("query_dataset")
        # save only last 1000 samples
        self.full_df = self.full_df.iloc[-1000:]

        loss, accuracy = self.active_trainer.evaluate(
            self.datasets["query_dataset"].query_dataloader()
        )
        print(f"Loss: {loss:.2f} \t Accuracy: {accuracy:0.2f}")
        random_samples = self.get_random_samples()
        self.set_info_df("random_dataset", info_df=random_samples)

        return random_samples

    def prepare_for_retraining(self, retraining_needed: bool = True) -> pd.DataFrame:
        """Prepares edge module to begin new round of retraining.

        Edge updates the query_dataset, so that in can actively query
        for samples to send to the datacenter. Random samples are
        collected and sent to the datacenter in order to select
        a model to begin retraining from.

        Args:
            retraining_needed (bool): Boolean flag sent indicating if
            retraining is necessary

        Returns:
            pd.DataFrame: dataframe of randomly sampled data to be
            sent to the datacenter
        """
        if retraining_needed:
            query_df = self.full_df.copy()
            self.set_info_df("query_dataset", info_df=query_df)

            random_samples = self.get_random_samples()
            return random_samples

    async def send_handshake(self, port=9999) -> str:
        """Initial handshake message to initiate communication
           between edge and datacenter.

        Args:
            port (int, optional): port on localhost to sent
            requests to datacenter. Should be set to same port
            being served on by datacenter. Defaults to 9999.

        Returns:
            str: "Hello" response message from server indicating
            it is ready to receive communications.
        """
        comm_client = AsyncClient(port=port)

        message = await comm_client.send_data("handshake")

        return message

    async def run(self, config) -> None:
        """Function used to run edge in communication mode
        with datacenter

        For communicating between edge and datacenter submodules
        the asyncio package is used to send messages asyncronously.
        The datacenter starts a server that listens on localhost for
        messages from the edge. The edge sends request messages
        containing data that is processed by the datacenter.
        """
        simulator = SimulatorModule(config=config, mode=self.mode)
        n_iter = self.active_params.n_iter
        curr_iter = 0
        retraining_needed = False
        while curr_iter < n_iter + 1:
            if retraining_needed == False:
                curr_iter += 1
                print("Running Iteration: {}".format(curr_iter))
                new_samples = simulator.retrieve_data(curr_iter)
                random_dataframe = self.normal_operation(curr_iter, new_samples)

                comm_client = AsyncClient()
                _, _ = await comm_client.send_data(
                    "normal", random_data=random_dataframe
                )

                if curr_iter == 1:
                    if self.active_params.state_estimation_host == "edge":
                        self.state_estimation.reset(
                            self.datasets["random_dataset"].random_dataloader()
                        )
                    else:
                        comm_client = AsyncClient()
                        _, _ = await comm_client.send_data("reset_state")

                if self.active_params.state_estimation_host == "edge":
                    retraining_needed = self.check_model_state()
                else:
                    comm_client = AsyncClient()
                    retraining_needed, _ = await comm_client.send_data("check_state")

                if retraining_needed == True:
                    """
                    Prepare for retraining - Data Center selects
                    model to retrain from
                    """
                    random_dataframe = self.prepare_for_retraining()

                    comm_client = AsyncClient()
                    _, checkpoint = await comm_client.send_data(
                        "prep_retraining", random_data=random_dataframe
                    )

                    self.load_model_from_checkpoint(
                        self.model_config.architecture,
                        self.dataset_config.num_classes,
                        checkpoint,
                    )
            elif retraining_needed == True:
                """
                Active Learning Phase - Edge queries data, sends to
                Data Center. Data Center performs retraining on
                queried data
                """

                _, query_dataset = self.run_query(with_update=True)
                query_dataframe = query_dataset.info_df
                random_dataframe = self.get_random_samples()

                comm_client = AsyncClient()
                _, checkpoint = await comm_client.send_data(
                    "retraining",
                    query_data=query_dataframe,
                    random_data=random_dataframe,
                )

                if self.active_params.state_estimation_host == "edge":
                    retraining_needed = self.check_model_state()
                else:
                    comm_client = AsyncClient()
                    retraining_needed, _ = await comm_client.send_data("check_state")

                self.load_model_from_checkpoint(
                    self.model_config.architecture,
                    self.dataset_config.num_classes,
                    checkpoint,
                )

                if self.active_params.state_estimation_host == "edge":
                    self.state_estimation.reset(
                        self.datasets["random_dataset"].random_dataloader()
                    )
                else:
                    comm_client = AsyncClient()
                    _, _ = await comm_client.send_data("reset_state")
