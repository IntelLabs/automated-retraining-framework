# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import asyncio
import os

from automated_retraining.submodules.datacenter_utils import DataCenterModule
from automated_retraining.submodules.edge_utils import EdgeModule
from automated_retraining.submodules.simulator_utils import SimulatorModule
from automated_retraining.submodules.submodule_utils import SubModule


class ActiveLearning(SubModule):
    """Active learning submodule to facility the decoupled (edge/datacenter) active
    learning model.

    Args:
        SubModule: Base submodule to build new modules from.
    """

    def __init__(self, config: str, **kwargs) -> None:
        """Set up module used for active learning

        Args:
            config (str): Config to use in submodule.
        """
        self.__dict__.update(kwargs)
        self.config = config
        self.parse_config(config)

    def edge(self, mode) -> None:
        """Run the edge when edge and datacenter
        are run communicating via localhost.
        """
        edge = EdgeModule(
            config=self.config, chkpt=self.active_params.starting_chkpt, mode=mode
        )
        response, _ = asyncio.run(edge.send_handshake())
        if response == "Hello":
            print("Handshake received, beginning communication with datacenter")
            asyncio.run(edge.run(self.config))

    def datacenter(self, mode) -> None:
        """Run the datacenter when edge and datacenter
        are run communicating via localhost.
        """
        dc = DataCenterModule(
            config=self.config, chkpt=self.active_params.starting_chkpt, mode=mode
        )

        dc.run()

    def edge_datacenter_retraining(self, mode) -> None:
        """Active learning loop using decoupled edge and datacenter modules."""
        simulator = SimulatorModule(config=self.config, mode=mode)
        edge = EdgeModule(
            config=self.config, chkpt=self.active_params.starting_chkpt, mode=mode
        )
        dc = DataCenterModule(
            config=self.config, chkpt=self.active_params.starting_chkpt, mode=mode
        )

        state_estimation_host = (
            edge if edge.active_params.state_estimation_host == "edge" else dc
        )
        n_iter = self.active_params.n_iter
        curr_iter = 0
        retraining_needed = False
        while curr_iter < n_iter + 1:
            while not retraining_needed:
                print(f"\nRunning iteration {curr_iter}\n")
                curr_iter += 1
                new_samples = simulator.retrieve_data(curr_iter)
                random_dataframe = edge.normal_operation(curr_iter, new_samples)
                dc.normal_operation(random_dataframe)
                if curr_iter == 1:
                    state_estimation_host.state_estimation.reset(
                        state_estimation_host.datasets[
                            "random_dataset"
                        ].random_dataloader()
                    )
                retraining_needed = state_estimation_host.check_model_state()

            if retraining_needed:
                query_set = edge.prepare_for_retraining()
                dc.prepare_for_retraining(query_set)

            while retraining_needed and dc.al_iters_complete < dc.active_params.n_iter:
                _, query_dataset = edge.run_query(with_update=True)
                query_dataframe = query_dataset.info_df
                random_dataframe = edge.get_random_samples()
                al_chkpt = dc.retraining(query_dataframe, random_dataframe)
                edge.load_model_from_checkpoint(
                    edge.model_config.architecture,
                    edge.dataset_config.num_classes,
                    al_chkpt,
                )
                retraining_needed = state_estimation_host.check_model_state()
                print(f"Retraining needed: {retraining_needed}")

            state_estimation_host.state_estimation.reset(
                state_estimation_host.datasets["random_dataset"].random_dataloader()
            )

    def run(self):
        """Run active learning"""
        if self.mode == "demo":
            self.edge_datacenter_retraining(mode=self.mode)
        elif self.mode == "edge":
            self.edge(mode=self.mode)
        elif self.mode == "datacenter":
            self.datacenter(mode=self.mode)
