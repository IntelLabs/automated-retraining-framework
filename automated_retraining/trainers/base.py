# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import os
from csv import DictWriter
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from automated_retraining.models.base_model import BaseModel


class BaseTrainer:
    """Class used to facilitate the training loop in PyTorch."""

    def __init__(
        self,
        model: BaseModel,
        training_config: SimpleNamespace,
        train_loader: Optional[Callable] = None,
        val_loader: Optional[Callable] = None,
        log_interval: int = 10,
        **kwargs,
    ) -> None:
        """Sets up additional training parameters such as logging directories for model
        checkpoints, early stopping parameters, and milestones.

        Args:
            model (BaseModel): Input model, typically ActiveModel or BaseModel
            training_config (SimpleNamespace): Config file parameters
            log_interval (int, optional): Log interval for milestones. Defaults to 10.
            train_loader (Callable): When called, sets up the training dataloader. Defaults to None
            val_loader (Optional[Callable]): When called, sets up the val dataloader. Defaults to None.
        """
        self.chkpt_prefix = "base_trainer"
        self.model: BaseModel = model
        self.train_loader = train_loader
        if training_config.with_validation:
            self.val_loader = val_loader
        self.training_config: SimpleNamespace = training_config
        self.training_config.log_interval = log_interval
        if training_config.device == "cuda":
            self.model.cuda()
        if not hasattr(self.training_config, "logger"):
            self.training_config.logger = "csvfile"
        self.__configure_log_dir()
        self.__configure_early_stopping()
        self.__configure_save_best_model()
        self.__configure_milestones()

    def __configure_log_dir(self) -> None:
        """Configure directory to save results to"""
        if not self.training_config.log_dir:
            self.training_config.log_dir = "./results/"
        if not os.path.exists(self.training_config.log_dir):
            os.makedirs(self.training_config.log_dir)

    def _reset_early_stopping(self) -> None:
        """Reset metrics used in early stopping."""
        if "accuracy" in self.training_config.stopping_metric:
            self.best_stopping_metric = 0.0
        elif "loss" in self.training_config.stopping_metric:
            self.best_stopping_metric = 1e6
        self.epochs_without_improvement = 0

    def __configure_early_stopping(self) -> None:
        """Configured metrics to check for early
        stopping, if enabled.
        """
        if not hasattr(self.training_config, "early_stopping"):
            self.training_config.early_stopping = False
        if self.training_config.early_stopping:
            # for early stopping
            if self.training_config.stopping_metric not in self.get_log_keys():
                print(
                    f"Invalid Early Stopping Metric: {self.training_config.stopping_metric}"
                )
                if self.training_config.with_validation:
                    self.training_config.stopping_metric = "validation/epoch_loss"
                else:
                    self.training_config.stopping_metric = "train/epoch_loss"
                # self.training_config.patience = 5
                print(
                    f"Using defaults '{self.training_config.stopping_metric}', and patience={self.training_config.patience}"
                )
            self._reset_early_stopping()

    def __configure_save_best_model(self) -> None:
        """Configure metrics for selecting best model
        to save, if enabled.
        """
        if not hasattr(self.training_config, "save_best"):
            self.training_config.save_best = True
        if self.training_config.save_best:
            if not hasattr(self.training_config, "saving_metric"):
                if hasattr(self.training_config, "stopping_metric"):
                    self.training_config.saving_metric = (
                        self.training_config.stopping_metric
                    )
                    print(
                        f"No saving metric entered. Using stopping metric: '{self.training_config.saving_metric}'"
                    )
                else:
                    if self.training_config.with_validation:
                        self.training_config.saving_metric = "validation/epoch_loss"
                    else:
                        self.training_config.saving_metric = "train/epoch_loss"
                    print(
                        f"No saving metric entered. Using default: '{self.training_config.saving_metric}'"
                    )
            if self.training_config.saving_metric not in self.get_log_keys():
                print(f"Invalid Saving Metric: {self.training_config.saving_metric}")
                if self.training_config.with_validation:
                    self.training_config.stopping_metric = "validation/epoch_loss"
                else:
                    self.training_config.stopping_metric = "train/epoch_loss"
                self.best_saving_metric = 1e6
                print(f"Using defaults '{self.training_config.saving_metric}'")
            if "accuracy" in self.training_config.saving_metric:
                self.best_saving_metric = 0.0
            elif "loss" in self.training_config.saving_metric:
                self.best_saving_metric = 1e6

    def __configure_milestones(self) -> None:
        """Configure model saving milestones"""
        if not hasattr(self.training_config, "use_milestones"):
            self.training_config.use_milestones = False
        if self.training_config.use_milestones:
            if not hasattr(self.training_config, "log_interval"):
                self.training_config.log_interval = 10
                print(
                    f"No log interval entered. Using defaults '{self.training_config.log_interval}'"
                )
            self.training_config.milestones = set(
                range(
                    0,
                    self.training_config.max_epochs,
                    self.training_config.log_interval,
                )
            ) | {self.training_config.max_epochs}

    def __get__logger(
        self, fieldnames: List[str], logger: str = "csvfile"
    ) -> Union[SummaryWriter, DictWriter]:
        """Gets the selected logger object, either tensorboard or
           csv file. If csv file is selected will write the header
           to the csv file.

        Args:
            fieldnames (List[str]): Names of fields that will be logged.
                Used to write the header of the csv file.
            logger (str, optional): Which logger to use, tensorboard or csvfile.
                Defaults to "csvfile".

        Returns:
            Union[SummaryWriter, DictWriter]: returns the selected writer object.
        """
        if logger == "tensorboard":
            writer = SummaryWriter(self.training_config.log_dir, flush_secs=60 * 1)
        else:
            log_file = os.path.join(self.training_config.log_dir, "log.csv")
            writer = DictWriter
            with open(log_file, "w") as csvfile:
                csv_writer = writer(csvfile, fieldnames=fieldnames)
                csv_writer.writeheader()

        return writer

    def __close_logger(self, writer: Union[SummaryWriter, DictWriter]) -> None:
        """Closes the logger object if necessary. Currently
           only used for tensorboard logger.

        Args:
            writer (Union[SummaryWriter, DictWriter]): writer
                object to close
        """
        if isinstance(writer, SummaryWriter):
            writer.flush()
            writer.close()

    def __log__scalar(
        self,
        writer: Union[SummaryWriter, DictWriter],
        epoch: int,
        scalar_dict: Dict[str, float],
    ) -> None:
        """Logs metrics to logger of choice.

        Args:
            writer (SummaryWriter): Logger used, default tensorboard
            epoch (int): Current epoch
            scalar_dict (Dict[str, float]): Dictionary containing metrics to be logged
        """
        if isinstance(writer, SummaryWriter):
            for key in scalar_dict.keys():
                scalar = scalar_dict[key]
                if isinstance(scalar, list):
                    for idx, value in enumerate(scalar):
                        writer.add_scalar(key + "_{}".format(idx), value, epoch)
                else:
                    writer.add_scalar(key, scalar, epoch)
        elif writer is DictWriter:
            log_file = os.path.join(self.training_config.log_dir, "log.csv")
            log_keys = list(scalar_dict.keys())
            scalar_dict["epoch"] = epoch
            fieldnames = ["epoch"] + log_keys
            with open(log_file, "a") as csvfile:
                csv_writer = writer(csvfile, fieldnames=fieldnames)
                csv_writer.writerow(scalar_dict)

    def get_log_keys(self) -> List[str]:
        """Gathers list of metrics to log.

        Returns:
            List[str]: List of metrics (keys) to log.
        """
        log_keys = [
            "train/epoch_loss",
            "train/epoch_accuracy",
            "validation/epoch_loss",
            "validation/epoch_accuracy",
        ]
        if not self.training_config.with_validation:
            log_keys = [log_key for log_key in log_keys if "val" not in log_key]

        return log_keys

    def train_model(self) -> None:
        """Main training loop"""
        log_keys = self.get_log_keys()
        fieldnames = ["epoch"] + log_keys
        writer = self.__get__logger(
            fieldnames=fieldnames, logger=self.training_config.logger
        )
        log_dict: dict = dict.fromkeys(log_keys, None)
        self.history: dict = {}
        for key in log_keys:
            self.history[key] = []
        # train
        for epoch in range(0, self.training_config.max_epochs):
            train_loss, train_accuracy = self.train_one_epoch(epoch)
            log_values: List[Tuple[str, float]] = [
                ("train/epoch_loss", train_loss),
                ("train/epoch_accuracy", train_accuracy),
            ]
            if self.training_config.with_validation:
                val_loss, val_accuracy = self.validation_one_epoch(epoch)
                log_values.extend(
                    [
                        ("validation/epoch_loss", val_loss),
                        ("validation/epoch_accuracy", val_accuracy),
                    ]
                )
            self.model.scheduler.step()
            # Log Data
            log_dict = dict(log_values)
            [self.history[key].append(log_dict[key]) for key in log_keys]
            self.__log__scalar(writer, epoch, log_dict)
            stop_early = self.__epoch__end(epoch, log_dict)
            if stop_early:
                break
        if self.model.calibration is not None:
            logits, labels = self.model.get_logits_and_labels(
                self.train_loader(), self.training_config.device
            )
            self.model.calibration.reset(logits, labels)
        self.__close_logger(writer)

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """One epoch of training

        Args:
            epoch (int): Epoch number

        Returns:
            Tuple[float, float]: Epoch loss and epoch accuracy
        """
        self.model.train()
        epoch_loss: float = 0.0
        epoch_accuracy: float = 0.0
        with tqdm(
            self.train_loader(), unit="batch", initial=1, mininterval=1e-3
        ) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Training Epoch {epoch}")
                loss, accuracy = self.mini_batch((data, target))
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                epoch_loss += loss.item()
                epoch_accuracy = accuracy + epoch_accuracy
                tepoch.set_postfix(
                    Batch_Loss=loss.item(),
                    Batch_Acc=accuracy,
                    Epoch_Avg_Loss=epoch_loss / tepoch.n,
                    Epoch_Avg_Acc=epoch_accuracy / tepoch.n,
                )
        epoch_loss /= len(self.train_loader())
        epoch_accuracy = epoch_accuracy / len(self.train_loader())
        return epoch_loss, epoch_accuracy

    def validation_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Validation on one epoch

        Args:
            epoch (int): Current epoch

        Returns:
            Tuple[float, float]: Epoch loss and accuracy in the validation step.
        """
        self.model.eval()
        epoch_loss: float = 0.0
        epoch_accuracy: float = 0.0
        with torch.no_grad():
            with tqdm(
                self.val_loader(), unit="batch", initial=1, mininterval=1e-3
            ) as tepoch:
                for data, target in tepoch:
                    tepoch.set_description(f"Validation Epoch {epoch}")
                    loss, accuracy = self.mini_batch((data, target))
                    epoch_loss += loss.item()
                    epoch_accuracy = accuracy + epoch_accuracy
                    tepoch.set_postfix(
                        Batch_Loss=loss.item(),
                        Batch_Acc=accuracy,
                        Epoch_Avg_Loss=epoch_loss / tepoch.n,
                        Epoch_Avg_Acc=epoch_accuracy / tepoch.n,
                    )
        epoch_loss /= len(self.val_loader())
        epoch_accuracy = epoch_accuracy / len(self.val_loader())
        return epoch_loss, epoch_accuracy

    def evaluate(
        self, dataloader: torch.utils.data.dataloader.DataLoader
    ) -> Tuple[float, float]:
        """Evaluate model performance using pre built dataloader.

        Args:
            dataloader (torch.utils.data.dataloader.DataLoader): Dataloader to use for
            evalutaion

        Returns:
            Tuple[float, float]: Evaluation loss and accuracy
        """
        self.model.eval()
        eval_loss = 0
        eval_accuracy = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                loss, accuracy = self.mini_batch(batch)
                eval_loss += loss.item()
                eval_accuracy = accuracy + eval_accuracy
        eval_loss /= len(dataloader)
        eval_accuracy = eval_accuracy / len(dataloader)
        return eval_loss, eval_accuracy

    def mini_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.tensor, float]:
        """Run a mini batch through the model and compute accuracy and loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch containing labels and data

        Returns:
            Tuple[torch.tensor, float]: Loss and accuracy
        """
        batch = [batch.to(self.training_config.device) for batch in batch]
        logits = self.model(batch)
        metrics_out = self.model.compute_metrics(logits, batch)
        if self.model.training:
            loss = self.model.loss(logits, batch)
            (accuracy,) = metrics_out
        else:
            (
                accuracy,
                loss,
            ) = metrics_out
        return loss, accuracy

    def save_model(self, save_path: str) -> None:
        """Save the model to a checkpoint file.

        Args:
            save_path (str): Path where model checkpoint will be saved.
        """
        torch.save(self.model.model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")

    def __epoch__end(self, epoch: int, log_dict: Dict[str, float]) -> bool:
        """Steps to take after an epoch. Typically validation and early stopping checks.

        Args:
            epoch (int): Current epoch
            log_dict (Dict[str, float]): Metrics being logged/tracked.

        Returns:
            bool: Early stopping boolean. Stop is true, continue is false.
        """
        self.__training__epoch__end(epoch, log_dict)

        if self.training_config.with_validation:
            self.__validation__epoch__end(epoch, log_dict)

        if self.training_config.save_best:
            self.__save__best(
                epoch, log_dict, metric=self.training_config.saving_metric
            )

        if self.training_config.early_stopping:
            stop_early = self.__early__stopping(
                epoch,
                log_dict,
                metric=self.training_config.stopping_metric,
                patience=self.training_config.patience,
            )
        else:
            stop_early = False
        return stop_early

    def __training__epoch__end(self, epoch: int, log_dict: Dict[str, float]) -> None:
        """Steps to take at the end of a training epoch. Typically saving milestones.

        Args:
            epoch (int): Current epoch
            log_dict (Dict[str, float]): Dictionary containing logged metrics
        """
        if self.training_config.use_milestones:
            if (epoch + 1) in self.training_config.milestones:
                file_path = os.path.join(
                    self.training_config.log_dir,
                    f"{self.chkpt_prefix}_{epoch + 1}.pt",
                )
                self.save_model(file_path)

    def __validation__epoch__end(self, epoch: int, log_dict: Dict[str, float]) -> None:
        """Steps to take at the end of a validation epoch.

        Args:
            epoch (int): Current epoch
            log_dict (Dict[str, float]): Metrics being logged
        """
        pass

    def __early__stopping(
        self,
        epoch: int,
        log_dict: Dict[str, float],
        metric: str = "validation/epoch_loss",
        patience: int = 5,
    ) -> bool:
        """Early stopping check to prevent over fitting.

        Args:
            epoch (int): Current epoch
            log_dict (Dict[str, float]): Metrics being logged
            metric (str, optional): Early stopping mteric. Defaults to "validation/epoch_loss".
            patience (int, optional): Early stopping patience. Defaults to 5.

        Returns:
            bool: Flag to specify if early stopping should occur or not.
        """
        metric_val = log_dict[metric]
        if metric_val < 0:
            # Likely issue with metric related to validation, and self.training_config.with_validation=False
            return False

        if "accuracy" in metric:
            if metric_val > self.best_stopping_metric:
                self.best_stopping_metric = metric_val
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        elif "loss" in metric:
            if metric_val < self.best_stopping_metric:
                self.best_stopping_metric = metric_val
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        else:
            return False

        if self.epochs_without_improvement > patience:
            print("Early Stopping Occurred")
            print(f"Using {metric}, best value was {self.best_stopping_metric}")
            file_path = os.path.join(
                self.training_config.log_dir, f"{self.chkpt_prefix}_final.pt"
            )
            if "ActiveTrainer" not in self.__class__.__name__:
                self.save_model(file_path)
            return True
        else:
            return False

    def __save__best(
        self,
        epoch: int,
        log_dict: Dict[str, float],
        metric: str = "validation/epoch_loss",
    ):
        """Save the best model checkpoint

        Args:
            epoch (int): Current epoch
            log_dict (Dict[str, float]): Metrics being logged
            metric (str, optional): Early stopping metric. Defaults to "validation/epoch_loss".
        """
        metric_val = log_dict[metric]
        file_path = os.path.join(
            self.training_config.log_dir, f"{self.chkpt_prefix}_best.pt"
        )
        if "accuracy" in metric:
            if metric_val > self.best_saving_metric:
                print(
                    f"New Highest Accuracy: {metric_val}. Best value was {self.best_saving_metric}."
                )
                self.best_saving_metric = metric_val
                self.save_model(file_path)
        elif "loss" in metric:
            if metric_val < self.best_saving_metric:
                print(
                    f"New Lowest Loss: {metric_val}. Best value was {self.best_saving_metric}."
                )
                self.best_saving_metric = metric_val
                self.save_model(file_path)
