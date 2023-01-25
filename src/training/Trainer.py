from src.config.config import Config
from src.helpers.timer import Timer
from src.helpers.general_helpers import print_ram_util_info, print_gpu_util_info, create_path_if_not_exists
from src.pretraining_methods.Objectives import Objectives
from src.training.training_methods import mlm, nsp, combinations
from src.tracking.figures import plot_batch_loss

import numpy as np
import torch
from torch import nn
from typing import Callable
import logging
import os
from collections import defaultdict
from six.moves import cPickle as pickle
from comet_ml import Experiment

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TrainingException(Exception):
    pass


class EvalException(Exception):
    pass


class Trainer:

    def __init__(self, config: Config, model: nn.Module, optimizer, scheduler,
                 train_data_loader, val_data_loader, comet_ml_experiment, model_input_extractor=None,
                 target_extractor=None, loss_calculator=None):
        self.config = config
        self.model = model
        self.timer = Timer()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data_loader
        self.val_data = val_data_loader
        self.comet_ml_experiment: Experiment = comet_ml_experiment
        self.get_model_inputs = model_input_extractor
        self.get_targets = target_extractor
        self.get_losses = loss_calculator
        self.batch_losses = []

    def set_model_input_extractor(self, func: Callable):
        self.get_model_inputs = func

    def set_target_extractor(self, func: Callable):
        self.get_targets = func

    def set_loss_calculator(self, func: Callable):
        self.get_losses = func

    def is_ready(self):
        if self.optimizer is None:
            _logger.error(f"No optimizer set for Trainer.")
            return False
        if self.scheduler is None:
            _logger.error(f"No scheduler set for Trainer.")
            return False
        if self.train_data is None:
            _logger.error("No training data loader set for Trainer.")
            return False
        if self.get_model_inputs is None:
            _logger.error("No model input extractor set for Trainer.")
            return False
        if self.get_targets is None:
            _logger.error("No target extractor set for Trainer.")
            return False
        if self.get_losses is None:
            _logger.error("No loss calculator set for Trainer.")
            return False
        # todo add further necessary attributes
        return True

    def get_acc_calculator(self, objective_name):
        if objective_name == Objectives.MLM.name:
            return mlm.get_correct_predictions
        elif objective_name == Objectives.NSP.name:
            return nsp.get_correct_predictions

    def cumulative_accuracy(self, acc_nominators, acc_denominators, model_outputs, targets):
        for key in acc_nominators.keys():
            acc_calculator = self.get_acc_calculator(key)
            acc_nominator, acc_denominator = acc_calculator(model_outputs, targets)
            acc_nominators[key] += acc_nominator
            acc_denominators[key] += acc_denominator
        return acc_nominators, acc_denominators

    def calc_accuracies(self, acc_nominators, acc_denominators):
        accuracies = {key: 0 for key in self.model.objectives.keys()}
        for key in self.model.objectives.keys():
            nominator = acc_nominators.get(key)
            denominator = acc_denominators.get(key)
            accuracies[key] = nominator / denominator
        return accuracies

    def log_epoch_accuracies(self, accuracies, epoch):
        for key in accuracies.keys():
            self.comet_ml_experiment.log_metric(f"epoch_acc_{key}", accuracies.get(key), step=epoch)

    def log_best_accuracies(self, best_accuracies, best_accuracies_idx):
        for key in best_accuracies.keys():
            self.comet_ml_experiment.log_metric(f"best_acc_{key}", best_accuracies.get("key"))
        for key in best_accuracies_idx.keys():
            self.comet_ml_experiment.log_metric(f"best_acc_{key}_idx", best_accuracies_idx.get(key))

    def print_loss_acc(self, loss, accuracies, train=True):
        if train:
            mode = "Training"
        else:
            mode = "Validation"
        print(f"\n{mode} Loss: {loss}")
        print(f"{mode} Accuracy:")
        for key in accuracies:
            print(f"{key}: {accuracies.get(key)}")

    def update_history(self, history, train_loss, val_loss, train_accs, val_accs):
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        for key in train_accs.keys():
            history[f"train_acc_{key}"].append(train_accs.get(key))
        for key in val_accs.keys():
            history[f"val_acc_{key}"].append(val_accs.get(key))

        combined_train_acc = sum([value for key, value in train_accs.items()]) / len(
            list(train_accs.keys()))
        history["combined_train_acc"].append(combined_train_acc)

        combined_val_acc = sum([value for key, value in val_accs.items()]) / len(list(val_accs.keys()))
        history["combined_val_acc"].append(combined_val_acc)

        return history, combined_train_acc, combined_val_acc

    def update_save_best_accuracy(self, accuracies, best_accuracies, best_idx, save_path, epoch):
        for key in accuracies.keys():
            if accuracies.get(key) > best_accuracies.get(key):
                best_accuracies[key] = accuracies.get(key)
                self.save_model_state_dict(save_path, f"best_model_{key}.bin")
                best_idx[key] = epoch
        return best_accuracies, best_idx

    def save_model_state_dict(self, path, model_name):
        if torch.cuda.device_count() > 1:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(path, model_name))

    def train_epoch(self, n_examples, with_grad_clipping=False, grad_clip_norm=1.0):
        if not self.is_ready():
            raise TrainingException("Training could not be initiated. Not all necessary Trainer attributes are set.")

        self.model = self.model.train()

        losses = []
        acc_nominators = {key: 0 for key, value in self.model.objectives.items()}
        acc_denominators = {key: 0 for key, value in self.model.objectives.items()}
        num_batches = self.config.get_total_batches(n_examples)

        if self.config.TRIAL_RUN:
            trial_batches = self.config.get_trial_batch_num(n_examples)
            print(f"Trial run. Training on {trial_batches} batches.")
            num_batches = trial_batches

        i = 0

        for d in self.train_data:
            model_inputs = self.get_model_inputs(d)
            targets = self.get_targets(d)

            model_outputs = self.model(**model_inputs)

            # cumulatively calculate accuracy
            acc_nominators, acc_denominators = self.cumulative_accuracy(acc_nominators, acc_denominators, model_outputs,
                                                                        targets)

            loss = self.get_losses(model_outputs, targets, self.config)
            losses.append(loss.item())

            self.comet_ml_experiment.log_metric("batch_loss", loss.item(), step=i)
            self.batch_losses.append(loss.item())
            if i % 10 == 0:
                figure = plot_batch_loss(self.batch_losses)
                self.comet_ml_experiment.log_figure("Batch losses", figure, overwrite=True)

            loss.backward()
            if with_grad_clipping:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)  # For example max_norm=1.0
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            i += 1

            # checking gradients with this code? https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/17

            if self.config.TRIAL_RUN:
                if i == trial_batches:
                    break

            if self.config.TRIAL_RUN and i % 10 == 0:
                print(f"Completed batch {i} / {num_batches - 1}")
            if not self.config.TRIAL_RUN and i % 100 == 0:
                print(f"Completed batch {i} / {num_batches - 1}")

            if i == 1:
                for i in range(torch.cuda.device_count()):
                    print_gpu_util_info(i)
                print_ram_util_info()

        accuracies = self.calc_accuracies(acc_nominators, acc_denominators)

        return accuracies, np.mean(losses)

    def eval_epoch(self, n_examples):
        if not self.is_ready():
            raise EvalException("Evaluation could not be initiated. Not all necessary Trainer attributes are set")

        self.model = self.model.eval()
        losses = []
        acc_nominators = {key: 0 for key, value in self.model.objectives.items()}
        acc_denominators = {key: 0 for key, value in self.model.objectives.items()}
        num_batches = self.config.get_total_batches(n_examples)

        if self.config.TRIAL_RUN:
            trial_batches = self.config.get_trial_batch_num(n_examples)
            print(f"Trial run. Evaluating {trial_batches} batches.")

        i = 0

        with torch.no_grad():
            for d in self.val_data:
                model_inputs = self.get_model_inputs(d)
                targets = self.get_targets(d)

                model_outputs = self.model(**model_inputs)

                acc_nominators, acc_denominators = self.cumulative_accuracy(acc_nominators, acc_denominators,
                                                                            model_outputs, targets)

                loss = self.get_losses(model_outputs, targets, self.config)
                losses.append(loss.item())

                i += 1

                if self.config.TRIAL_RUN:
                    if i == trial_batches:
                        break

                if self.config.TRIAL_RUN and i % 10 == 0:
                    print(f"Completed batch {i} / {num_batches}")
                if not self.config.TRIAL_RUN and i % 100 == 0:
                    print(f"Completed batch {i} / {num_batches}")

        accuracies = self.calc_accuracies(acc_nominators, acc_denominators)

        return accuracies, np.mean(losses)

    def train(self, n_train, n_eval, with_grad_clipping=False, grad_clip_norm=1.0):
        model_path = self.config.MODEL_PATH
        create_path_if_not_exists(model_path)

        self.comet_ml_experiment.log_parameter("num_epochs", self.config.EPOCHS)
        self.comet_ml_experiment.log_parameter("batch_size", self.config.BATCH_SIZE)

        # initialising history dict for tracking metrics locally
        history = defaultdict(list)
        best_accuracies = {key: 0 for key in self.model.objectives.keys()}
        best_accuracies["combined"] = 0
        best_accuracies_idx = {key: 0 for key in self.model.objectives.keys()}
        best_accuracies_idx["combined"] = 0

        self.timer.reset()
        epoch_timer = Timer()
        self.timer.start()

        with self.comet_ml_experiment.train():

            for epoch in range(self.config.EPOCHS):
                print(f"\nEpoch {epoch} / {self.config.EPOCHS - 1}")
                print("-" * 20)

                epoch_timer.reset()
                epoch_timer.start()

                #with self.comet_ml_experiment.train():
                train_accuracies, train_loss = self.train_epoch(n_train, with_grad_clipping, grad_clip_norm)

                self.comet_ml_experiment.log_metric("epoch_train_loss", train_loss, step=epoch)
                self.log_epoch_accuracies(train_accuracies, epoch)

                self.print_loss_acc(train_loss, train_accuracies, train=True)

                #with self.comet_ml_experiment.test():
                val_accuracies, val_loss = self.eval_epoch(n_eval)

                self.comet_ml_experiment.log_metric("epoch_val_loss", val_loss, step=epoch)
                self.log_epoch_accuracies(val_accuracies, epoch)

                self.print_loss_acc(val_loss, val_accuracies, train=False)

                history, combined_train_acc, combined_val_acc = self.update_history(history, train_loss, val_loss,
                                                                                    train_accuracies, val_accuracies)

                self.comet_ml_experiment.log_metric("combined_train_acc", combined_train_acc, step=epoch)
                self.comet_ml_experiment.log_metric("combined_val_acc", combined_val_acc, step=epoch)

                val_accuracies["combined"] = combined_val_acc
                best_accuracies, best_accuracies_idx = self.update_save_best_accuracy(val_accuracies, best_accuracies,
                                                                                      best_accuracies_idx, model_path,
                                                                                      epoch)

                epoch_timer.stop()
                print(f"\nTraining + Validation time of epoch {epoch:}")
                epoch_timer.print_elapsed()

        self.save_model_state_dict(model_path, "final_model.bin")

        self.log_best_accuracies(best_accuracies, best_accuracies_idx)

        self.timer.stop()
        print("\nTraining time:")
        self.timer.print_elapsed()
        print("\n")
        self.timer.reset()

        for key in best_accuracies_idx.keys():
            print(f"Best model for {key} - epoch index: {best_accuracies_idx.get(key)}")

        self.comet_ml_experiment.end()

        history_path = os.path.join(self.config.OUTPUTDIR, "reports")
        create_path_if_not_exists(history_path)
        with open(os.path.join(history_path, "history.pkl"), "wb") as f:
            pickle.dump(history, f)
        # load again with: with open(file, "rb") as f: history = pickle.load(f)
