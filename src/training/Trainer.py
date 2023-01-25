from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import Config
from src.helpers.Timer import Timer
from src.helpers.general_helpers import create_path_if_not_exists
from src.training.Metrics import Metrics
from src.models.pretraining_model import PretrainingModel

import torch
from typing import Callable
import logging
import os
from collections import defaultdict
from six.moves import cPickle as pickle
from comet_ml import Experiment
from accelerate import Accelerator
import math

logging.basicConfig()

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TrainingException(Exception):
    pass


class EvalException(Exception):
    pass


class Trainer:

    def __init__(self, config: Config, model: PretrainingModel, optimizer, scheduler,
                 train_data_loader, val_data_loader, comet_ml_experiment, accelerator: Accelerator = None,
                 model_input_extractor=None,
                 target_extractor=None, loss_calculator=None, path=None):
        self.config = config
        self.model: PretrainingModel = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data_loader
        self.val_data = val_data_loader
        self.comet_ml_experiment: Experiment = comet_ml_experiment
        self.get_model_inputs = model_input_extractor
        self.get_targets = target_extractor
        self.get_losses = loss_calculator
        self.model_name = None
        self.accelerator = accelerator
        self.metrics = self.init_metrics()
        self.path = path

    def init_metrics(self):
        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            objective_keys = self.model.module.objectives.keys()
        else:
            objective_keys = self.model.objectives.keys()
        return Metrics(self.config, objective_keys)

    def set_model_input_extractor(self, func: Callable):
        self.get_model_inputs = func

    def set_target_extractor(self, func: Callable):
        self.get_targets = func

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
        # todo add further necessary attributes
        return True

    def log_interval_accuracies(self, accuracies, interval):
        if self.comet_ml_experiment:
            for key in accuracies.keys():
                self.comet_ml_experiment.log_metric(f"interval_acc_{key}", accuracies.get(key), step=interval)

    def log_interval_metrices(self, metrices, interval):
        pass

    def log_best_accuracies(self, best_accuracies, best_accuracies_idx):
        if self.comet_ml_experiment:
            for key in best_accuracies.keys():
                self.comet_ml_experiment.log_metric(f"best_acc_{key}", best_accuracies.get(key))
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
        print("\n")

    def update_history(self, history, train_loss, val_loss, train_accs, val_accs, combined_train_acc=None,
                       combined_val_acc=None):
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        for key in train_accs.keys():
            history[f"train_acc_{key}"].append(train_accs.get(key))
        for key in val_accs.keys():
            history[f"val_acc_{key}"].append(val_accs.get(key))

        if combined_train_acc:
            history["combined_train_acc"].append(combined_train_acc)
        if combined_val_acc:
            history["combined_val_acc"].append(combined_val_acc)

        return history

    def update_save_best_accuracy(self, accuracies, best_accuracies, best_idx, save_path, iteration):
        for key in accuracies.keys():
            if accuracies.get(key) > best_accuracies.get(key):
                best_accuracies[key] = accuracies.get(key)
                if self.model_name:
                    self.save_model_state_dict(save_path, f"{self.model_name}_best_model.bin")
                else:
                    self.save_model_state_dict(save_path, f"best_model_{key}.bin")
                best_idx[key] = iteration
        return best_accuracies, best_idx

    def save_model_state_dict(self, path, model_name):
        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        output = os.path.join(path, model_name)
        print("Save state dict to %s" % output)
        torch.save(state_dict, output)

    def create_training_checkpoint(self, completed_steps, completed_epochs):
        #print(f"Creating checkpoint", flush=True)
        base_checkpoint_path = os.path.join(self.config.OUTPUTDIR, "training_checkpoint")
        create_path_if_not_exists(base_checkpoint_path)

        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), os.path.join(base_checkpoint_path, "latest_model"))

        optimizer_dict = {
            "completed_epochs": completed_epochs,
            "completed_steps": completed_steps,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if self.config.NUM_GPUS:
            optimizer_dict["scaler"] = self.accelerator.scaler.state_dict()
        if self.accelerator.is_main_process:
            self.accelerator.save(optimizer_dict, os.path.join(base_checkpoint_path, "optimization"))

    def process_batch(self, batch, iteration, interval_length, acc_nominators, acc_denominators, losses, grad_clipping,
                      grad_clip_norm):
        with self.accelerator.accumulate(self.model):
            if isinstance(self.get_model_inputs, dict):
                objective = batch['objective'][0]
                model_inputs = self.get_model_inputs[objective](batch)
            else:
                model_inputs = self.get_model_inputs(batch)
            targets = model_inputs.get("labels")

            model_outputs = self.model(**model_inputs)

            # cumulatively calculate accuracy
            current_nominators, current_denominators = self.metrics.get_accuracy_values(model_outputs, targets)
            # preparing for gathering
            current_nominators = {key: torch.tensor(value).to(self.config.DEVICE) for key, value in
                                  current_nominators.items()}
            current_denominators = {key: torch.tensor(value).to(self.config.DEVICE) for key, value in
                                    current_denominators.items()}
            # gathering results from all GPUs
            current_nominators, current_denominators = self.accelerator.gather_for_metrics(
                (current_nominators, current_denominators))

            # processing gathered results
            current_nominators = {key: torch.sum(value).item() for key, value in current_nominators.items()}
            current_denominators = {key: torch.sum(value).item() for key, value in current_denominators.items()}

            acc_nominators, acc_denominators = self.metrics.cumulative_accuracy(acc_nominators, acc_denominators,
                                                                                current_nominators,
                                                                                current_denominators)

            loss = model_outputs.get("loss")

            loss_value = torch.tensor([loss.item()]).to(self.config.DEVICE)
            loss_values = self.accelerator.gather_for_metrics(loss_value).cpu()
            average_batch_loss = torch.mean(loss_values, dim=0, keepdim=True)
            losses = torch.cat((losses, average_batch_loss))
            if len(losses) == 100:
                losses = torch.mean(losses, dim=0, keepdim=True)

            if self.accelerator.is_main_process and self.comet_ml_experiment and (
                    iteration % 50 == 0 or iteration % interval_length == (interval_length - 1)):
                with self.comet_ml_experiment.train():
                    self.comet_ml_experiment.log_metric("batch_loss", round(average_batch_loss.item(), 3),
                                                        step=iteration)

            self.accelerator.backward(loss)
            if grad_clipping:
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    # For example max_norm=1.0
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return acc_nominators, acc_denominators, losses

    def train_interval(self, train_iterator, interval_length, current_iteration, total_iterations,
                       with_grad_clipping=True, grad_clip_norm=1.0):
        if current_iteration == 0:
            if not self.is_ready():
                raise TrainingException(
                    "Training could not be initiated. Not all necessary Trainer attributes are set.")

        self.model.train()

        losses = torch.tensor([])

        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            items = self.model.module.objectives.items()
        else:
            items = self.model.objectives.items()

        acc_nominators = {key: 0 for key, value in items}
        acc_denominators = {key: 0 for key, value in items}

        iteration = current_iteration

        batch_timer = Timer()
        batch_timer.start()

        while iteration < total_iterations:
            try:
                batch = next(train_iterator)
            except StopIteration:
                if self.accelerator.is_main_process:
                    print(
                        f"\nFinished Epoch {iteration / len(self.train_data)} / {self.config.determine_epochs(len(self.train_data))}")
                    print(
                        "------------------------------------------------------------------------------------------\n")

                self.create_training_checkpoint(iteration, int(iteration / len(self.train_data)))

                self.train_data.dataset.epoch_number += 1
                train_iterator = iter(self.train_data)
                batch = next(train_iterator)

            acc_nominators, acc_denominators, losses = self.process_batch(batch, iteration, interval_length,
                                                                          acc_nominators, acc_denominators, losses,
                                                                          with_grad_clipping, grad_clip_norm)
            if iteration % 500 == 0 and self.accelerator.is_main_process:
                print(f"Completed batch {iteration} / {total_iterations - 1}")

            if iteration % 100 == 0 and iteration != 0:
                batch_timer.stop()
                elapsed_minutes = batch_timer.get_elapsed_minutes()
                if self.accelerator.is_main_process and self.comet_ml_experiment:
                    self.comet_ml_experiment.log_metric("100_batches_time", elapsed_minutes, step=iteration)
                batch_timer.reset()
                batch_timer.start()

            iteration += 1

            if self.config.TRIAL_RUN and iteration == total_iterations:
                break

            if iteration % interval_length == 0:
                break

        batch_timer.stop()

        accuracies = self.metrics.calc_accuracies(acc_nominators, acc_denominators)
        mean_loss = torch.mean(losses)

        return accuracies, mean_loss, train_iterator, iteration


    def validate(self):
        if not self.is_ready():
            raise EvalException("Evaluation could not be initiated. Not all necessary Trainer attributes are set")

        self.model.eval()
        losses = torch.tensor([])
        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            items = self.model.module.objectives.items()
        else:
            items = self.model.objectives.items()
        acc_nominators = {key: 0 for key, value in items}
        acc_denominators = {key: 0 for key, value in items}
        num_batches = len(self.val_data)

        if self.config.TRIAL_RUN:
            trial_batches = self.config.get_trial_batch_num(num_batches)
            if self.accelerator.is_main_process:
                print(f"Trial run. Evaluating {trial_batches} batches.")
            num_batches = trial_batches

        i = 0

        with torch.no_grad():
            for d in self.val_data:
                if isinstance(self.get_model_inputs, dict):
                    objective = d['objective'][0]
                    model_inputs = self.get_model_inputs[objective](d)
                else:
                    model_inputs = self.get_model_inputs(d)

                targets = model_inputs.get("labels")

                model_outputs = self.model(**model_inputs)

                # gather all results from all processes in case of multi GPUs
                # predictions, targets = self.accelerator.gather_for_metrics((predictions, targets))

                current_nominators, current_denominators = self.metrics.get_accuracy_values(model_outputs, targets)

                # preparing for gathering
                current_nominators = {key: torch.tensor(value).to(self.config.DEVICE) for key, value in
                                      current_nominators.items()}
                current_denominators = {key: torch.tensor(value).to(self.config.DEVICE) for key, value in
                                        current_denominators.items()}
                # gathering results from all GPUs
                current_nominators, current_denominators = self.accelerator.gather_for_metrics(
                    (current_nominators, current_denominators))

                # processing gathered results
                current_nominators = {key: torch.sum(value).item() for key, value in current_nominators.items()}
                current_denominators = {key: torch.sum(value).item() for key, value in current_denominators.items()}

                acc_nominators, acc_denominators = self.metrics.cumulative_accuracy(acc_nominators, acc_denominators,
                                                                                    current_nominators,
                                                                                    current_denominators)

                loss = model_outputs.get("loss")

                loss_value = torch.tensor([loss.item()]).to(self.config.DEVICE)
                loss_values = self.accelerator.gather_for_metrics(loss_value).cpu()
                average_batch_loss = torch.mean(loss_values, dim=0, keepdim=True)
                losses = torch.cat((losses, average_batch_loss))
                if len(losses) == 100:
                    losses = torch.mean(losses, dim=0, keepdim=True)

                if i % 1000 == 0:
                    if self.accelerator.is_main_process:
                        print(f"Completed batch {i} / {num_batches}")

                i += 1

                if self.config.TRIAL_RUN:
                    if i == trial_batches:
                        break

        accuracies = self.metrics.calc_accuracies(acc_nominators, acc_denominators)

        return accuracies, torch.mean(losses)

    def train(self, with_grad_clipping=True, grad_clip_norm=1.0, model_name=None):
        total_iterations = self.config.determine_train_steps(len(self.train_data))
        interval_length = self.config.determine_interval_length(len(self.train_data))

        model_path = self.path if self.path else self.config.MODEL_PATH
        create_path_if_not_exists(model_path)
        self.model_name = model_name

        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            objectives = [key for key in self.model.module.objectives.keys()]
        else:
            objectives = [key for key in self.model.objectives.keys()]

        history = defaultdict(list)

        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            best_accuracies = {key: 0 for key in self.model.module.objectives.keys()}
        else:
            best_accuracies = {key: 0 for key in self.model.objectives.keys()}
        if len(objectives) > 1:
            best_accuracies["combined"] = 0
        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            best_accuracies_idx = {key: 0 for key in self.model.module.objectives.keys()}
        else:
            best_accuracies_idx = {key: 0 for key in self.model.objectives.keys()}
        if len(objectives) > 1:
            best_accuracies_idx["combined"] = 0

        train_iterator = iter(self.train_data)
        iteration = 0

        interval = 0

        interval_timer = Timer()
        interval_times = []
        training_timer = Timer()
        training_timer.start()
        validate_timer = Timer()

        if self.config.TRIAL_RUN:
            trial_steps, trial_intervals = self.config.get_trial_step_and_interval(total_iterations, interval_length)
            total_iterations = trial_steps
            interval_length = trial_intervals
            if self.accelerator.is_main_process:
                print(f"\nTRIAL RUN")

        if self.accelerator.is_main_process:
            if self.comet_ml_experiment:
                self.comet_ml_experiment.log_parameter("total_steps", total_iterations)
                self.comet_ml_experiment.log_parameter("interval_len", interval_length)
            print(f"Training for {total_iterations} steps with validations every {interval_length} steps.\n")
            print(f"Epoch length: {self.config.EPOCH_LEN}\n")

        while iteration < total_iterations:
            if self.accelerator.is_main_process:
                print(f"\n TRAINING INTERVAL {interval} / {math.ceil(total_iterations / interval_length) - 1}")
                print(
                    f"Steps {iteration} - {iteration + interval_length - 1 if iteration + interval_length - 1 <= total_iterations - 1 else total_iterations}")
                print("-----------------------------------------------------\n")

            interval_timer.reset()
            interval_timer.start()

            train_accuracies, train_loss, train_iterator, iteration = self.train_interval(train_iterator,
                                                                                          interval_length,
                                                                                          iteration, total_iterations,
                                                                                          with_grad_clipping,
                                                                                          grad_clip_norm)

            if self.accelerator.is_main_process:
                if self.comet_ml_experiment:
                    with self.comet_ml_experiment.train():
                        self.comet_ml_experiment.log_metric("interval_loss", train_loss, step=interval)
                        self.log_interval_accuracies(train_accuracies, interval)
                self.print_loss_acc(train_loss, train_accuracies, train=True)

            if len(objectives) > 1:
                combined_train_acc = self.metrics.calc_combined_accuracies(train_accuracies)
                if self.accelerator.is_main_process and self.comet_ml_experiment:
                    with self.comet_ml_experiment.train():
                        self.comet_ml_experiment.log_metric("combined_acc", combined_train_acc, step=interval)

            validate_timer.start()
            val_accuracies, val_loss = self.validate()
            validate_timer.stop()

            if self.accelerator.is_main_process:
                validation_minutes = validate_timer.get_elapsed_minutes()
                if self.comet_ml_experiment:
                    self.comet_ml_experiment.log_metric("validation_time", validation_minutes, step=interval)
                print(f"\nValidation took {validation_minutes} minutes.\n")

            if self.accelerator.is_main_process:
                if self.comet_ml_experiment:
                    with self.comet_ml_experiment.validate():
                        self.comet_ml_experiment.log_metric("interval_loss", val_loss, step=interval)
                        self.log_interval_accuracies(val_accuracies, interval)
                self.print_loss_acc(val_loss, val_accuracies, train=False)

            if len(objectives) > 1:
                combined_train_acc = self.metrics.calc_combined_accuracies(train_accuracies)
                combined_val_acc = self.metrics.calc_combined_accuracies(val_accuracies)
                if self.accelerator.is_main_process and self.comet_ml_experiment:
                    with self.comet_ml_experiment.validate():
                        self.comet_ml_experiment.log_metric("combined_acc", combined_val_acc, step=interval)
            else:
                combined_val_acc = None
                combined_train_acc = None

            history = self.update_history(history, train_loss, val_loss, train_accuracies, val_accuracies,
                                          combined_train_acc, combined_val_acc)

            if len(objectives) > 1:
                val_accuracies["combined"] = combined_val_acc

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                best_accuracies, best_accuracies_idx = self.update_save_best_accuracy(val_accuracies, best_accuracies,
                                                                                      best_accuracies_idx, model_path,
                                                                                      iteration)

            interval_timer.stop()
            interval_minutes = interval_timer.get_elapsed_minutes()
            interval_times.append(interval_minutes)
            if self.accelerator.is_main_process and self.comet_ml_experiment:
                self.comet_ml_experiment.log_metric("interval_time", interval_minutes, step=interval)
                print(f"\nTime for interval {interval}:")
                interval_timer.print_elapsed()
            interval_timer.start()

            interval += 1

        self.accelerator.wait_for_everyone()
        interval_timer.stop()
        training_timer.stop()
        interval_times = torch.tensor(interval_times)
        avg_interval_duration = round(torch.mean(interval_times).item(), 3)
        if self.accelerator.is_main_process:
            if self.comet_ml_experiment:
                self.comet_ml_experiment.log_metric("avg_interval_duration", avg_interval_duration)
                self.comet_ml_experiment.log_metric("total_time", training_timer.get_elapsed_minutes())
            print(f"\nTotal training time:")
            training_timer.print_elapsed()

        if self.accelerator.is_main_process:
            if self.model_name:
                self.save_model_state_dict(model_path, f"{self.model_name}_final_model.bin")
            else:
                self.save_model_state_dict(model_path, "final_model.bin")

            self.log_best_accuracies(best_accuracies, best_accuracies_idx)

            for key in best_accuracies_idx.keys():
                print(f"Best model for {key} - step: {best_accuracies_idx.get(key)}")

            if self.comet_ml_experiment:
                self.comet_ml_experiment.end()

            history_path = os.path.join(self.config.OUTPUTDIR, "reports")
            create_path_if_not_exists(history_path)
            with open(os.path.join(history_path, "history.pkl"), "wb") as f:
                pickle.dump(history, f)
        # load again with: with open(file, "rb") as f: history = pickle.load(f)

        del self.model

class IRTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.test_loader = kwargs.pop('test_loader')
        self.run_number = kwargs.pop('run_number')
        self.eval_output = kwargs.pop('eval_output')
        super().__init__(*args, **kwargs)

    def train(self, with_grad_clipping=True, grad_clip_norm=1.0, model_name=None):
        super().train(with_grad_clipping, grad_clip_norm, model_name)

    def predict(self, data):
        if not self.is_ready():
            raise EvalException("Evaluation could not be initiated. Not all necessary Trainer attributes are set")

        self.model.eval()
        losses = torch.tensor([])
        if self.config.NUM_GPUS and self.config.NUM_GPUS >= 1:
            items = self.model.module.objectives.items()
        else:
            items = self.model.objectives.items()
        acc_nominators = {key: 0 for key, value in items}
        acc_denominators = {key: 0 for key, value in items}
        num_batches = len(data)

        if self.config.TRIAL_RUN:
            trial_batches = self.config.get_trial_batch_num(num_batches)
            if self.accelerator.is_main_process:
                print(f"Trial run. Evaluating {trial_batches} batches.")
            num_batches = trial_batches

        i = 0

        predictions = []
        with torch.no_grad():
            for d in data:
                if isinstance(self.get_model_inputs, dict):
                    objective = d['objective'][0]
                    model_inputs = self.get_model_inputs[objective](d)
                else:
                    model_inputs = self.get_model_inputs(d)

                targets = model_inputs.get("labels")

                model_outputs = self.model(**model_inputs)
                predictions += torch.argmax(model_outputs['IR'], dim=1).tolist()

                # gather all results from all processes in case of multi GPUs
                # predictions, targets = self.accelerator.gather_for_metrics((predictions, targets))

                current_nominators, current_denominators = self.metrics.get_accuracy_values(model_outputs, targets)

                # preparing for gathering
                current_nominators = {key: torch.tensor(value).to(self.config.DEVICE) for key, value in
                                      current_nominators.items()}
                current_denominators = {key: torch.tensor(value).to(self.config.DEVICE) for key, value in
                                        current_denominators.items()}
                # gathering results from all GPUs
                current_nominators, current_denominators = self.accelerator.gather_for_metrics(
                    (current_nominators, current_denominators))

                # processing gathered results
                current_nominators = {key: torch.sum(value).item() for key, value in current_nominators.items()}
                current_denominators = {key: torch.sum(value).item() for key, value in current_denominators.items()}

                acc_nominators, acc_denominators = self.metrics.cumulative_accuracy(acc_nominators, acc_denominators,
                                                                                    current_nominators,
                                                                                    current_denominators)

                loss = model_outputs.get("loss")

                loss_value = torch.tensor([loss.item()]).to(self.config.DEVICE)
                loss_values = self.accelerator.gather_for_metrics(loss_value).cpu()
                average_batch_loss = torch.mean(loss_values, dim=0, keepdim=True)
                losses = torch.cat((losses, average_batch_loss))
                if len(losses) == 100:
                    losses = torch.mean(losses, dim=0, keepdim=True)

                if i % 1000 == 0:
                    if self.accelerator.is_main_process:
                        print(f"Completed batch {i} / {num_batches}")

                i += 1

                if self.config.TRIAL_RUN:
                    if i == trial_batches:
                        break

        accuracies = self.metrics.calc_accuracies(acc_nominators, acc_denominators)

        return predictions, accuracies, torch.mean(losses)

    "Evaluates the model using the test_loader and creates the trec file."
    def evaluate(self):
        predictions, accuracy, loss = self.predict(self.test_loader)
        predictions = np.array(predictions)
        # convert predictions to trec format
        df = self.test_loader.dataset.data.to_pandas()
        result = {k: [] for k in ['Query_Id', 'Post_Id', 'Rank', 'Score']}
        for question_id, data in df.groupby('question_id'):
            idx = data.index.tolist()
            preds = predictions[idx]

            result['Query_Id'] += len(data) * [question_id]

            sorted_indices = np.argsort(preds)[::-1]

            for i, index in enumerate(sorted_indices):
                result['Post_Id'].append(data['answer_id'][idx[index]])
                result['Rank'].append(i + 1)
                result['Score'].append(preds[index])

        result['Run_number'] = len(result['Score']) * [self.run_number]
        result_df = pd.DataFrame.from_dict(result)

        result_df.to_csv(self.eval_output, sep='\t')



