import datetime
import os
import json
from typing import Optional
import pickle
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from discrete_nn.settings import model_path
import gc


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.loss_funct = None

    def _epoch_eval_callback(self, validation_dataset: DataLoader) -> Optional[dict]:
        """
        method called by the train model method after the training step in each epoch. Allows custom subclasses
        to do additional evaluations and returns additional epoch stats. Used for logit models to include discretized
        evaluations
        :return: a dictionary with additional epoch metrics or None
        """
        return None

    def _model_testing_callback(self, testing_dataset: DataLoader):
        """
        method called by the train model method when training is done and testing the model with the test dataset.
        Allows custom subclasses to do additional evaluations and returns additional epoch stats.
        Used for logit models to include discretized evaluations
        :return: a dictionary with additional epoch metrics or None
        """
        return None

    def _evaluate(self, dataset_generator):
        """
        Evaluates a method using dataset_generator
        :param dataset_generator: a sample generator

        :return: evaluation_loss, evaluation_acc, classification_report_dict
        """
        self.eval()
        validation_losses = []
        targets = []
        predictions = []
        # disables gradient calculation since it is not needed
        with torch.no_grad():
            gc.collect()
            for batch_inx, (X, Y) in enumerate(dataset_generator):
                outputs = self(X)
                loss = self.loss_funct(outputs, Y)
                validation_losses.append(float(loss))
                predictions += torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1).tolist()
                targets += Y.tolist()
        return self._gen_stats(targets, predictions, validation_losses)

    def evaluate_model(self, dataset: DataLoader) -> Dict:
        """
        Evaluates the model with dataset
        :param dataset: the dataset to evaluate with
        :return:
        """
        loss, acc, class_report_dict = self._evaluate(dataset)
        stats = defaultdict(list)
        stats["loss"].append(loss)
        stats["acc"].append(acc)
        stats["classification_report"].append(class_report_dict)
        return stats

    def evaluate_and_save_to_disk(self, dataset, name):
        """given a dataset extending Pytorch's Dataset class, and a name for the folder where the results will be placed,
        evaluates the network."""
        stats = self.evaluate_model(dataset)
        self.save_to_disk(stats, name, False)

    @staticmethod
    def _gen_stats(targets, predictions, losses):
        """ generates basic training/evaluation information"""
        eval_loss = np.mean(losses)
        eval_acc = accuracy_score(targets, predictions)
        class_report_dict = classification_report(targets, predictions, output_dict=True)
        return eval_loss, eval_acc, class_report_dict

    def set_net_parameters(self, param_dict):
        raise NotImplementedError

    def get_net_parameters(self):
        raise NotImplementedError

    def _train_epoch(self, dataset_generator):
        self.train()
        self.zero_grad()
        batch_loss_train = []
        targets = []
        predictions = []
        # training part of epoch
        for batch_inx, (X, Y) in enumerate(dataset_generator):
            gc.collect()
            self.optimizer.zero_grad()  # reset gradients from previous iteration
            # do forward pass
            net_output = self(X)
            # compute loss
            loss = self.loss_funct(net_output, Y)
            # backward propagate loss
            loss.backward()
            self.optimizer.step()
            batch_loss_train.append(float(loss))
            predictions += torch.nn.functional.softmax(net_output, dim=1).argmax(dim=1).tolist()
            targets += Y.tolist()
        # print(f"training losses {batch_loss_train}")
        return self._gen_stats(targets, predictions, batch_loss_train)

    def save_to_disk(self, stats, name: str, save_model=True):
        """Saves model's pickled class as pickle, the training metrics and a copy of the weight parameters as a pickle
        to disk
        :returns path to the folder containing the model and its metrics"""
        now = datetime.datetime.now()
        container_folder = os.path.join(model_path, name + f"-{now.year}-{now.month}-{now.day}"
                                                                    f"--h{now.hour}m{now.minute}")

        os.mkdir(container_folder)

        with open(os.path.join(container_folder, "metrics.json"), "w") as f:
            json.dump(stats, f)
        if save_model:
            model = self

            with open(os.path.join(container_folder, f"{model.__class__.__name__}.pickle"), "wb") as f:
                torch.save(model, f)

            with open(os.path.join(container_folder, f"{model.__class__.__name__}.param.pickle"), "wb") as f:
                torch.save(model.get_net_parameters(), f)

        return container_folder

    def train_model(self, training_dataset, validation_dataset, test_dataset, epochs, model_name,
                    evaluate_before_train: bool = False):
        """
        Trains the model with a training dataset and uses the validation_dataset to _evaluate it at every epoch
        :param training_dataset: generator for training data
        :param validation_dataset: ... for validation
        :param test_dataset: ... for test
        :param epochs: number of epochs to train for
        :param model_name: a name for the model (important for saving to disk)
        :param evaluate_before_train: if set, model will be evaluate before training (useful in the case of a logit
        model initialized with real weights). The untrained model will be saved to disk
        :return: the path to the folder where metrics were saved
        """

        if evaluate_before_train:
            eval_stats = defaultdict(list)
            test_loss, test_acc, test_class_report = self._evaluate(test_dataset)
            eval_stats["test_loss"] = test_loss
            eval_stats["test_acc"] = test_acc
            eval_stats["test_classification_report"] = test_class_report
            eval_stats.update(self._model_testing_callback(test_dataset))
            self.save_to_disk(eval_stats, f"{model_name}-untrained", save_model=False)
        stats = defaultdict(list)

        for epoch_in in tqdm(range(epochs), desc="Training Network. Epoch:"):
            training_loss, training_acc, training_class_report = self._train_epoch(training_dataset)
            training_loss_post_update, training_acc_post_update, training_class_report_post_update = self._evaluate(
                training_dataset)
            # starting epochs evaluation
            validation_loss, validation_acc, validation_class_report = self._evaluate(validation_dataset)

            stats["training_loss"].append(training_loss)
            stats["training_acc"].append(training_acc)
            stats["training_classification_report"].append(training_class_report)

            stats["training_loss_post_update"].append(training_loss_post_update)
            stats["training_acc_post_update"].append(training_acc_post_update)
            stats["training_classification_report_post_update"].append(training_class_report_post_update)

            stats["validation_loss"].append(validation_loss)
            stats["validation_acc"].append(validation_acc)
            stats["validation_classification_report"].append(validation_class_report)

            # calls subclasses callback so they can add any metric the wish
            for metric_name, metric_value in self._epoch_eval_callback(validation_dataset).items():
                stats[metric_name].append(metric_value)

            print(f"epoch {epoch_in + 1}/{epochs}: "
                  f"train loss: {training_loss:.4f} / "
                  f"validation loss: {validation_loss:.4f} /"
                  f"validation acc: {validation_acc} /"
                  f"validation precision: {validation_class_report['weighted avg']['precision']} /"
                  f"validation recall: {validation_class_report['weighted avg']['recall']} /")

        test_loss, test_acc, test_class_report = self._evaluate(test_dataset)
        stats["test_loss"] = test_loss
        stats["test_acc"] = test_acc
        stats["test_classification_report"] = test_class_report
        stats.update(self._model_testing_callback(test_dataset))

        return self.save_to_disk(stats, f"{model_name}-trained")


class LogitModel(BaseModel):
    def __init__(self):
        super().__init__()

    def _epoch_eval_callback(self, validation_dataset : DataLoader) -> Optional[dict]:
        sample_stats = self.evaluate_discretized_from_logit_models("sample", validation_dataset, 10, None)
        argmax_stats = self.evaluate_discretized_from_logit_models("argmax", validation_dataset, 1, None)
        stats = dict()
        stats["validation_loss_discrete_sample"] = sample_stats["mean_loss"]
        stats["validation_acc_discrete_sample"] = sample_stats["mean_acc"]
        stats["validation_loss_discrete_argmax"] = argmax_stats["mean_loss"]
        stats["validation_acc_discrete_argmax"] = argmax_stats["mean_acc"]
        return stats

    def _model_testing_callback(self, testing_dataset: DataLoader):
        sample_stats = self.evaluate_discretized_from_logit_models("sample", testing_dataset, 10, None)
        argmax_stats = self.evaluate_discretized_from_logit_models("argmax", testing_dataset, 1, None)
        stats = dict()
        stats["test_loss_discrete_sample"] = sample_stats["mean_loss"]
        stats["test_acc_discrete_sample"] = sample_stats["mean_acc"]
        stats["test_loss_discrete_argmax"] = argmax_stats["mean_loss"]
        stats["test_acc_discrete_argmax"] = argmax_stats["mean_acc"]
        return stats


    def generate_discrete_networks(self, method: str) -> BaseModel:
        raise NotImplementedError

    def evaluate_discretized_from_logit_models(self, discretization_method: str, dataset: DataLoader,
                                               num_trials: int, result_save_path):
        """
        Given a logit model (such as a ternary one), generates a discrete one from it using the provided
        discretization method and evaluates it with dataset.

        :param model: the logit model
        :param discretization_method: e.g. sample
        :param dataset: the dataset to evaluate the discrete model with
        :param num_trials: the number of independent evaluations (discretizing the model again everytime)
        :param result_save_path: the path to save results to. If None does not save
        :return: a copy of the metrics dictionary
        """
        # gets device being used for dataset by looking into dataset used by dataloader
        device = dataset.dataset[0][0].device

        results = []
        for i in range(num_trials):
            # discretizes
            disc_model = model.generate_discrete_networks(discretization_method)
            disc_model = disc_model.to(device)
            stats = disc_model.evaluate_model(dataset)
            results.append(stats)
        mean_loss = sum([result["loss"][0] for result in results]) / num_trials
        mean_acc = sum([result["acc"][0] for result in results]) / num_trials

        stats = {"trials": results, "mean_loss": mean_loss, "mean_acc": mean_acc}
        if result_save_path is not None:
            if not os.path.exists(result_save_path):
                os.mkdir(result_save_path)
            with open(os.path.join(result_save_path, "metrics.json"), "w") as f:
                json.dump(stats, f)
        return stats


class AlternateDiscretizationBaseModel(BaseModel):
    """
    This class is the base implementation of generic models which apply a discretization step during training
    """

    def __init__(self):
        super().__init__()

    def set_net_parameters(self, param_dict):
        raise NotImplementedError

    def get_net_parameters(self):
        raise NotImplementedError

    def discretize(self):
        """Discretizes a model's weights. Model dependent"""
        raise NotImplementedError

    def train_model(self, training_dataset, validation_dataset, test_dataset, epochs, model_name,
                    evaluate_before_train: bool = False):
        """
        Trains the model with a training dataset and uses the validation_dataset to evaluate it at every epoch. Results
        are saved to disk.
        :param training_dataset: generator for training data
        :param validation_dataset: ... for validation
        :param test_dataset: ... for test
        :param epochs: number of epochs to train for
        :param model_name: a name for the model (important for saving to disk)
        :param evaluate_before_train: if set, model will be evaluate before training (useful in the case of a logit
        model initialized with real weights). The untrained model will be saved to disk
        :return:
        """
        if evaluate_before_train:
            eval_stats = defaultdict(list)
            test_loss, test_acc, test_class_report = self._evaluate(test_dataset)
            eval_stats["test_loss"] = test_loss
            eval_stats["test_acc"] = test_acc
            eval_stats["test_classification_report"] = test_class_report
            self.save_to_disk(eval_stats, f"{model_name}-untrained")
        stats = defaultdict(list)

        for epoch_in in tqdm(range(epochs), desc="Training Network. Epoch:"):
            training_loss, training_acc, training_class_report = self._train_epoch(training_dataset)
            # call discretization method
            with torch.no_grad():
                self.discretize()
            # starting epochs evaluation
            validation_loss, validation_acc, validation_class_report = self._evaluate(validation_dataset)
            training_loss_post_update, training_acc_post_update, training_class_report_post_update = self._evaluate(
                training_dataset)

            stats["training_loss"].append(training_loss)
            stats["training_acc"].append(training_acc)
            stats["training_classification_report"].append(training_class_report)

            stats["training_loss_post_discretize"].append(training_loss_post_update)
            stats["training_acc_post_discretize"].append(training_acc_post_update)
            stats["training_classification_report_post_discretize"].append(training_class_report_post_update)

            stats["validation_loss_post_discretize"].append(validation_loss)
            stats["validation_acc_post_discretize"].append(validation_acc)
            stats["validation_classification_report_post_discretize"].append(validation_class_report)

            print(f"epoch {epoch_in + 1}/{epochs}: "
                  f"train loss: {training_loss:.4f} / "
                  f"validation loss: {validation_loss:.4f} /"
                  f"validation acc: {validation_acc} /"
                  f"validation precision: {validation_class_report['weighted avg']['precision']} /"
                  f"validation recall: {validation_class_report['weighted avg']['recall']} /")

        test_loss, test_acc, test_class_report = self._evaluate(test_dataset)
        stats["test_loss"] = test_loss
        stats["test_acc"] = test_acc
        stats["test_classification_report"] = test_class_report
        self.save_to_disk(stats, f"{model_name}-trained")
