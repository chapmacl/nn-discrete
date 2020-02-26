import datetime
import os
import json
import pickle
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import torch
from collections import defaultdict
from discrete_nn.settings import model_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device == "cuda:0":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.loss_funct = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            for batch_inx, (X, Y) in enumerate(dataset_generator):
                outputs = self(X)
                outputs.to(self.device)
                loss = self.loss_funct(outputs, Y)
                validation_losses.append(loss)
                predictions += torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1).tolist()
                targets += Y.tolist()
        return self._gen_stats(targets, predictions, validation_losses)

    def evaluate_and_save_to_disk(self, dataset, name):
        loss, acc, class_report_dict = self._evaluate(dataset)
        stats = defaultdict(list)
        stats["loss"].append(loss)
        stats["acc"].append(acc)
        stats["classification_report"].append(class_report_dict)
        self.save_to_disk(stats, name, False)

    @staticmethod
    def _gen_stats(targets, predictions, losses):
        """ generates basic training/evaluation information"""
        eval_loss = torch.mean(torch.stack(losses)).item()
        eval_acc = accuracy_score(targets, predictions)
        class_report_dict = classification_report(targets, predictions, output_dict=True)
        return eval_loss, eval_acc, class_report_dict

    def set_net_parameters(self, param_dict):
        raise NotImplementedError

    def get_net_parameters(self):
        raise NotImplementedError

    def _train_epoch(self, dataset_generator):
        self.train()
        batch_loss_train = []
        targets = []
        predictions = []
        # training part of epoch
        for batch_inx, (X, Y) in enumerate(dataset_generator):
            print(batch_inx)
            self.optimizer.zero_grad()  # reset gradients from previous iteration
            # do forward pass
            net_output = self(X)
            # compute loss
            loss = self.loss_funct(net_output, Y)
            # backward propagate loss
            loss.backward()
            self.optimizer.step()
            batch_loss_train.append(loss)
            predictions += torch.nn.functional.softmax(net_output, dim=1).argmax(dim=1).tolist()
            targets += Y.tolist()
        #print(f"training losses {batch_loss_train}")
        return self._gen_stats(targets, predictions, batch_loss_train)

    def save_to_disk(self, stats, name: str, save_model=True):
        """Saves model's pickled class as pickle, the training metrics and a copy of the weight parameters as a pickle
        to disk"""
        now = datetime.datetime.now()
        container_folder = os.path.join(model_path, self.__class__.__name__ + "-" + name + f"-{now.year}-{now.month}-{now.day}"
                                                                    f"--h{now.hour}m{now.minute}")
        os.mkdir(container_folder)

        with open(os.path.join(container_folder, "metrics.json"), "w") as f:
            json.dump(stats, f)
        if save_model:
            with open(os.path.join(container_folder, f"{self.__class__.__name__}.pickle"), "wb") as f:
                pickle.dump(self, f)

            with open(os.path.join(container_folder, f"{self.__class__.__name__}.param.pickle"), "wb") as f:
                pickle.dump(self.get_net_parameters(), f)

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
            # starting epochs evaluation
            validation_loss, validation_acc, validation_class_report = self._evaluate(validation_dataset)

            stats["training_loss"].append(training_loss)
            stats["training_acc"].append(training_acc)
            stats["training_classification_report"].append(training_class_report)
            stats["validation_loss"].append(validation_loss)
            stats["validation_acc"].append(validation_acc)
            stats["validation_classification_report"].append(validation_class_report)

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
