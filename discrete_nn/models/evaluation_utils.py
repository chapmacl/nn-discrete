import json
import os

from torch.utils.data import DataLoader

from discrete_nn.models.base_model import LogitModel


def evaluate_discretized_from_logit_models(model: LogitModel, discretization_method: str, dataset: DataLoader,
                                           num_trials: int, result_save_path):
    """
    Given a logit model (such as a ternary one), generates a discrete one from it using the provided
    discretization method and evaluates it with dataset.

    :param model: the logit model
    :param discretization_method: e.g. sample
    :param dataset: the dataset to evaluate the discrete model with
    :param num_trials: the number of independent evaluations (discretizing the model again everytime)
    :param result_save_path: the path to save results to
    :return:
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
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    with open(os.path.join(result_save_path, "metrics.json"), "w") as f:
        json.dump(stats, f)
