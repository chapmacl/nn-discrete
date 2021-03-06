import json
import matplotlib.pyplot as plt
import os
from typing import Optional

"""
with open('metrics.json') as f:
    d = json.load(f)

values = d["training_loss"]
values2 = d["validation_loss_post_discretize"]
report1 = d["training_classification_report"]
report2 = d["validation_classification_report_post_discretize"]

values.pop(0)
values2.pop(0)

fig = plt.figure()
plt.plot(values, label = "train")
plt.plot(values2, label = "validation")
title = 'MNIST Manual Train vs Validation Loss'
fig.suptitle(title, fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=16)
#plt.show()
plt.savefig(title + '.png')

values = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
values2 = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

for epoch in range(len(report1)):
    for num in range(10):
        values[str(num)].append(report1[epoch][str(num)]['f1-score'])
        values2[str(num)].append(report2[epoch][str(num)]['f1-score'])

for number, numberV in zip(values, values2):
    fig2 = plt.figure()
    plt.plot(values[number], label = "train")
    plt.plot(values2[numberV], label = "validation")
    title = str(number + " F1 score (Manual)")
    fig2.suptitle(title, fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('F1', fontsize=16)
    #plt.show()
    plt.savefig(title + '.png')

"""


def plot_acc(experiment_folder, plot_discrete_stats: bool):
    """

    :param experiment_folder:
    :param plot_discrete_stats: if True, plots the discrete stats
    :return:
    """
    # opening metrics file
    with open(os.path.join(experiment_folder, "metrics.json")) as f:
        metrics = json.load(f)

    train_acc = None
    val_acc_logit = None
    val_acc_disc = {}

    if "training_loss_post_discretize" in metrics:
        # its an alternate discretization experiment
        train_acc = metrics["training_acc_post_discretize"]
        val_acc_logit = metrics["validation_acc_post_discretize"]
    else:
        # other types of experiment
        train_acc = metrics["training_acc_post_update"]
        val_acc_logit = metrics["validation_acc"]
        if plot_discrete_stats is True:
            val_acc_disc["argmax"] = metrics["validation_acc_discrete_argmax"]
            val_acc_disc["sample"] = metrics["validation_acc_discrete_sample"]
        else:
            raise ValueError(f"unknown discrete metric {plot_discrete_stats}")
    fig, ax = plt.subplots()

    ax.plot(val_acc_logit, 'b', label="val. acc (logit)")
    ax.plot(train_acc, 'r', label="train acc (logit)")
    for key, stats in val_acc_disc.items():
        ax.plot(stats, label=f"val. acc discrete ({key})")
    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('accuracy', fontsize=16)

    ax.legend()
    plt_save_path = os.path.join(experiment_folder, "acc_plot.png")
    fig.savefig(plt_save_path)


def plot_loss(experiment_folder):
    # opening metrics file
    with open(os.path.join(experiment_folder, "metrics.json")) as f:
        metrics = json.load(f)

    if "training_loss_post_discretize" in metrics:
        # its an alternate discretization experiment
        train_loss = metrics["training_loss_post_discretize"]
        val_loss = metrics["validation_loss_post_discretize"]
    else:
        # other types of experiment
        train_loss = metrics["training_loss_post_update"]
        val_loss = metrics["validation_loss"]
    fig, ax = plt.subplots()
    ax.plot(train_loss, 'r', label="train loss")
    ax.plot(val_loss, 'b', label="val. loss")
    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('loss', fontsize=16)

    # plt.show()
    ax.legend()
    plt_save_path = os.path.join(experiment_folder, "loss_plot.png")
    fig.savefig(plt_save_path)
