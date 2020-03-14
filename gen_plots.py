import json
import matplotlib.pyplot as plt


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
