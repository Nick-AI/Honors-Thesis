import matplotlib.pylab as plt
import json
import numpy as np


def plot_graphs(dataset, save=True):

    source_file = './Results/log_' + dataset + '.json'
    with open(source_file, "r") as f:
        d = json.load(f)

    train_accuracy = 100 * (np.array(d["train_loss"])[:, 1])
    test_accuracy = 100 * (np.array(d["test_loss"])[:, 1])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Accuracy')
    ax1.plot(train_accuracy, color="tomato", linewidth=2, label='train_acc')
    ax1.plot(test_accuracy, color="steelblue", linewidth=2, label='test_acc')
    ax1.legend(loc=0)

    train_loss = np.array(d["train_loss"])[:, 0]
    test_loss = np.array(d["test_loss"])[:, 0]

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(train_loss, '--', color="tomato", linewidth=2, label='train_loss')
    ax2.plot(test_loss, '--', color="steelblue", linewidth=2, label='test_loss')
    ax2.legend(loc=1)

    ax1.grid(True)

    if save:
        fig.savefig('./figures/' + dataset + '.svg')

    plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    for d_set in ['Reuters_TfIDf', 'Reuters_withEmbeddings', 'Reuters_simple_nn', 'SOData_TfIDf',
                  'SOData_withEmbeddings', 'SOData_simple_nn', 'AmazonRev_simple_nn', 'AmazonRev_withEmbeddings']:
        plot_graphs(d_set)
