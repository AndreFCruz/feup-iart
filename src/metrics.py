import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


# Precision-Recall Curve
def plot_precision_recall_curve(model, X_test, Y_test, filepath=None):
    Y_predictions = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(Y_test, Y_predictions)

    average_precision = average_precision_score(Y_test, Y_predictions)

    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                    color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    
    if filepath is not None:
        plt.savefig(filepath)
    plt.show()

def plot_precision_recall(lines, fname):
    plt.clf()

    for precision, recall, label in lines:
        plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.savefig(fname)
