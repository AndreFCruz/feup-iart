import sys, os, io
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from functools import reduce
import random
import string

BASE_PATH = '../'
DATA_PATH = BASE_PATH + 'dataset'
MODELS_DIR = BASE_PATH + "models"


def cross_validation(model, X, Y, k, train_function, test_function):
    """
    Performs k-fold cross-validation on a given model.
    Calls the passed train_model function and test_model
     for every data split.
    """
    kf = KFold(n_splits=k)
    scores = []
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        # TODO clear model weights before retraining on new set
        train_function(model, X_train, Y_train)
        score = test_function(model, X_test, Y_test)
        scores.append(score)

    # Return average of scores
    return reduce((lambda x, y: x + y), scores) / len(scores)

def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, epochs=10, batch_size=16)

def evaluate_model(model, X_test, Y_test):
    return model.evaluate(X_test, Y_test)[1]

def evaluate_classwise(model, X_test, Y_test):
    predictions = model.predict(X_test)
    Y_test_predictions = [int(y + 0.5) for y in predictions]
    return classification_report(Y_test, Y_test_predictions)

def train_test_split(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(test_ratio * len(data))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices,:], data[test_indices,:]

def load_pulsar_csv(path = DATA_PATH):
    csv_path = os.path.join(path, 'HTRU_2.csv')
    return np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

def save_model(model, name, acc=None):
    name += str(model.input.shape[1])
    for layer in model.layers:
        name += "-" + str(layer.output.shape[1])

    name += "_" + (('_%.2f' % acc) if acc is not None else "")
    path = os.path.join(MODELS_DIR, name + ".h5")
    model.save(path)

def generate_random_string(n=4):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

class Logger(object):
    def __init__(self, path):
        self.file = open(path, 'w')
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        self.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
    def close(self):
        sys.stdout = self.stdout
        self.file.close()
