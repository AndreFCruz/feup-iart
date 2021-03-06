import sys, os, io
import numpy as np
from keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from functools import reduce
import random
import string

BASE_PATH = '../'
DATA_PATH = BASE_PATH + 'dataset'
MODELS_DIR = BASE_PATH + "models"


def cross_validate(model, X, Y, k, train_function, test_function):
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

        # train model
        train_function(model, X_train, Y_train)

        # test/evaluate model
        run_scores = test_function(model, X_test, Y_test)
        scores.append(run_scores)

        # clear model weights for next iteration
        reset_weights(model)

    # Sum scores
    avg_scores = {}
    for dic in scores:
        print(dic)
        for key in dic:
            if key not in avg_scores:
                avg_scores[key] = 0
            avg_scores[key] += dic[key]

    # Average scores
    for key in avg_scores:
        avg_scores[key] /= k

    return avg_scores
    

def avg(lst):
    return reduce((lambda x, y: x + y), lst) / len(lst)

def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=0)

def evaluate_model(model, X_test, Y_test):
    acc = model.evaluate(X_test, Y_test)[1]
    y_pred = [int(p + 0.5) for p in model.predict(X_test)]
    ret = precision_recall_fscore_support(Y_test, y_pred)

    return {
        'acc': acc,
        'precision_0': ret[0][0],
        'precision_1': ret[0][1],
        'recall_0': ret[1][0],
        'recall_1': ret[1][1],
        'f1_0': ret[2][0],
        'f1_1': ret[2][1],
        'support_0': ret[3][0],
        'support_1': ret[3][1]
    }

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

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
