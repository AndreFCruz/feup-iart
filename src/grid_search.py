from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pickle


def grid_search_params(create_model, param_grid, X, Y, epochs=50):
    # SKLearn wrapper for Keras classifier
    model = KerasClassifier(build_fn=create_model, epochs=epochs, verbose=2)
    # Setup Grid
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    # Train Models
    grid_results = grid.fit(X, Y)

    # Show Results
    print_grid_search_results(grid_results)

    # Save Best Model
    model_name = 'grid_search_best_' + ('%.2f' % (grid.best_score_ * 100)) + '.h5'
    model_path = '../models' + model_name
    grid.best_estimator_.model.save(model_path)
    return grid.best_estimator_


def print_grid_search_results(grid_results):
    print("Best: %f using %s" % (grid_results.best_score_, grid_results.best_params_))
    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    from utils import load_pulsar_csv
    pulsars = load_pulsar_csv()
    X, Y = pulsars[:, :-1], pulsars[:, -1]

    from Logger import Logger
    logger = Logger('../logs/grid_search.txt')

    from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

    # Parameter Grid for Searching
    param_grid = {
        'input_dim': [8],
        'first_neurons': [32, 16, 8, 4, 2],
        'second_neurons': [16, 8, 4, 2],
        'optimizer': [SGD(), RMSprop(), Adadelta(), Adam(), Adamax(), Nadam()]
    }

    from models import create_model_grid_search
    best_model = grid_search_params(create_model_grid_search, param_grid, X, Y, epochs=75)

    from utils import evaluate_model
    evaluate_model(best_model.model, X, Y)

    logger.close()