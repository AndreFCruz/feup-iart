import numpy as np
from utils import *
from models import create_model

np.random.seed(42)

MODEL_NAME = '2nd' + generate_random_string(n=4)
LOGS_FILE = '../logs/' + MODEL_NAME + '.txt'

# Load data and split dataset
pulsars = load_pulsar_csv()
train_set, test_set = train_test_split(pulsars, 0.2)

X_train, Y_train = train_set[:, :-1], train_set[:, -1]
X_test, Y_test = test_set[:, :-1], test_set[:, -1]

# Create Model
model = create_model(np.size(X_train, axis=1))

# Train Model
model.fit(X_train, Y_train, epochs=10, batch_size=16,
        validation_data=(X_test, Y_test),
        # class_weight={0: 0.5, 1: 1.0}
        )

# Evaluate Model
logger = Logger(LOGS_FILE)
scores = model.evaluate(X_test, Y_test)
print("Overall Accuracy: %.2f\n" % (scores[1] * 100))
print(evaluate_classwise(model, X_test, Y_test))
logger.close()

# K-Fold Cross-Validation
score = cross_validation(
                model, pulsars[:,:-1], pulsars[:,-1], k=3,
                train_function=train_model, test_function=evaluate_model
            )
print("Cross Validated Score: %f" % score)

# print('Saving model...')
# save_model(model, MODEL_NAME, scores[1] * 100)
