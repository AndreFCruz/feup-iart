import numpy as np
from utils import *
from models import *

np.random.seed(42)

LOGS_FILE = '../logs/' + MODEL_NAME

## Load data and split dataset
pulsars = load_pulsar_csv()
train_set, test_set = train_test_split(pulsars, 0.2)

X_train, Y_train = train_set[:, :-1], train_set[:, -1]
X_test, Y_test = test_set[:, :-1], test_set[:, -1]



# Evaluate Model
logger = Logger(LOGS_FILE)
scores = model.evaluate(X_test, Y_test)
print("Overall Accuracy: %.2f\n" % (scores[1] * 100))
print(evaluate_classwise(model, X_test, Y_test))
logger.close()

print('Saving model...')
save_model(model, , scores[1] * 100)
model.save('models/' + MODEL_NAME + '_%.2f.h5' % (scores[1] * 100))
