import sys
import numpy as np
from utils import cross_validate, train_test_split, generate_random_string, load_pulsar_csv, evaluate_classwise, evaluate_model
from Logger import Logger
from models import create_model
from keras.callbacks import TensorBoard, EarlyStopping
from tensorboard import summary as summary_lib
import tensorflow as tf
from PRTensorBoard import PRTensorBoard

np.random.seed(42)

MODEL_NAME = '2nd' + (sys.argv[1] if len(sys.argv) > 1 else generate_random_string(n=4))
LOGS_FILE = '../logs/' + MODEL_NAME + '.txt'

# Load data and split dataset
pulsars = load_pulsar_csv()
train_set, test_set = train_test_split(pulsars, 0.2)

X_train, Y_train = train_set[:, :-1], train_set[:, -1]
X_test, Y_test = test_set[:, :-1], test_set[:, -1]

# Create Model
model = create_model(np.size(X_train, axis=1))

# Train Model
callbacks = [
        PRTensorBoard(log_dir=('../Graph/' + MODEL_NAME), write_images=True),
        EarlyStopping(monitor='val_acc', patience=15)
        ]

### NOTE Balancing class weights
# from sklearn.utils import class_weight
# class_weight = class_weight.compute_class_weight('balanced',
#                                                  np.unique(Y_train),
#                                                  Y_train)
# print("Class weight: ", class_weight)
###

# Train Model
model.fit(X_train, Y_train, epochs=100, batch_size=16,
        validation_data=(X_test, Y_test),
        callbacks=callbacks
        )


## Evaluate Model
logger = Logger(LOGS_FILE)
scores = model.evaluate(X_test, Y_test)
print("Overall Accuracy: %.2f\n" % (scores[1] * 100))
print(evaluate_classwise(model, X_test, Y_test))
ret = evaluate_model(model, X_test, Y_test)
print("Classwise Results: ", ret)
logger.close()


## NOTE Undersampling!
# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(return_indices=True)
# X_resampled, y_resampled, idx_resampled = rus.fit_sample(pulsars[:,:-1], pulsars[:,-1])

## NOTE K-Fold Cross-Validation
# from utils import train_model, evaluate_model
# ret = cross_validate(model, pulsars[:,:-1], pulsars[:,-1], 10,
#         train_model,
#         evaluate_model)

# print("Result: ", ret)
## NOTE END

# print('Saving model...')
# save_model(model, MODEL_NAME, scores[1] * 100)
