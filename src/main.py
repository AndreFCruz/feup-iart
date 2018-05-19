import numpy as np
from utils import *
from models import create_model
from keras.callbacks import TensorBoard, EarlyStopping
from tensorboard import summary as summary_lib
import tensorflow as tf
from PRTensorBoard import PRTensorBoard
from metrics import plot_precision_recall_curve

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
        # PRTensorBoard(log_dir=('../Graph/' + MODEL_NAME), write_images=True),
        EarlyStopping(monitor='val_acc', patience=15)
        ]
# callbacks = [ TensorBoard(log_dir='../Graph/' + MODEL_NAME, histogram_freq=5, write_graph=True, write_images=True) ]

### NOTE trying to balance class weights
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_train),
                                                 Y_train)
print("Class weight: ", class_weight)
###

# Train Model
model.fit(X_train, Y_train, epochs=100, batch_size=16,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
        class_weight={0: 1, 1: 5}
        )


# Evaluate Model
logger = Logger(LOGS_FILE)
scores = model.evaluate(X_test, Y_test)
print("Overall Accuracy: %.2f\n" % (scores[1] * 100))
print(evaluate_classwise(model, X_test, Y_test))
logger.close()


# Randomly trying new things
plot_precision_recall_curve(model, X_test, Y_test, 'pr_curve.png')
## TODO


# K-Fold Cross-Validation
# score = cross_validation(
#                 model, pulsars[:,:-1], pulsars[:,-1], k=3,
#                 train_function=train_model, test_function=evaluate_model
#             )
# print("Cross Validated Score: %f" % score)

# print('Saving model...')
# save_model(model, MODEL_NAME, scores[1] * 100)
