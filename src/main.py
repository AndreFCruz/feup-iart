import numpy as np
from utils import *
from models import create_model
from keras.callbacks import TensorBoard
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

np.random.seed(42)

MODEL_NAME = '2nd' + (sys.argv[0] if len(sys.argv) > 0 else generate_random_string(n=4))
LOGS_FILE = '../logs/' + MODEL_NAME + '.txt'

# Load data and split dataset
pulsars = load_pulsar_csv()
train_set, test_set = train_test_split(pulsars, 0.2)

X_train, Y_train = train_set[:, :-1], train_set[:, -1]
X_test, Y_test = test_set[:, :-1], test_set[:, -1]

# Create Model
model = create_model(np.size(X_train, axis=1))

# Train Model
tb_callback = TensorBoard(log_dir='../Graph/' + MODEL_NAME, histogram_freq=5, write_graph=True, write_images=True)

model.fit(X_train, Y_train, epochs=10, batch_size=16,
        validation_data=(X_test, Y_test),
        callbacks=[tb_callback]
        # class_weight={0: 0.5, 1: 1.0}
        )


# Evaluate Model
logger = Logger(LOGS_FILE)
scores = model.evaluate(X_test, Y_test)
print("Overall Accuracy: %.2f\n" % (scores[1] * 100))
print(evaluate_classwise(model, X_test, Y_test))
logger.close()

# Precision-Recall Curve
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
          average_precision)
        )
plt.show()

# K-Fold Cross-Validation
# score = cross_validation(
#                 model, pulsars[:,:-1], pulsars[:,-1], k=3,
#                 train_function=train_model, test_function=evaluate_model
#             )
# print("Cross Validated Score: %f" % score)

# print('Saving model...')
# save_model(model, MODEL_NAME, scores[1] * 100)
