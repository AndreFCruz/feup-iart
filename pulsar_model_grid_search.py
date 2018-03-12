
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Activation


# In[2]:


import numpy as np
import os

RAND_SEED = 42

np.random.seed(RAND_SEED)

DATA_PATH = 'dataset'


# In[3]:


from scipy.io import arff
import pandas as pd

def load_pulsar_csv(path = DATA_PATH):
    csv_path = os.path.join(path, 'HTRU_2.csv')
    return np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

def load_pulsar_arff(path = DATA_PATH):
    arff_path = os.path.join(path, 'HTRU_2.arff')
    return arff.loadarff(arff_path)


# In[4]:


pulsars = load_pulsar_csv()


# In[5]:


import numpy as np

def split_train_dataset(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(test_ratio * len(data))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices,:], data[test_indices,:]

# Use hash of identifier to decide if instance goes into train or test set


# In[6]:


# Save Model
import os

MODELS_DIR = "models"

def save_model(model, name, acc=None, suffix=""):
    name += str(model.input.shape[1])
    for layer in model.layers:
        name += "-" + str(layer.output.shape[1])
    
    name += "_" + suffix + "_"
    name += "_" + (("%.2f" % acc) if acc is not None else "")
    path = os.path.join(MODELS_DIR, name + ".h5")
    model.save(path)


# In[7]:


X, Y = pulsars[:, :-1], pulsars[:, -1]

train_set, test_set = split_train_dataset(pulsars, 0.2)


# In[8]:


X_train, Y_train = train_set[:, :-1], train_set[:, -1]
X_test, Y_test = test_set[:, :-1], test_set[:, -1]


# In[9]:


# Create Model
input_dimension = np.size(X_train, axis=1)

def create_model(optimizer='adam', loss='binary_crossentropy', first_layer=16, second_layer=8):
    model = Sequential()
    model.add(Dense(first_layer, input_dim=input_dimension, activation='relu'))
    if (second_layer > 1):
        model.add(Dense(second_layer, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


# In[10]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# create model
model = KerasClassifier(build_fn=create_model, epochs=150, verbose=0)

# Parameters for GridSearch
param_grid = [
#    {
#        'optimizer': ['rmsprop'],
#        'batch_size': [3, 5]
#    },
    {
        'optimizer': ['adam'],
        'batch_size': [15],
        'second_layer': [0, 8, 12],
        'first_layer': [8, 12, 16, 32]
    }
]

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)


# In[11]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


# In[14]:


model_path = os.path.join(MODELS_DIR, 'grid_search_8_8_' + ('%.2f' % (grid.best_score_ * 100)) + '.h5')
grid.best_estimator_.model.save(model_path)

