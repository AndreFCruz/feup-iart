# coding: utf-8

# In[1]:
from keras.models import Sequential
from keras.layers import Dense, Activation

# In[2]:

import numpy as np
import os

np.random.seed(42)

BASE_PATH = '../'
DATA_PATH = BASE_PATH + 'dataset'
MODELS_DIR = BASE_PATH + "models"

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

# In[99]:

# Save Model
import os

def save_model(model, name, acc=None):
    name += str(model.input.shape[1])
    for layer in model.layers:
        name += "-" + str(layer.output.shape[1])
    
    name += "_" + (("%.2f" % acc) if acc is not None else "")
    path = os.path.join(MODELS_DIR, name + ".h5")
    model.save(path)

# In[6]:

train_set, test_set = split_train_dataset(pulsars, 0.2)

# In[7]:

X_train, Y_train = train_set[:, :-1], train_set[:, -1]
X_test, Y_test = test_set[:, :-1], test_set[:, -1]

# In[185]:

# Create Model
model = Sequential()

input_dimension = np.size(X_train, axis=1)
model.add(Dense(16, input_dim=input_dimension, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# In[186]:

# Compile Model
from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[187]:

# Fit the Model
model.fit(X_train, Y_train, epochs=50, batch_size=16)

# In[188]:

# Evaluate the Model
scores = model.evaluate(X_test, Y_test)

# In[189]:

print("Accuracy:", scores[1] * 100)

# In[190]:

save_model(model, 'pulsar', scores[1] * 100)