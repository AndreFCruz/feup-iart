
# coding: utf-8

# In[1]:


import numpy as np
import os

np.random.seed(42)

BASE_PATH = '../'
DATA_PATH = BASE_PATH + 'dataset'
MODELS_DIR = BASE_PATH + "models"


# In[2]:


from scipy.io import arff

def load_pulsar_csv(path = DATA_PATH):
    csv_path = os.path.join(path, 'HTRU_2.csv')
    return np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

def load_pulsar_arff(path = DATA_PATH):
    arff_path = os.path.join(path, 'HTRU_2.arff')
    return arff.loadarff(arff_path)


# In[3]:


pulsars = load_pulsar_csv()


# In[4]:


import numpy as np

def split_train_dataset(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(test_ratio * len(data))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices,:], data[test_indices,:]

# Use hash of identifier to decide if instance goes into train or test set


# In[5]:


# Save Model
import os

def save_model(model, name, acc=None):
    name += str(model.input.shape[1])
    for layer in model.layers:
        name += "-" + str(layer.output.shape[1:])
    
    name += "_" + (("%.2f" % acc) if acc is not None else "")
    path = os.path.join(MODELS_DIR, name + ".h5")
    model.save(path)


# In[6]:


train_set, test_set = split_train_dataset(pulsars, 0.2)

X_train, Y_train = train_set[:, :-1], train_set[:, -1]
X_test, Y_test = test_set[:, :-1], test_set[:, -1]


# In[7]:


from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Reshape

# Create Model
input_dimension = np.size(X_train, axis=1)

def create_model_cnn():
    inputs = Input(shape=(input_dimension,), dtype='float32')
    reshape = Reshape((input_dimension,1))(inputs) # reshape input (?,8) to conv input (?,8,1)
    conv_0 = Conv1D(64, kernel_size=6, activation='relu')(reshape)
    pool = MaxPooling1D()(conv_0)
    flatten = Flatten()(pool)
    dropout = Dropout(0.2)(flatten)
    output = Dense(1, activation='sigmoid')(dropout)
    # Compile model
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[8]:


model = create_model_cnn()


# In[9]:


# Fit the Model
model.fit(X_train, Y_train, epochs=150, batch_size=15, validation_data=[X_test, Y_test])


# In[10]:


from sklearn.metrics import classification_report

# Classification metrics on test data
predictions = model.predict(X_test)
Y_test_predictions = [int(y + 0.5) for y in predictions]
print(classification_report(Y_test, Y_test_predictions))


# In[11]:


# Evaluate the Model
scores = model.evaluate(X_test, Y_test)
print("Accuracy:", scores[1] * 100)

save_model(model, 'pulsar_CNN_', scores[1] * 100)


# In[61]:


from random import random

oversampled_positive_examples = []
undersampled_negative_exampled = []

for i in range(len(X_train)):
    # oversampled positive examples
    if Y_train[i] == 1 and random() > 0.2:
        oversampled_positive_examples.append(i)
        
    # negative exampled that remain in train dataset (undersampled)
    elif Y_train[i] == 0 and random() > 0.4:
        undersampled_negative_exampled.append(i)        


# In[62]:


X_undersampled = X_train[undersampled_negative_exampled]
Y_undersampled = Y_train[undersampled_negative_exampled]

X_over_under = np.concatenate((X_undersampled, [X_train[i] for i in oversampled_positive_examples]))
Y_over_under = np.concatenate((Y_undersampled, [Y_train[i] for i in oversampled_positive_examples]))


# In[66]:


# One more run on oversampled and undersampled dataset
model.fit(X_over_under, Y_over_under, epochs=250, batch_size=30, validation_data=[X_test, Y_test])


# In[67]:


from sklearn.metrics import classification_report

# Classification metrics on test data
predictions = model.predict(X_test)
Y_test_predictions = [int(y + 0.5) for y in predictions]
print(classification_report(Y_test, Y_test_predictions))


# In[68]:


# Evaluate the Model
scores = model.evaluate(X_test, Y_test)
print("Accuracy:", scores[1] * 100)

save_model(model, 'pulsar_CNN_', scores[1] * 100)

