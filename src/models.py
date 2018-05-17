from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Conv1D
from keras.layers.advanced_activations import LeakyReLU


def create_model(input_dim):
    input = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(input)
    x = Dense(16, activation='relu')(input)
    output = Dense(1, activation='sigmoid')(input)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
