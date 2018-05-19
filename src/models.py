from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Conv1D
from keras.layers.advanced_activations import LeakyReLU


def create_model(input_dim):
    input = Input(shape=(input_dim,))
    x = LeakyReLU(alpha=0.3)(input)
    x = Dense(8)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(8)(x)
    x = Activation('relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_model_grid_search(input_dim, first_neurons=8,
                            second_neurons=8, activate_input=True,
                            optimizer='adam'):
    input = Input(shape=(input_dim,))
    if activate_input:
        x = LeakyReLU(alpha=0.3)(input)
    x = Dense(first_neurons)(x if activate_input else input)
    x = LeakyReLU(alpha=0.3)(x)
    if second_neurons > 0:
        x = Dense(second_neurons)(x)
    x = Activation('relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model