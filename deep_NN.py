# %%


from tensorflow import keras
from tensorflow.keras import layers


def build_model_NN(units,input_dim,output_size):
    inputs = keras.Input(shape=(input_dim,), name="features")
    x1 = layers.Dense(units, activation="relu")(inputs)
    x2 = layers.Dense(units, activation="relu")(x1)
    outputs = layers.Dense(output_size, activation="softmax",name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_model_LSTM(units,input_dim,output_size):

    lstm_layer = layers.RNN(
        layers.LSTMCell(units), input_shape=(None, input_dim)
    )
    model = keras.models.Sequential(
        [
            lstm_layer,
            layers.BatchNormalization(),
            layers.Dense(output_size),
        ]
    )
    return model


def build_model_RNN(units,input_dim,output_size):

    RNN_layer = layers.RNN(
        layers.SimpleRNNCell(units), input_shape=(None, input_dim)
    )
    model = keras.models.Sequential(
        [
            RNN_layer,
            layers.BatchNormalization(),
            layers.Dense(output_size),
        ]
    )
    return model

