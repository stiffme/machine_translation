import tensorflow.keras as keras
import tensorflow as tf


def create_lstm(units):
    if tf.test.is_gpu_available(cuda_only=True):
        return keras.layers.CuDNNLSTM(units, return_sequences=True, return_state=True)
    else:
        return keras.layers.LSTM(units, return_sequences=True, return_state=True)


def create_blstm(units):
    return keras.layers.Bidirectional(create_lstm(units), merge_mode='concat')

