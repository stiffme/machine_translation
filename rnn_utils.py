import tensorflow.keras as keras
import tensorflow as tf


def lstm(units):
    if tf.test.is_gpu_available(cuda_only=True):
        return keras.layers.CuDNNLSTM(units, return_sequences=True, return_state=True)
    else:
        return keras.layers.LSTM(units, return_sequences=True, return_state=True)


def blstm(units):
    return keras.layers.Bidirectional(lstm(units), merge_mode='concat')

