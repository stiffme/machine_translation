import tensorflow.keras as keras
from rnn_utils import create_lstm
import numpy as np

def decoder(Tx, Ty, units, machine_vocab, a, hidden, cell):
    lstm = create_lstm(units)
    dense = keras.layers.Dense(len(machine_vocab), activation='softmax')

    outputs = []
    for i in range(Ty):
        # context [m, 1, 2*n_a]
        context = attention_context(hidden, a, Tx)

        # the LSTM is based on context only
        # hidden , cell [m, n_s]
        _, hidden, cell = lstm(context, initial_state=[hidden, cell])

        # output [m, len(machine_vocab)]
        output = dense(hidden)

        outputs.append(output)
    # outputs list of [m, len(machine_vocab)]
    # [Ty, m, len(machine_vocab)]
    return outputs


def attention_context(s_prev, a, Tx):
    # for attention calculation
    denser1 = keras.layers.TimeDistributed(keras.layers.Dense(10, activation='tanh'))
    denser2 = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='relu'))
    softmax = keras.layers.Softmax(axis=1)
    dot = keras.layers.Dot(axes=1)
    repeatVector = keras.layers.RepeatVector(Tx)
    concator = keras.layers.Concatenate(axis=-1)
    # s_prev [m,Tx, n_s]
    s_prev = repeatVector(s_prev)

    # a [m, Tx, n_a]
    # s_a [m, Tx, n_s + n_a]
    s_a = concator([s_prev, a])

    # s_a [m, Tx, 10]
    s_a = denser1(s_a)

    # s_a [m, Tx, 1]
    s_a = denser2(s_a)

    # softm [m, Tx, 1]
    softm = softmax(s_a)

    # softm [m, Tx, 1] dot a [m, Tx, 2*n_a] = [m, 1, 2*n_a]
    context = dot([softm, a])
    return context

# class AttensionDecoder(keras.Model):
#     def __init__(self, Tx, Ty, units, machine_vocab):
#         super(AttensionDecoder, self).__init__()
#         self.lstm = create_lstm(units)
#         self.Tx = Tx
#         self.Ty = Ty
#         self.repeatVector = keras.layers.RepeatVector(Tx)
#         self.concator = keras.layers.Concatenate(axis=-1)
#         self.dense = keras.layers.Dense(len(machine_vocab), activation='softmax')
#
#         #for attention calculation
#         self.denser1 = keras.layers.TimeDistributed(keras.layers.Dense(10, activation='tanh'))
#         self.denser2 = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='relu'))
#         self.softmax = keras.layers.Softmax(axis=1)
#         self.dot = keras.layers.Dot(axes=1)
#
#
#
#     def call(self, a, hidden, cell):
#         outputs = []
#         for i in range(self.Ty):
#             # context [m, 1, 2*n_a]
#             context = self.attention_context(hidden, a)
#
#             # the LSTM is based on context only
#             # hidden , cell [m, n_s]
#             _, hidden, cell = self.lstm(context, initial_state=[hidden, cell])
#
#             # output [m, len(machine_vocab)]
#             output = self.dense(hidden)
#
#             outputs.append(output)
#         # outputs list of [m, len(machine_vocab)]
#         # [Ty, m, len(machine_vocab)]
#         return outputs
#
#
#     def attention_context(self, s_prev, a):
#         # s_prev [m,Tx, n_s]
#         s_prev = self.repeatVector(s_prev)
#
#         # a [m, Tx, n_a]
#         # s_a [m, Tx, n_s + n_a]
#         s_a = self.concator([s_prev, a])
#
#         # s_a [m, Tx, 10]
#         s_a = self.denser1(s_a)
#
#         # s_a [m, Tx, 1]
#         s_a = self.denser2(s_a)
#
#         # softm [m, Tx, 1]
#         softm = self.softmax(s_a)
#
#         # softm [m, Tx, 1] dot a [m, Tx, 2*n_a] = [m, 1, 2*n_a]
#         context = self.dot([softm, a])
#         return context