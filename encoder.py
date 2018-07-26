
import tensorflow.keras as keras
from rnn_utils import create_blstm


def encoder(units, x_input):
    bLSTM = create_blstm(units)
    output_seq, _, _, _, _ = bLSTM(x_input)
    return output_seq
#
# class AttensionEncoder(keras.Model):
#     """
#     Encoder of the attension model
#     Input (m, Tx,),
#     """
#
#     def __init__(self, units):
#         super(AttensionEncoder, self).__init__()
#         self.blstm = blstm(units)
#
#     def call(self, x_input):
#         # x_input [m, Tx, len(human)]
#
#         # x [m, Tx, units]
#         output_seq, _, _, _, _ = self.blstm(x_input)
#         return output_seq

