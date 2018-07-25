
import tensorflow.keras as keras
from rnn_utils import blstm


class AttensionEncoder(keras.Model):
    """
    Encoder of the attension model
    Input (m, Tx,),
    """

    def __init__(self, vocab_size, embedding_dim, units):
        super(AttensionEncoder, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.blstm = blstm(units)

    def call(self, x_input):
        # x_input [m, Tx]
        # x [m, Tx, embedding_dim]
        x = self.embedding(x_input)

        # x [m, Tx, units]
        output_seq, _, _ = self.blstm(x)
        return output_seq