from __future__ import absolute_import, division, print_function
import tensorflow as tf
from encoder import AttensionEncoder
import numpy as np
from data_utils import generate_dataset, preprocess_dataset
import tensorflow.keras as keras


def main():
    tf.enable_eager_execution()
    dataset, human, machine, inv_machine = generate_dataset(11)
    Tx = 30
    Ty = 10
    n_a = 64
    n_s = 32
    embedding_dim = 12

    X, Y, _, _ = preprocess_dataset(dataset, human, machine, Tx, Ty)
    print('X shape is ' + str(np.shape(X)))
    print('Y shape is ' + str(np.shape(Y)))
    # X (m, Tx)
    # Y (m, Ty)
    dataset = tf.data.Dataset.from_tensor_slices()
    model = create_model(Tx, Ty, embedding_dim, n_a, n_s, human, machine)


def create_model(dataset, Tx, Ty, embedding_dim, n_a, n_s, human, machine):
    x_input = keras.Input(shape=(Tx,))

    x = AttensionEncoder(len(human),embedding_dim, n_a)(x_input)

    return keras.Model(inputs=x_input, outputs=x)



if __name__ == '__main__':
    main()