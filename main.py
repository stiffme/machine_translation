from encoder import encoder
import numpy as np
from data_utils import generate_dataset, preprocess_dataset
import tensorflow.keras as keras
from decoder import decoder
import json

def main():
    m = 20000
    dataset, human, machine, inv_machine = generate_dataset(m)
    Tx = 30
    Ty = 10
    n_a = 64
    n_s = 32

    dictionary = {'human': human, 'machine':machine, 'inv_machine':inv_machine}
    with open('dictionary.json', 'w') as f:
        f.write(json.dumps(dictionary))


    X, Y, Xoh, Yoh = preprocess_dataset(dataset, human, machine, Tx, Ty)
    print("\n")
    print('machine_vocab is ' + str(len(machine)))
    print('human_vocab is ' + str(len(human)))
    print('X shape is ' + str(np.shape(X)))
    print('Y shape is ' + str(np.shape(Y)))
    print('Xoh shape is ' + str(np.shape(Xoh)))
    print('Yoh shape is ' + str(np.shape(Yoh)))
    # X (m, Tx)
    # Y (m, Ty)
    # Xoh (m, Tx, len(human_vocab)]
    # Yoh (m, Ty, len(machine_vocab)]
    model = create_model(Tx, Ty, n_a, n_s, human, machine)

    # initialize hidden and cell to zeros
    h0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))

    # Yoh [m, Ty, len(machin_vocab)
    # swap the axis
    Yoh = list(np.transpose(Yoh, [1, 0, 2]))
    print('new Yoh shape is ' + str(np.shape(Yoh)))

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit([Xoh, h0, c0], Yoh,
              batch_size=128,
              epochs=6,
              validation_split=0.1)

    # model.save('my_model.h5')
    model.save_weights('my_model_weight.h5')


def create_model(Tx, Ty, n_a, n_s, human, machine):
    x_input = keras.Input((Tx, len(human)))
    hidden = keras.Input((n_s,))
    cell = keras.Input((n_s,))

    # a [m, Tx, 2*n_a]
    a = encoder(n_a, x_input)
    print('a shape is ' + str(a.shape))

    # outputs [Ty, m, len(machine_vocab)]
    outputs = decoder(Tx, Ty, n_s, machine, a, hidden, cell)
    # outputs = AttensionDecoder(Tx, Ty, n_s, machine)(a, hidden, cell)
    return keras.Model(inputs=[x_input, hidden, cell], outputs=outputs)


if __name__ == '__main__':
    main()