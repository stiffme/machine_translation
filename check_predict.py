from main import create_model
import json
from data_utils import string_to_tensor, tensor_to_string
import tensorflow.keras as keras
import numpy as np

def main():
    Tx = 30
    Ty = 10
    n_a = 64
    n_s = 32

    with open('dictionary.json', 'r') as f:
        data = json.loads(f.read())

    human = data['human']
    machine = data['machine']
    inv_machine = data['inv_machine']

    model = create_model(Tx, Ty, n_a, n_s, human, machine)
    model.load_weights('my_model_weight.h5')

    while True:
        input_string = input('Enter a date\n')
        # input_string = '4 Apr 1983'
        x = string_to_tensor(input_string, Tx, human)
        xoh = keras.utils.to_categorical(x, len(human))
        h0 = np.zeros((1, n_s))
        c0 = np.zeros((1, n_s))
        xoh = np.array([xoh])

        # output list(Ty) of [m, n_s]
        outputs = model.predict([xoh, h0, c0])

        # output [Ty, m, n_s]
        outputs = np.array(outputs)

        # output [m, Ty, n_s]
        outputs = np.transpose(outputs,(1,0,2))

        #yoh [Ty, n_s]
        yoh = outputs[0]

        # yoh [Ty,]
        yoh = np.argmax(yoh, axis=-1)

        y = ''.join(tensor_to_string(yoh, inv_machine))
        print(y)


if __name__ == '__main__':
    main()
