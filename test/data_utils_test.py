import unittest
from data_utils import *
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_generate_dataset(self):
        m = 10
        dataset, human, machine, inv_machine = generate_dataset(m)
        assert np.shape(dataset)[0] == m
        X, Y, Xoh, Yoh = preprocess_dataset(dataset, human, machine, 30, 30)
        inv_human = {v: k for k, v in human.items()}

        for i in range(m):
            dx = dataset[i][0]
            x = X[i]
            cx = ''.join(tensor_to_string(x, inv_human)[:len(dx)])
            assert cx == dx

        for i in range(m):
            dy = dataset[i][1]
            y = Y[i]
            cy = ''.join(tensor_to_string(y, inv_machine)[:len(dy)])
            assert cy == dy

    def test_string_to_tensor(self):
        vocab = dict(zip(['a', 'b', 'c', '<unk>', '<pad>'], range(5)))
        string = 'bcx'
        tensor = string_to_tensor(string, 5, vocab)
        assert np.equal(tensor, [1, 2, 3, 4, 4]).all()


if __name__ == '__main__':
    unittest.main()
