from faker import Faker
from babel.dates import format_date
import random
from tqdm import tqdm
import numpy as np
import tensorflow.keras as keras

fake = Faker()

# the date format
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']
# the default locale
LOCALES = ['en_US']


def generate_date():
    """
    generate a fake date
    :return: (human string, machine string, date object)
    """

    dt = fake.format('date_object')
    try:
        human = format_date(dt, random.choice(FORMATS), locale=random.choice(LOCALES))
        human = human.lower()
        human = human.replace(',', '')
        machine = dt.isoformat()
    except AttributeError as _:
        return None, None, None
    return human, machine, dt


def generate_dataset(m):
    """
    generate a dataset with m examples and vocabularies
    :param m: number of examples
    :return: dataset, human, machine, inv_machine
    """
    human_vocab = set()
    machine_vocab = set()
    dataset = []

    for _ in tqdm(range(m)):
        h, m, _ = generate_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    # human vocabulate should be something like '-' => 0, '0' -> 1
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'],
                     list(range(len(human_vocab) + 2))))

    # inv_machine is something like 0 -> '_',
    machine = dict(zip(sorted(machine_vocab) + ['<unk>', '<pad>'],
                     list(range(len(machine_vocab) + 2))))

    # machine is reverse of inv_machine
    inv_machine = {v: k for k, v in machine.items()}

    return dataset, human, machine, inv_machine


def string_to_tensor(string, length, vocab):
    """
    map a string into a list of intergers of indexes in the vocabulary
    :param string: input string
    :param length: the max length of the output
    :param vocab: the vocabulary to use
    :return: list of integers of size = length
    """
    string = string.lower()
    string = string.replace(',', '')
    if len(string) > length:
        string = string[:length]

    assert len(string) <= length

    ret = list(map(lambda x: vocab.get(x, vocab['<unk>']), string))

    if len(ret) < length:
        ret += [vocab.get('<pad>')] * (length - len(ret))

    assert len(ret) == length
    return ret


def tensor_to_string(tensor, vocab):
    """
    Map from tensor (list of integers) back to string
    :param tensor: input tensor
    :param vocab: mapping vocabulary
    :return: the mapped list of the string
    """

    ret = [vocab.get(i) for i in tensor]
    return ret


def preprocess_dataset(dataset, human_vocab, machine_vocab, Tx, Ty):
    """
    preprocess dataset
    :param dataset: the input dataset with format [(x1, y1), (x2, y2)...]
    :param human_vocab: the vocabulary to map x
    :param machine_vocab: the vacabulary to map y
    :param Tx: the max length of x
    :param Ty: the max length of y
    :return: X, Y, Xoh (one-hot), Yoh (one-hot)
    """

    X, Y = zip(*dataset)
    X = np.array(list(map(lambda x:
                          string_to_tensor(x, Tx, human_vocab), X)))

    Y = np.array(list(map(lambda y:
                          string_to_tensor(y, Ty, machine_vocab), Y)))

    Xoh = np.array(list(map(lambda tx:
                            keras.utils.to_categorical(tx), X)))

    Yoh = np.array(list(map(lambda ty:
                            keras.utils.to_categorical(ty), Y)))

    return X, Y, Xoh, Yoh
