import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import config


def load_data(split_seed=config.SPLIT_SEED):
    if config.DATASET != "digits":
        raise ValueError("load_data currently supports only the digits dataset")
    digits = load_digits()

    X = digits.data.astype(np.float32) / config.PIXEL_MAX
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=split_seed)

    return X_train, X_test, y_train, y_test
