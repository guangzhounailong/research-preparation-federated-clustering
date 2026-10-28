"""Load, scale, and split the Digits dataset."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .config import DEFAULT_CONFIG


def load_data(split_seed=None, *, config=DEFAULT_CONFIG):
    if split_seed is None:
        split_seed = config.split_seed
    if config.dataset != "digits":
        raise ValueError("load_data currently supports only the digits dataset")
    digits = load_digits()

    X = digits.data.astype(np.float32) / config.pixel_max
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=split_seed)

    return X_train, X_test, y_train, y_test
