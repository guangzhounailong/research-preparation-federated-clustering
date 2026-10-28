"""Split data into IID or non-IID client partitions."""

import numpy as np

def partition_data(y, num_clients, mode, seed):
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y should be 1 dimension")

    if not 1 <= num_clients <= len(y):
        raise ValueError("number of client should between 1 and total samples")

    rng = np.random.default_rng(seed)
    if mode == "iid":
        indices = rng.permutation(len(y))
        client_indices = np.array_split(indices, num_clients)
    elif mode == "noniid":
        classes = np.unique(y)
        shuffled_classes = rng.permutation(classes)
        class_groups = np.split(shuffled_classes, num_clients)

        client_indices = []

        for client_classes in class_groups:
            # Select all samples from the two classes assigned to this client.
            mask = np.isin(y, client_classes)
            indices = np.flatnonzero(mask)

            # Shuffle the sample order within this client.
            indices = rng.permutation(indices)

            client_indices.append(indices)
    else:
        raise ValueError("mode must be 'iid' or 'noniid'")

    return client_indices