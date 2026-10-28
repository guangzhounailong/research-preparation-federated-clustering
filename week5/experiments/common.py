"""保存作图所需表示，避免可视化阶段重新训练。"""

import numpy as np


def save_features(output_dir, name, Z_train, Z_test):
    folder = output_dir / "representations"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f"{name}.npz", Z_train=Z_train, Z_test=Z_test)


def load_features(output_dir, name):
    with np.load(output_dir / "representations" / f"{name}.npz") as features:
        return features["Z_train"], features["Z_test"]
