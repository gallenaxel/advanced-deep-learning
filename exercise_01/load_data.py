import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from matplotlib import pyplot as plt
from torchsummary import summary

DATA_PATH = "../datasets/galah4"

def get_data() -> tuple[np.ndarray, np.ndarray]:
    spectra = np.load(f"{DATA_PATH}/spectra.npy")
    # labels: mass, age, l_bol, dist, t_eff, log_g, fe_h, SNR
    labelNames = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
    labels = np.load(f"{DATA_PATH}/labels.npy")
    # We only use the three labels: t_eff, log_g, fe_h, SNR
    label_slice = slice(-4, -1)
    labelNames = labelNames[label_slice]
    labels = labels[:, label_slice]

    spectra = np.log(np.maximum(spectra, 0.2))

    return spectra, labels


def get_train_test_val(data_x, data_y) -> tuple[TensorDataset]:
    torch.manual_seed(42)
    num_samples = len(data_x)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    tra, val, tst = random_split(TensorDataset(data_x, data_y),
                                 [train_size, val_size, test_size],
                                 )

    return tra, val, tst

def get_batch_loaders(data_x, data_y, batch_size=10):
    torch.manual_seed(42)

    tra, val, tst = get_train_test_val(data_x, data_y)

    tra_loader = DataLoader(tra, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    tst_loader = DataLoader(tst, batch_size=batch_size, shuffle=False)

    return tra_loader, val_loader, tst_loader


if __name__ == "__main__":
    spectra, labels = get_data()
    for i in range(10):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(spectra[i], lw=1)
        ax.set_title(f"Star {i}")
        plt.tight_layout()
        plt.savefig(f"plots/star_spectrum_{i}.png")

    for i in range(len(labels)):
        pass
    print(labels)