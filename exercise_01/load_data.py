from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torchsummary import summary

DATA_PATH = "../datasets/galah4"

labelNamesAll = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
label_slice = slice(-4, -1)
labelNames = labelNamesAll[label_slice].copy()


def get_data() -> tuple[np.ndarray, np.ndarray]:
    spectra = np.load(f"{DATA_PATH}/spectra.npy")
    # labels: mass, age, l_bol, dist, t_eff, log_g, fe_h, SNR

    labels = np.load(f"{DATA_PATH}/labels.npy")
    # We only use the three labels: t_eff, log_g, fe_h, SNR

    labels = labels[:, label_slice]
    spectra = np.log(np.maximum(spectra, 0.2))

    return spectra, labels


def get_train_test_val(data_x, data_y) -> tuple[TensorDataset]:
    torch.manual_seed(42)
    num_samples = len(data_x)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    tra, val, tst = random_split(
        TensorDataset(data_x, data_y),
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


def plot_spectra(spectra, labels):
    for i in range(10):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(spectra[i], lw=1)
        ax.set_title(f"Star {i}")
        plt.tight_layout()
        plt.savefig(f"plots/star_spectrum_{i}.png")
        plt.close()


def plot_labels_single(labels):
    for i, label_name in enumerate(labelNames):
        plt.hist(labels[:, i].flatten())
        plt.xlabel(label_name)
        plt.savefig(f"plots/{labelNames[i]}.png")
        plt.close()


def plot_correlogram(labels, outname="labels_overall"):
    data_labels = pd.DataFrame(labels, columns=labelNames)
    sns.pairplot(data_labels, kind="hist")
    plt.tight_layout()
    plt.savefig(f"plots/{outname}.png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        "This module provides data loading and plotting as a script"
    )
    parser.add_argument("--spectra", action="store_true", help="Plot some spectra")
    parser.add_argument(
        "--single-labels", action="store_true", help="Plot the labels on their own"
    )
    parser.add_argument("--correlogram", action="store_true", help="Plot correlogram")
    args = parser.parse_args()
    spectra, labels = get_data()

    if args.spectra:
        plot_spectra(spectra, labels)

    if args.single_labels:
        plot_labels_single(labels)

    if args.correlogram:
        plot_correlogram(labels)
