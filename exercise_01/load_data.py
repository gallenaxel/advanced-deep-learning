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


if __name__ == "__main__":
    spectra, labels = get_data()
    for i in range(10):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(spectra[i], lw=1)
        ax.set_title(f"Star {i}")
        plt.tight_layout()
        plt.savefig(f"plots/star_spectrum_{i}.png") 