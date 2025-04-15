
import sys
sys.path.append("../")
import os

import pickle

from functools import partial

from tqdm import tqdm

import torch.nn as nn

import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from exercise_01.load_data import get_data, plot_correlogram
from exercise_01.models import ConvNNModel
from uncert_pred_loss import nll_loss
from util import run_training_loop, get_device, EarlyStopping

import matplotlib.pyplot as plt

num_epochs = 50

def main():

    #TODO: Argparse this
    project_dir = "training_stuff_conv_dev"
    model = None # get a model loader, also used for plotting

    batch_size = 10

    if not os.path.exists(project_dir):
        os.mkdir(project_dir)

    spectra_train, labels_train = get_data("train") # spectra_train shape (8914, 16384)
    spectra_val, labels_val = get_data("val") # spectra_train shape (8914, 16384)

    labels_scaler = StandardScaler()
    labels_scaler.fit(labels_train)
    labels_train = labels_scaler.transform(labels_train)
    labels_val = labels_scaler.transform(labels_val)
    plot_correlogram(labels_train, outname=f"{project_dir}/correlogram_scaled")

    with open(f"{project_dir}/label_scaler.pickle", 'wb') as f:
        pickle.dump(labels_scaler, f)

    device = get_device()

    dataset_train = TensorDataset(
        torch.tensor(spectra_train.astype(np.float32)).to(device),
        torch.tensor(labels_train.astype(np.float32)).to(device))
    dataset_val = TensorDataset(
        torch.tensor(spectra_val.astype(np.float32)).to(device),
        torch.tensor(labels_val.astype(np.float32)).to(device))

    model = ConvNNModel(n_labels=6)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6)
    criterion = partial(nll_loss, n_labels=3)
    es_watcher = EarlyStopping(threshold=0.01, patience=10)

    train_losses = []
    val_losses = []
    lrs = []

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        # currently no batches are run, needs dataloader
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            forward_pass_outputs = model(batch_x)
            loss = criterion(forward_pass_outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss/len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        lr = scheduler.get_last_lr()[0]

        if epoch > 0:
            if val_loss < val_losses[-1]:
                torch.save(model.state_dict(), f"{project_dir}/best_model.pth")
        val_losses.append(val_loss)
        lrs.append(lr)
        stop = es_watcher.step(val_loss)
        if stop:
            print(f"Early stopping applied after epoch {epoch}")
            break

    training_metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lrs": lrs,
    }
    with open(f"{project_dir}/training_metrics.pickle", 'wb') as f:
        pickle.dump(training_metrics, f)

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{project_dir}/training_performance")

if __name__ == "__main__":
    main()