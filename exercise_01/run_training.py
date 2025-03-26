
import sys
sys.path.append("../")

import pickle

from tqdm import tqdm

import torch.nn as nn

import torch
import numpy as np

from sklearn.preprocessing import StandardScaler

from load_data import get_data, get_batch_loaders, plot_correlogram
from models import ConvNNModel
from util import run_training_loop, get_device, EarlyStopping

import matplotlib.pyplot as plt

num_epochs = 10

def main():
    spectra, labels = get_data() # spectra shape (8914, 16384)

    labels_scaler = StandardScaler()
    labels_scaler.fit(labels)
    labels = labels_scaler.transform(labels)
    plot_correlogram(labels, outname="correlogram_scaled")

    with open("training_stuff_conv/label_scaler.pickle", 'wb') as f:
        pickle.dump(labels_scaler, f)

    device = get_device()

    spectra = torch.tensor(spectra.astype(np.float32)).to(device).unsqueeze(1) #TODO make this nicer
    labels = torch.tensor(labels.astype(np.float32)).to(device)

    print(np.shape(spectra))

    model = ConvNNModel(n_labels=3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    criterion = nn.functional.mse_loss
    es_watcher = EarlyStopping()

    train_losses = []
    val_losses = []
    lrs = []

    train_batch_loader, val_batch_loader, test_batch_loader = get_batch_loaders(spectra, labels)
    for epoch in tqdm(range(num_epochs)):

        train_loss = 0
        # currently no batches are run, needs dataloader
        for batch_idx, (batch_x, batch_y) in enumerate(train_batch_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            forward_pass_outputs = model(batch_x)
            loss = criterion(forward_pass_outputs.ravel(), batch_y.ravel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss/len(train_batch_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_batch_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_batch_loader)
        scheduler.step(val_loss)
        lr = scheduler.get_last_lr()[0]

        if epoch > 0:
            if val_loss < val_losses[-1]:
                torch.save(model.state_dict(), "training_stuff_conv/best_model.pth")

        val_losses.append(val_loss)
        lrs.append(lr)
        stop = es_watcher.step(val_loss)
        if stop:
            print(f"Early stooping applied after epoch {epoch}")
            break

    training_metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lrs": lrs,
    }
    with open("training_stuff_conv/training_metrics.pickle", 'wb') as f:
        pickle.dump(training_metrics, f)

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_stuff_conv/training_performance")

if __name__ == "__main__":
    main()