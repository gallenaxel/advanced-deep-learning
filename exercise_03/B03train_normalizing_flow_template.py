import sys

sys.path.append("../")

import time
import sys
import os
import argparse
import glob
import subprocess
import pickle
from functools import partial

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
import jammy_flows
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


from models.nf_model import CombinedModel, TinyCNNEncoder, ConvNNModel
from deconstruct_pdf_step_by_step import deconstruct_pdf_layer_by_layer, plot_global
from exercise_01.load_data import get_data, labelNames
from util import get_device, EarlyStopping


DATA_PATH = "../datasets/galah4/"
fp64_on_cpu = False

# Hyperparameters
learning_rate = 0.8e-5
batch_size = 32


def nf_loss(inputs, batch_labels, model):
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(
        batch_labels, inputs
    )  # get the probability of the labels given the input data
    loss = -log_pdfs.mean()  # take the negative mean of the log probabilities
    return loss


# Defining the normalizng flow model is a bit more involved and requires knowledge of the jammy_flows library.
# Therefore, we provide the relevant code here.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-normalizing_flow_type",
        default="diagonal_gaussian",
        choices=["diagonal_gaussian", "full_gaussian", "full_flow"],
    )
    args = parser.parse_args()
    print("Using normalizing flow type ", args.normalizing_flow_type)

     # Define folder for saving plots
    if not os.path.exists(args.normalizing_flow_type):
        os.makedirs(args.normalizing_flow_type)  # Create folder if it doesn't exist
    training_stuff_dir = f"{args.normalizing_flow_type}/training_stuff_nf_solution"
    num_epochs = 400
    batch_size = 20
    initial_lr = 1e-4

    if not os.path.exists(training_stuff_dir):
        os.mkdir(training_stuff_dir)

    model = CombinedModel(ConvNNModel, 16384, nf_type=args.normalizing_flow_type)

    # Detect and use Apple Silicon GPU (MPS) if available
    device = torch.device(get_device())
    print(device)
    if args.normalizing_flow_type == "full_flow" and device.type == "mps":
        # MPS does not support double precision, therefore we need to run the flow on the CPU
        fp64_on_cpu = True
    print(f"Using device: {device}, performing fp64 on CPU: {fp64_on_cpu}")
    model.to(device)

    spectra_train, labels_train = get_data("train")  # spectra_train shape (8914, 16384)
    spectra_val, labels_val = get_data("val")  # spectra_train shape (8914, 16384)
    
    labels_scaler = StandardScaler()
    labels_scaler.fit(labels_train)
    labels_train = labels_scaler.transform(labels_train)
    labels_val = labels_scaler.transform(labels_val)

    with open(f"{training_stuff_dir}/label_scaler.pickle", 'wb') as f:
        pickle.dump(labels_scaler, f)
    dataset_train = TensorDataset(
        torch.tensor(spectra_train.astype(np.float32)).to(device),
        torch.tensor(labels_train.astype(np.float32)).to(device),
    )
    dataset_val = TensorDataset(
        torch.tensor(spectra_val.astype(np.float32)).to(device),
        torch.tensor(labels_val.astype(np.float32)).to(device),
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=6
    )
    criterion = partial(nf_loss, model=model)
    es_watcher = EarlyStopping(threshold=0.01, patience=10)

    train_losses = []
    val_losses = []
    lrs = []

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    iteration = 0
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        # currently no batches are run, needs dataloader
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # batch_x = batch_x[0].unsqueeze(-1)  # Get batch data and add a dimension
            # batch_x = batch_x.to(dtype=torch.float64)  # Convert to double precision
            optimizer.zero_grad()  # Zero the gradients
            #res = model(batch_x)  # Forward pass
            loss = criterion(batch_x, batch_y).cpu() # Compute loss
            loss.backward()  # Backward pass
            train_loss += float(loss)
            optimizer.step()  # Update parameters

            # Every 1000 batch iterations, plot the current fit
            if iteration % 300 == 0:
                current_loss = loss.item()
                print(f"Epoch {epoch} Iteration {batch_idx}, Loss = {current_loss:.4f}")
                with torch.no_grad():
                    fw_list, bw_list = deconstruct_pdf_layer_by_layer(model.pdf)
                    plot_global(
                        fw_list, bw_list, iteration, loss.cpu().detach().numpy(),
                        training_stuff_dir + "/plots/"
                    )
            iteration += 1


        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
           for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                #predictions = model(batch_x)
                loss = float(criterion(batch_x, batch_y).cpu())
                val_loss += loss
               
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        lr = scheduler.get_last_lr()[0]

        if epoch > 0:
            if val_loss < val_losses[-1]:
                torch.save(model.state_dict(), f"{training_stuff_dir}/best_model.pth")
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
    with open(f"{training_stuff_dir}/training_metrics.pickle", "wb") as f:
        pickle.dump(training_metrics, f)

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{training_stuff_dir}/training_performance")
