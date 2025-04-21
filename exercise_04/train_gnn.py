
import sys
sys.path.append("../")
import os

import time
import argparse
import io
from datetime import datetime
import pickle

import numpy as np
import awkward as awk
#from gnn_encoder import GNNEncoder, collate_fn_gnn
#from gnn_trafo_helper import train_model, evaluate_model, normalize_x, normalize_y, normalize_time, denormalize_x, denormalize_y, get_img_from_matplotlib

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
#from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import torch.nn.functional as F

from util import get_device, EarlyStopping
from data_utils import collate_fn_gnn

DATA_PATH = "../datasets/iceCube/"  # path to the data

batch_size = 50
project_dir = "dev/"
num_epochs = 300
initial_lr = 0.0001


class AwkScaler:
    def __init__(self):
        self.fitted = False

    def fit(self, x):
        if not self.fitted:
            self.mean = awk.mean(x)
            self.std = awk.std(x)
            self.fitted = True
        else:
            raise RuntimeError("Stupid user already fitted")

    def transform(self, x):
        if self.fitted:
            return (x - self.mean)/self.std
        else:
            raise RuntimeError("Stupid user did not fit")

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if self.fitted:
            return (x * self.std) +  self.mean
        else:
            raise RuntimeError("Stupid user did not fit")



train_dataset = awk.from_parquet(os.path.join(DATA_PATH, "train.pq"))
label_x_scaler = AwkScaler()
label_y_scaler = AwkScaler()
label_x_scaler.fit(train_dataset["xpos"])
label_y_scaler.fit(train_dataset["ypos"])

time_scaler = AwkScaler()
x_scaler = AwkScaler()
y_scaler = AwkScaler()
times = train_dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                                # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
x = train_dataset["data"][:, 1:2, :]
y = train_dataset["data"][:, 2:3, :]
time_scaler.fit(times)
x_scaler.fit(x)
y_scaler.fit(y)


def scale_input(dataset):
    """Open dataset, take apart and scale, put back together.

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    times = dataset["data"][:, 0:1, :]
    x = dataset["data"][:, 1:2, :]
    y = dataset["data"][:, 2:3, :]
    norm_times = time_scaler.transform(times)
    norm_x = x_scaler.transform(x)
    norm_y = y_scaler.transform(y)

    return awk.concatenate([norm_times, norm_x, norm_y], axis=1)

def main():
    train_dataset = awk.from_parquet(os.path.join(DATA_PATH, "train.pq"))
    val_dataset = awk.from_parquet(os.path.join(DATA_PATH, "val.pq"))
    test_dataset = awk.from_parquet(os.path.join(DATA_PATH, "test.pq"))

    

    times = train_dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                                # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
    x = train_dataset["data"][:, 1:2, :]
    y = train_dataset["data"][:, 2:3, :]

    time_scaler = AwkScaler()
    x_scaler = AwkScaler()
    y_scaler = AwkScaler()

    time_scaler.fit(times)
    x_scaler.fit(x)
    y_scaler.fit(y)

    # Concatenate the normalized data back together

    # -----labels ----------
    # Normalize labels (this can be done in-place), e.g. by
    label_x_scaler = AwkScaler()
    label_y_scaler = AwkScaler()

    train_dataset["data"] = scale_input(train_dataset)
    train_dataset["xpos"] = label_x_scaler.fit_transform(train_dataset["xpos"])
    train_dataset["ypos"] = label_y_scaler.fit_transform(train_dataset["ypos"])

    val_dataset["data"] = scale_input(val_dataset)
    val_dataset["xpos"] = label_x_scaler.transform(val_dataset["xpos"])
    val_dataset["ypos"] = label_y_scaler.transform(val_dataset["ypos"])


    test_dataset["data"] = scale_input(test_dataset)
    test_dataset["xpos"] = label_x_scaler.transform(test_dataset["xpos"])
    test_dataset["ypos"] = label_y_scaler.transform(test_dataset["ypos"])

    # save all scalers 

    # Hint: You can define a helper function to normalize the data and you can use the same normalization for the validation and test datasets.

    # Create the DataLoader for training, validation, and test datasets
    # Important: We use the custom collate function to preprocess the data for GNN (see the description of the collate function for details)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)


    device = "cpu"#get_device()
    print(device)
    print("Train set dimension: ", train_dataset["data"].type)
    print("Validation set dimension: ", val_dataset["data"].type)
    print("Train set jagged array lengths: ", len([len(event) for event in train_dataset["data"]]))
    print("Validation set jagged array lengths: ", len([len(event) for event in val_dataset["data"]]))

    from models import GNNEncoder

    n_edge_features, n_latent_edge_features = 6, 64

    model = GNNEncoder(n_edge_features, n_latent_edge_features)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6)
    criterion = nn.functional.mse_loss
    es_watcher = EarlyStopping(threshold=0.01, patience=10)

    train_losses = []
    val_losses = []
    lrs = []


    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        # currently no batches are run, needs dataloader
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss/len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        lr = scheduler.get_last_lr()[0]

        if epoch == 0:
            torch.save(model.state_dict(), f"{project_dir}/best_model.pth")
        elif val_loss < val_losses[-1]:
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
    plt.close()
    
    
    model.eval()
    
    test_dataset = awk.from_parquet(os.path.join(DATA_PATH, "test.pq"))
    test_dataset["data"] = scale_input(test_dataset)
    test_dataset["xpos"] = label_x_scaler.transform(test_dataset["xpos"])
    test_dataset["ypos"] = label_y_scaler.transform(test_dataset["ypos"])

    test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False, collate_fn=collate_fn_gnn)
    
    y_pred_tensor = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predictions = model(batch_x)
        y_pred_tensor.append(predictions.detach().cpu())

    y_pred_tensor = torch.cat(y_pred_tensor,dim=0).squeeze()
    y_pred_tensor[:, 0] = label_x_scaler.inverse_transform(y_pred_tensor[:, 0])
    y_pred_tensor[:, 1] = label_y_scaler.inverse_transform(y_pred_tensor[:, 1])
    plt.hist2d(x=y_pred_tensor[:, 0], y=y_pred_tensor[:, 1], bins=(30, 30))
    plt.tight_layout()
    #plt.legend()
    plt.savefig(f"{project_dir}/plots/residuals.png")
    plt.close()



if __name__ == "__main__":
    main()