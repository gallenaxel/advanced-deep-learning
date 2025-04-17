import sys
import pickle
sys.path.append("../")
import argparse
import os

import torch
import torch.nn as nn
import numpy as np
import awkward as awk
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

#from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, random_split

from util import get_device
from data_utils import collate_fn_gnn

device = get_device()
 


def predict_labels(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    project_dir = f"{args.normalizing_flow_type}/training_stuff_nf_solution"
    from models import GNNEncoder
    from train_gnn import project_dir, DATA_PATH, scale_input
    from train_gnn import label_x_scaler, label_y_scaler
    
    n_edge_features, n_latent_edge_features = 6, 64
    model = GNNEncoder(n_edge_features, n_latent_edge_features)

    model.load_state_dict(torch.load(f"{project_dir}/best_model.pth", weights_only=True))
    model = model.to(device)
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
    

    return y_pred_tensor, test_dataset


def plot_residuals(args):
    project_dir = f"dev/plots"

    y_pred, data = predict_labels(args)
    print("Predicted labels")
    #residual_x = (y_pred[0, :] - data["xpos"].numpy()) / data["xpos"].numpy()
    #residual_y = (y_pred[1, :] - data["ypos"].numpy()) / data["ypos"].numpy()
    print(y_pred)
    plt.hist2d(x=y_pred[:, 0], y=y_pred[:, 1], bins=(30, 30))
    #plt.hist2d(x=residual_x, y=residual_y)
    
    #counts, bins, patch = plt.hist2d(x=y_pred[0, :], y=y_pred[1, :])
    #plt.xlabel(f"{label} " + r"$\frac{pred - true}{true}$", loc="right")
    #ax = plt.gca()
    #ax.text(0.2, 0.8, f'bias = {100.*mean:.1f} %\nstd = {100.*std:.1f} %',
    #    horizontalalignment='left',
    #    verticalalignment='top',
    #    transform=ax.transAxes)
    plt.tight_layout()
    #plt.legend()
    plt.savefig(f"{project_dir}/residuals.png")
    plt.close()
        
    return


def plot_distributions(args):
    project_dir = f"{args.normalizing_flow_type}/training_stuff_nf_solution"

    y_pred, y_pred_ucert, y_data, x_data = predict_labels(args)
    print(x_data.shape)

    df_pred = pd.DataFrame(y_pred, columns=labelNames)
    df_pred["type"] = "prediction"

    df_data = pd.DataFrame(y_data, columns=labelNames)
    df_data["type"] = "data"

    df = pd.concat([df_data, df_pred])

    sns.pairplot(df, kind="hist", hue="type")
    plt.savefig(f"{project_dir}/test_distributions_comaprisons.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-normalizing_flow_type",
        default="diagonal_gaussian",
        choices=["diagonal_gaussian", "full_gaussian", "full_flow"],
    )
    args = parser.parse_args()
    plot_residuals(args)
    #plot_distributions(args)

if __name__ == "__main__":

    main()