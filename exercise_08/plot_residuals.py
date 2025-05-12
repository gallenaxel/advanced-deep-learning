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

device = "cpu"#get_device()
 


def predict_labels(args) -> tuple[torch.Tensor, torch.Tensor]:
    project_dir = f"dev/"
    from models import GNNEncoder
    from train_gnn import project_dir, DATA_PATH, scale_input
    from train_gnn import label_x_scaler, label_y_scaler
    
    n_edge_features, n_latent_edge_features = 6, 64
    model = GNNEncoder(n_edge_features, n_latent_edge_features)

    model.load_state_dict(torch.load(f"{project_dir}/best_model.pth", weights_only=True))
    model.eval()
    model = model.to(device)
    
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
    project_dir = f"dev/"

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
    plt.savefig(f"{project_dir}/plots/residuals.png")
    plt.close()
        
    return


def plot_distributions(args):
    project_dir = f"dev/"

    y_pred, x_data = predict_labels(args)
    
    # Plot true interaction distribution
    plt.figure()
    plt.hist2d(x=np.array(x_data["xpos"]), y=np.array(x_data["ypos"]), bins=(30, 30))
    plt.xlabel("True X Position")
    plt.ylabel("True Y Position")
    plt.title("True Interaction Distribution")
    plt.savefig(f"{project_dir}data_plots/truth_interaction_distribution.png")
    plt.close()

    # Plot number of nodes distribution
    plt.figure()
    plt.hist(awk.count(x_data["data"][:, 0:1, :], axis=2), bins=30)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    plt.title("Number of Nodes Distribution")
    plt.savefig(f"{project_dir}data_plots/n_nodes_distribution.png")
    plt.close()

    # Plot feature means distribution
    plt.figure()
    for i, label in enumerate(["t", "x", "y"]):
        plt.hist(awk.mean(x_data["data"][:, i:i+1, :], axis=2), label=label, histtype="step", bins=30)
    plt.xlabel("Mean Feature Value per Graph")
    plt.ylabel("Frequency")
    plt.legend(title="Mean of features per graph")
    plt.title("Feature Means Distribution")
    plt.savefig(f"{project_dir}data_plots/feature_means_distribution.png")
    plt.close()

    # Plot predicted vs true x and y positions as subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # X Position subplot
    axs[0].scatter(x=np.array(x_data["xpos"]), y=y_pred[:, 0], alpha=0.5)
    axs[0].set_xlabel("True X Position")
    axs[0].set_ylabel("Predicted X Position")
    axs[0].set_title("Predicted vs True X Positions")

    # Y Position subplot
    axs[1].scatter(x=np.array(x_data["ypos"]), y=y_pred[:, 1], alpha=0.5)
    axs[1].set_xlabel("True Y Position")
    axs[1].set_ylabel("Predicted Y Position")
    axs[1].set_title("Predicted vs True Y Positions")

    plt.tight_layout()
    plt.savefig(f"{project_dir}data_plots/predicted_vs_true_positions.png")
    plt.close()
    
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-normalizing_flow_type",
        default="diagonal_gaussian",
        choices=["diagonal_gaussian", "full_gaussian", "full_flow"],
    )
    args = parser.parse_args()
    plot_residuals(args)
    plot_distributions(args)

if __name__ == "__main__":

    main()