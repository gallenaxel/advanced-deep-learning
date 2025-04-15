import sys
import pickle
sys.path.append("../")
import argparse


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, random_split


from exercise_01.load_data import get_data, labelNames
from util import get_device

device = get_device()
 


def predict_labels(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    project_dir = f"{args.normalizing_flow_type}/training_stuff_nf_solution"
    from exercise_03.models.nf_model import TinyCNNEncoder, CombinedModel, ConvNNModel
    model = CombinedModel(ConvNNModel, 16384, nf_type=args.normalizing_flow_type)

    model.load_state_dict(torch.load(f"{project_dir}/best_model.pth"))
    model = model.to(device)
    model.eval()
    x_data, y_data = get_data("test")
    x_data = x_data[:30]
    y_data = y_data[:30]



    with open(f"{project_dir}/label_scaler.pickle", 'rb') as f:
        labels_scaler = pickle.load(f)
    x_tensor = torch.tensor(x_data.astype(np.float32)).to(device)
    y_tensor = torch.tensor(y_data.astype(np.float32)).to(device)
    print(x_tensor.shape)
    
    y_tensor_plot = torch.tensor(labels_scaler.transform(y_data).astype(np.float32)).to(device)
    for n in [0, 120, 314]:
        model.visualize_pdf(
            x_tensor, f"{project_dir}/plots/output_pdf{n}.png", samplesize=1000, batch_index=0, truth=y_tensor_plot, labelNames=labelNames,
        )
        
    dataset_test = TensorDataset(
        x_tensor,
        y_tensor,
    )

    test_loader = DataLoader(dataset_test, batch_size=30, shuffle=False)
    
    
    y_pred_tensor = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predictions = model(batch_x)
        y_pred_tensor.append(predictions.detach().cpu())

      
    y_pred_tensor = torch.stack(y_pred_tensor, dim=0).squeeze()#model(x_tensor).detach().cpu()
    y_pred = labels_scaler.inverse_transform(y_pred_tensor[:, :3])
    
    y_pred_ucert = y_pred_tensor[:, 3:] * labels_scaler.scale_

    return y_pred, y_pred_ucert, y_data, x_data


def plot_residuals(args):
    project_dir = f"{args.normalizing_flow_type}/training_stuff_nf_solution"

    y_pred, y_pred_ucert, y_data, x_data = predict_labels(args)
    print("Predicted labels")
    residual = (y_pred - y_data) / y_data
    residual_ucert = (y_pred - y_data) / y_pred_ucert
    for i, label in enumerate(labelNames):
        low = np.percentile(residual[:, i], 5)
        high = np.percentile(residual[:, i], 95)
        bins = np.linspace(low, high, 30)
        mean = np.mean(residual[:, i])
        std = np.std(residual[:, i])
        counts, bins, patch = plt.hist(residual[:, i], label=label, bins=bins)
        plt.xlabel(f"{label} " + r"$\frac{pred - true}{true}$", loc="right")
        plt.ylabel('Counts',loc="top")
        plt.vlines(0, 0, counts.max(), linestyles="dashed", colors="k",label="Zero")
        plt.vlines(mean, 0, counts.max(), linestyles="dashed", colors="r",label=f"Predicted: {np.round(mean,4)}")
        ax = plt.gca()

        ax.text(0.2, 0.8, f'bias = {100.*mean:.1f} %\nstd = {100.*std:.1f} %',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{project_dir}/residuals_{label}.png")
        plt.close()
        

        bins = np.linspace(-3, 3, 30)
        #mean = np.mean(residual_ucert[:, i])
        #std = np.std(residual_ucert[:, i])
        counts, bins, patch = plt.hist(residual_ucert[:, i], label=label, bins=bins, density=True)
        plt.xlabel(f"{label} " + r"$\frac{pred - true}{\sigma_{pred}}$", loc="right")
        plt.ylabel('Counts',loc="top")
        plt.vlines(0, 0, counts.max(), linestyles="dashed", colors="k",label="Zero")
        #plt.vlines(mean, 0, counts.max(), linestyles="dashed", colors="r",label=f"Predicted: {np.round(mean,4)}")
        xx = np.linspace(bins[0], bins[-1], 100)
        plt.plot(xx, 1/(np.sqrt(2*np.pi)) * np.exp(-(xx**2)))
        ax = plt.gca()

        #ax.text(0.2, 0.8, f'bias = {100.*mean:.1f} %\nstd = {100.*std:.1f} %',
        #    horizontalalignment='left',
        #    verticalalignment='top',
        #    transform=ax.transAxes)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{project_dir}/residuals_ucert_{label}.png")
        plt.close()
        
        plt.hist(y_pred_ucert[:, i])
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{project_dir}/ucert_{label}.png")
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