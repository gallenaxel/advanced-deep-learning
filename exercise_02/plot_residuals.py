import sys
import pickle
sys.path.append("../")


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchsummary import summary

from exercise_01.load_data import get_data, labelNames
from util import get_device

device = get_device()

project_dir = 'training_stuff_conv_dev'

def predict_labels() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from exercise_01.models import CNNModel, ConvNNModel
    model = ConvNNModel(6).to(device)
    model.load_state_dict(torch.load(f"{project_dir}/best_model.pth"))
    model = model.to(device)
    model.eval()
    print(model)
    x_data, y_data = get_data("test")
    print(x_data.shape)


    with open(f"{project_dir}/label_scaler.pickle", 'rb') as f:
        labels_scaler = pickle.load(f)
    x_tensor = torch.tensor(x_data.astype(np.float32)).to(device)
    print(x_tensor.shape)
    y_pred_tensor = model(x_tensor).detach().cpu()
    y_pred = labels_scaler.inverse_transform(y_pred_tensor[:, :3])
    
    y_pred_ucert = torch.exp(y_pred_tensor[:, 3:]) * labels_scaler.scale_

    print(summary(model.cpu(), ( 16384,)))

    return y_pred, y_pred_ucert, y_data, x_data


def plot_residuals():
    y_pred, y_pred_ucert, y_data, x_data = predict_labels()
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


def plot_distributions():
    y_pred, y_pred_ucert, y_data, x_data = predict_labels()
    print(x_data.shape)

    df_pred = pd.DataFrame(y_pred, columns=labelNames)
    df_pred["type"] = "prediction"

    df_data = pd.DataFrame(y_data, columns=labelNames)
    df_data["type"] = "data"

    df = pd.concat([df_data, df_pred])

    sns.pairplot(df, kind="hist", hue="type")
    plt.savefig(f"{project_dir}/test_distributions_comaprisons.png")


def main():
    plot_residuals()
    plot_distributions()

if __name__ == "__main__":
    main()