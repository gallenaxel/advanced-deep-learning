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

from load_data import get_data, labelNames
from util import get_device

device = get_device()

project_dir = 'training_stuff_conv'

def predict_labels() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from models import CNNModel, ConvNNModel
    model = ConvNNModel(3).to(device)
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
    y_pred = labels_scaler.inverse_transform(y_pred_tensor)
    
    print(summary(model.cpu(), ( 16384,)))

    return y_pred, y_data, x_data


def plot_residuals():
    y_pred, y_data, x_data = predict_labels()
    residual = (y_pred - y_data) / y_data
    for i, label in enumerate(labelNames):
        low = np.percentile(residual[:, i], 5)
        high = np.percentile(residual[:, i], 95)
        mean = np.mean(residual[:, i])
        counts, bins, patch = plt.hist(residual[:, i], label=label, bins=np.linspace(low, high, 100))
        plt.xlabel(f"{label} " + r"$\frac{pred - true}{true}$", loc="right")
        plt.ylabel('Counts',loc="top")
        plt.vlines(0, 0, counts.max(), linestyles="dashed", colors="k",label="Zero")
        plt.vlines(mean, 0, counts.max(), linestyles="dashed", colors="r",label=f"Predicted: {np.round(mean,4)}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{project_dir}/residuals_{label}.png")
        plt.close()


    return


def plot_distributions():
    y_pred, y_data, x_data = predict_labels()
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