import sys
import pickle
sys.path.append("../")


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from load_data import get_data, labelNames
from util import get_device

device = get_device()

def predict_labels():
    from models import CNNModel
    model = CNNModel(3).to(device)
    model.load_state_dict(torch.load('training_stuff/best_model.pth'))
    model.eval()
    x_data, y_data = get_data()

    with open("training_stuff/label_scaler.pickle", 'rb') as f:
        labels_scaler = pickle.load(f)
    x_tensor = torch.tensor(x_data.astype(np.float32)).to(device)
    y_pred_tensor = model(x_tensor).detach().cpu()
    y_pred = labels_scaler.inverse_transform(y_pred_tensor)
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
        plt.vlines([0, mean], 0, counts.max(), linestyles="dashed", colors="k")
        plt.tight_layout()
        plt.savefig(f"training_stuff/residuals_{label}.png")
        plt.close()


    return


def plot_distributions():
    y_pred, y_data, x_data = predict_labels()

    df_pred = pd.DataFrame(y_pred, columns=labelNames)
    df_pred["type"] = "prediction"

    df_data = pd.DataFrame(y_data, columns=labelNames)
    df_data["type"] = "data"

    df = pd.concat([df_data, df_pred])

    sns.pairplot(df, kind="hist", hue="type")
    plt.savefig(f"training_stuff/test_distributions_comaprisons.png")


def main():
    plot_residuals()
    plot_distributions()

if __name__ == "__main__":
    main()