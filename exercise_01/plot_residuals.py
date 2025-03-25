import sys
sys.path.append("../")


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from load_data import get_data
from util import get_device

device = get_device()

def predict_labels():
    from models import CNNModel
    model = CNNModel(3).to(device)
    model.load_state_dict(torch.load('saved_models/best_model.pth'))
    model.eval()
    x_data, y_data = get_data()
    
    #TODO scale labels, and rescale with inverse
    x_tensor = torch.tensor(x_data.astype(np.float32)).to(device)
    y_pred_tensor = model(x_tensor).detach()
    y_pred = y_pred_tensor #TODO * y_std + y_mean
    return y_pred, y_data, x_data


def plot_residuals():
    y_pred, y_data, x_data = predict_labels()
    residual = y_pred.to("cpu") - y_data

    plt.hist(residual)
    plt.show()

    return


def main():
    plot_residuals()


if __name__ == "__main__":
    main()