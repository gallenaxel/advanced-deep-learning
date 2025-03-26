import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, n_labels):
        super(CNNModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(16384, 128),
            nn.ReLU(),
            nn.Linear(128, n_labels)
        )

    def forward(self, x):
        return self.net(x)
