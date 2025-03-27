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


class ConvNNModel(nn.Module):
    def __init__(self, n_labels):
        super(ConvNNModel, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=10, out_channels=2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_labels),
        )


    def forward(self, x):
        x.unsqueeze_(1)
        return self.net(x)