import torch.nn as nn

class Generator(nn.Module):
    """
    Generator Model
    """
    def __init__(self, image_dimension, latent_dimension):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dimension, 256),  # Example hidden dimension
            nn.ReLU(),
            nn.Linear(256, image_dimension),
            nn.Tanh()  # It is helpful to use the tanh activation function to force the output into the [-1,1] range that our normalized images have.
        )

    def forward(self, x):
        return self.gen(x)



class Discriminator(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self, image_dimension):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dimension, 256),  # Example hidden dimension
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),  # Output layer for binary classification
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.disc(x)