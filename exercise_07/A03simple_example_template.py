import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import torch
import seaborn as sns  # a useful plotting library on top of matplotlib
from tqdm.auto import tqdm # a nice progress bar

# # This is a simple example of a diffusion model in 1D.


# generate a dataset of 1D data from a mixture of two Gaussians
# this is a simple example, but you can use any distribution
data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1, 2])),
    torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1.]))
)

dataset = data_distribution.sample(torch.Size([10000]))  # create training data set
dataset_validation = data_distribution.sample(torch.Size([1000])) # create validation data set
# fig, ax = plt.subplots(1, 1)
# sns.histplot(dataset)
# plt.show()

# we will keep these parameters fixed throughout
# these parameters should give you an acceptable result
# but feel free to play with them
TIME_STEPS = 250
BETA = torch.tensor(0.02)
N_EPOCHS = 1#1000
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4

# define the neural network that predicts the amount of noise that was
# added to the data
# the network should have two inputs (the current data and the time step)
# and one output (the predicted noise)

class SimpleDiffusion(torch.nn.Module):
    def __init__(self):
        super(SimpleDiffusion, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 128),  # input: data + time step
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)  # output: predicted noise
        )

    def forward(self, x, t):
        # concatenate the data and time step as inputs
        x_t = torch.cat([x.view(-1, 1), t.view(-1, 1).expand(x.size(0), 1)], dim=1)
        return self.net(x_t)

g = SimpleDiffusion()

epochs = tqdm(range(N_EPOCHS))  # this makes a nice progress bar
for e in epochs: # loop over epochs
    g.train()
    indices = torch.randperm(dataset.shape[0])
    shuffled_dataset = dataset[indices]
    for i in range(0, shuffled_dataset.shape[0] - BATCH_SIZE, BATCH_SIZE):
        x0 = shuffled_dataset[i:i + BATCH_SIZE]

        # here, implement algorithm 1 of the DDPM paper (https://arxiv.org/abs/2006.11239)

        # Step 1: Sample noise
        noise = torch.randn_like(x0)

        # Step 2: Compute alpha and alpha_bar
        alpha = 1 - BETA
        alpha_bar = torch.cumprod(alpha.repeat(TIME_STEPS), dim=0)

        # Step 3: Forward diffusion process
        t = torch.randint(0, TIME_STEPS, (x0.shape[0],))
        alpha_bar_t = alpha_bar[t].view(-1, 1)
        noisy_x = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        # Step 4: Predict noise using the model
        predicted_noise = g(noisy_x, t)

        # Step 5: Compute loss (mean squared error between predicted and true noise)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Step 6: Backpropagation and optimization
        optimizer = torch.optim.Adam(g.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # compute the loss on the validation set
    g.eval()
    with torch.no_grad():
        x0 = dataset_validation



def sample_reverse(g, count):
    """
    Sample from the model by applying the reverse diffusion process

    Here, implement algorithm 2 of the DDPM paper (https://arxiv.org/abs/2006.11239)

    Parameters
    ----------
    g : torch.nn.Module
        The neural network that predicts the noise added to the data
    count : int
        The number of samples to generate in parallel

    Returns
    -------
    x : torch.Tensor
        The final sample from the model
    """
    
    x = torch.rand((count))
    g.eval()
    with torch.no_grad():
        # Step 1: Initialize x_T ~ N(0, I)
        x = torch.randn(count)

        # Step 2: Reverse diffusion process
        alpha = 1 - BETA
        alpha_bar = torch.cumprod(alpha.repeat(TIME_STEPS), dim=0)
        for t in reversed(range(TIME_STEPS)):
            alpha_t = alpha[t]
            alpha_bar_t = alpha_bar[t]
            alpha_bar_t_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)

            # Predict noise using the model
            t_tensor = torch.full((count,), t, dtype=torch.long)
            predicted_noise = g(x, t_tensor)

            # Compute mean of the reverse process
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
            )

            # Sample from the reverse process
            if t > 0:
                variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean  # No noise added at t=0

        return x




samples = sample_reverse(g, 1000)
samples = samples.detach().numpy()

# plot the samples
fig, ax = plt.subplots(1, 1)
bins = np.linspace(-10, 10, 50)
sns.kdeplot(dataset, ax=ax, color='blue', label='True distribution', linewidth=2)
sns.histplot(samples, ax=ax, bins=bins, color='red', label='Sampled distribution', stat='density')
ax.legend()
ax.set_xlabel('Sample value')
ax.set_ylabel('Sample count')
