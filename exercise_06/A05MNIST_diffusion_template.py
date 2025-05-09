import sys
sys.path.append("../")
import os

from tqdm import tqdm

import torchvision
# For image transforms
from torchvision import transforms
# For DATA SET
import torchvision.datasets as datasets
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
# FOR DATA LOADER
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from util import get_device

device = get_device()

# Hyperparameters
LEARNING_RATE = 4e-4
BATCH_SIZE = 64  # Batch size
N_EPOCHS = 5
IMAGE_SIZE = 28
TIME_STEPS = 1000
SAMPLING_TIMESTEPS = 10


# we define a transform that converts the image to tensor and normalizes it
myTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="../datasets/MNIST_transformed/", transform=myTransforms, download=True)
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="../datasets/MNIST_transformed/", train=False, download=False, transform=myTransforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


DIM = 32
DIM_MULTS = (1, 2, 5)
model = Unet(
    dim = DIM,
    dim_mults = DIM_MULTS,
    flash_attn = True,
    channels = 1
)#.to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = IMAGE_SIZE,
    timesteps = TIME_STEPS,           # number of steps
    sampling_timesteps = SAMPLING_TIMESTEPS    # number of sampling timesteps (using ddim for faster inference [see ddim paper])
)#.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model_epoch_1.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Pre-trained model loaded from {model_path}.")
else:
    print("No pre-trained model found. Training from scratch.")
    train_losses = []
    val_losses = []

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch [{epoch+1}/{N_EPOCHS}]")
        for batch_idx, (data, _) in progress_bar:
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optim.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_losses.append(epoch_loss / len(loader))
        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Training Loss: {epoch_loss/len(loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                loss = diffusion(data)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Validation Loss: {val_loss:.4f}")

        # Save the model after each epoch
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Save training metrics
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(os.path.join(model_dir, "training_metrics.pickle"), "wb") as f:
        pickle.dump(metrics, f)

    # Plot training and validation losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(model_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")

model.eval()
sampled_images = diffusion.sample(batch_size=10)
print("Sampled images generated.")

output_dir = "./sampled_images"
os.makedirs(output_dir, exist_ok=True)
grid_path = os.path.join(output_dir, "sampled_grid.png")
torchvision.utils.save_image(sampled_images, grid_path, nrow=5, normalize=True)
print(f"Sampled image grid saved to {grid_path}.")
