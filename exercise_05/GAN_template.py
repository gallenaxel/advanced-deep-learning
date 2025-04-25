import os
import pickle

from dataclasses import dataclass

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
# FOR TENSOR BOARD VISUALIZATION
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import matplotlib.pyplot as plt


from models import Generator, Discriminator

# Hyperparameters
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

latent_dimension = 64 # 64, 128, 256
# for simplicity we will flatten the image to a vector and to use simple MLP networks
# 28 * 28 * 1 flattens to 784
# you are also free to use CNNs
image_dimension = 28 * 28 * 1  # 784

# we define a tranform that converts the image to tensor and normalizes it with mean and std of 0.5
# which will convert the image range from [0, 1] to [-1, 1]
myTransforms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="../datasets/MNIST_transformed/",
                         transform=myTransforms,
                         download=True,
                         )
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=32, shuffle=True)

fixed_noise = torch.randn(32, latent_dimension).to(device)

@dataclass
class GANConfig:
    device: str
    image_dimension: int = 28 * 28 * 1
    lr: float = 3e-4
    batchSize: int = 32  # Batch size
    numEpochs: int = 100
    logStep: int = 625
    project_dir: str = "dev/"
    criterion: callable = nn.BCELoss()

    def __post_init__(self):
        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)

class GAN:
    def __init__(self, generator: nn.Module,
                 discriminator: nn.Module,
                 config: GANConfig,
                 ):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=config.lr)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=config.lr)
        self.current_epoch = 0
        self.step = 0
        self.disc_losses = []
        self.gen_losses = []
        self.lrs = []
        self.current_epoch_discriminator_loss = 0
        self.current_epoch_generator_loss = 0

    def update_generator(self, classification_score, real):
        batch_size = real.shape[0]
        real_labels = torch.ones(batch_size, 1, device=device)
        loss_generator = self.config.criterion(classification_score, real_labels)

        # Update the weights of the generator
        self.opt_generator.zero_grad()
        loss_generator.backward()
        self.opt_generator.step()

        return loss_generator

    def update_discriminator(self, generator_output, real):
        batch_size = real.shape[0]
        real_labels = torch.ones(batch_size, 1, device=device)
        output_real = self.discriminator(real)
        loss_real = self.config.criterion(output_real, real_labels)

        # - predict the discriminator output for fake images
        fake_labels = torch.zeros(batch_size, 1, device=device)
        output_fake = self.discriminator(generator_output.detach())
        loss_fake = self.config.criterion(output_fake, fake_labels)

        # - calculate the total loss for the discriminator
        loss_discriminator = (loss_real + loss_fake) / 2
        # - now update the weights of the discriminator by backpropagating the loss
        self.opt_discriminator.zero_grad()
        loss_discriminator.backward()
        self.opt_discriminator.step()

        return loss_discriminator

    def run_batch(self, real, batch_idx=None, epoch=None):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Step 1) generate fake images
        noise = torch.randn(batch_size, latent_dimension).to(device)
        fake = self.generator(noise)

        # - calculate the total loss for the discriminator
        loss_discriminator = self.update_discriminator(fake, real)

        # Train Generator:
        # Pass the fake images through the discriminator
        generator_output_classification_score = self.discriminator(fake)

        # Calculate the loss for the generator
        # We want the discriminator to classify the fake images as real (label = 1)
        loss_generator = self.update_generator(generator_output_classification_score, real)

        self.current_epoch_discriminator_loss += loss_discriminator.item()
        self.current_epoch_generator_loss += loss_generator.item()

        # print the progress
        print(f"\r Epoch [{epoch}/{self.config.numEpochs}] Batch {batch_idx}/{len(loader)} | Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}", end="")
        if batch_idx % self.config.logStep == 0:
            self.run_logStep(loss_discriminator, loss_generator)

    def end_epoch(self, epoch):
        self.disc_losses.append(self.current_epoch_discriminator_loss / len(loader))
        self.gen_losses.append(self.current_epoch_generator_loss / len(loader))
        self.current_epoch_discriminator_loss = 0
        self.current_epoch_generator_loss = 0
        training_metrics = {
            "disc_loss": self.disc_losses,
            "gen_loss": self.gen_losses,
            "lrs": self.lrs,
        }
        with open(f"{self.config.project_dir}/training_metrics.pickle", 'wb') as f:
            pickle.dump(training_metrics, f)

        # Plot discriminator and generator losses separately
        plt.figure(figsize=(10, 5))
        plt.plot(self.disc_losses, label="Discriminator Loss")
        plt.plot(self.gen_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.config.project_dir}/training_performance.png")
        plt.xlim(0, self.config.numEpochs+1)
        plt.savefig(f"{self.config.project_dir}/loss_animation/training_performance_animation_{epoch}.png")
        plt.close()

    def run_logStep(self, loss_discriminator, loss_generator):
        with torch.no_grad():
            # Generate noise via Generator, we always use the same noise to see the progression
            # Generate fixed noise for consistent visualization
            fake = self.generator(fixed_noise).reshape(-1, 1, 28, 28)
            # Get real data
            # make grid of pictures and add to tensorboard
            imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
            torchvision.utils.save_image(fake, f"{self.config.project_dir}/example_images/step{self.step}.png")

            # Initialize SummaryWriter for TensorBoard
            writer = SummaryWriter(log_dir=os.path.join(self.config.project_dir, "tb_logs"))

            # Add images to TensorBoard
            writer.add_image("Fake Images", imgGridFake, global_step=self.step)

            # Add losses to TensorBoard
            writer.add_scalar("Loss/Discriminator", loss_discriminator.item(), global_step=self.step)
            writer.add_scalar("Loss/Generator", loss_generator.item(), global_step=self.step)

            # Close the writer
            writer.close()

            # increment step
            self.step += 1

    def generate_sample_grid(self, name):
        noise = torch.randn(32, latent_dimension).to(device)
        fake = self.generator(noise).reshape(-1, 1, 28, 28)
        # Get real data
        # make grid of pictures and add to tensorboard
        imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
        torchvision.utils.save_image(fake, f"{self.config.project_dir}/final_example_{name}.png")


def main():
    config = GANConfig(device)
    gan = GAN(
              Generator(image_dimension, latent_dimension).to(device),
              Discriminator(image_dimension).to(device),
              config,
              )
    for epoch in range(config.numEpochs):
        for batch_idx, (real, _) in enumerate(loader):
            gan.run_batch(real, batch_idx, epoch)
        gan.end_epoch(epoch)

    for i in range(5):
        gan.generate_sample_grid(i)

if __name__ == "__main__":
    main()