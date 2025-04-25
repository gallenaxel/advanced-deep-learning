import os
import pickle

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
lr = 3e-4
batchSize = 32  # Batch size
numEpochs = 100
logStep = 625  # the number of steps to log the images and losses to tensorboard

project_dir = "dev/"
# Create the directory if it does not exist
if not os.path.exists(project_dir):
    os.makedirs(project_dir)
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
loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

fixed_noise = torch.randn(32, latent_dimension).to(device)


def main():
    # initialize networks and optimizers
    discriminator = Discriminator(image_dimension).to(device)
    generator = Generator(image_dimension, latent_dimension).to(device)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
    opt_generator = optim.Adam(generator.parameters(), lr=lr)

    # This is a binary classification task, so we use Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    disc_losses = []
    gen_losses = []
    lrs = []

    # Training Loop
    step = 0
    print("Started Training and visualization...")
    for epoch in range(numEpochs):
        # loop over batches
        epoch_discriminator_loss = 0
        epoch_generator_loss = 0
        for batch_idx, (real, _) in enumerate(loader):
            # First we train the discriminator on real images vs. generated images

            # Get the real images and flatten them
            # for simplicity, we flatten the image to a vector and to use simple MLP networks
            # 28 * 28 * 1 flattens to 784
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # Step 1) generate fake images
            noise = torch.randn(batch_size, latent_dimension).to(device)
            fake = generator(noise)

            # Step 2) Train Discriminator:
            # - predict the discriminator output for real images
            real_labels = torch.ones(batch_size, 1, device=device)
            output_real = discriminator(real)
            loss_real = criterion(output_real, real_labels)

            # - predict the discriminator output for fake images
            fake_labels = torch.zeros(batch_size, 1, device=device)
            output_fake = discriminator(fake.detach())
            loss_fake = criterion(output_fake, fake_labels)

            # - calculate the total loss for the discriminator
            loss_discriminator = (loss_real + loss_fake) / 2

            # - now update the weights of the discriminator by backpropagating the loss
            opt_discriminator.zero_grad()
            loss_discriminator.backward()
            opt_discriminator.step()

            # Train Generator:
            # Pass the fake images through the discriminator
            output_fake = discriminator(fake)

            # Calculate the loss for the generator
            # We want the discriminator to classify the fake images as real (label = 1)
            loss_generator = criterion(output_fake, real_labels)

            # Update the weights of the generator
            opt_generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()

            epoch_discriminator_loss += loss_discriminator.item()
            epoch_generator_loss += loss_generator.item()

            # print the progress
            print(f"\r Epoch [{epoch}/{numEpochs}] Batch {batch_idx}/{len(loader)} | Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}", end="")

            # Log the losses and example images to tensorboard
            if batch_idx % logStep == 0:
                with torch.no_grad():
                    # Generate noise via Generator, we always use the same noise to see the progression
                    # Generate fixed noise for consistent visualization
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    # Get real data
                    data = real.reshape(-1, 1, 28, 28)
                    # make grid of pictures and add to tensorboard
                    imgGridFake = torchvision.utils.make_grid(fake, normalize=True)

                    torchvision.utils.save_image(fake, f"{project_dir}/example_images/step{step}.png")

                    imgGridReal = torchvision.utils.make_grid(data, normalize=True)

                    # Initialize SummaryWriter for TensorBoard
                    writer = SummaryWriter(log_dir=os.path.join(project_dir, "tb_logs"))

                    # Add images to TensorBoard
                    writer.add_image("Fake Images", imgGridFake, global_step=step)
                    writer.add_image("Real Images", imgGridReal, global_step=step)

                    # Add losses to TensorBoard
                    writer.add_scalar("Loss/Discriminator", loss_discriminator.item(), global_step=step)
                    writer.add_scalar("Loss/Generator", loss_generator.item(), global_step=step)



                    # Close the writer
                    writer.close()

                    # increment step
                    step += 1

        disc_losses.append(epoch_discriminator_loss / len(loader))
        gen_losses.append(epoch_generator_loss / len(loader))
        training_metrics = {
            "disc_loss": disc_losses,
            "gen_loss": gen_losses,
            "lrs": lrs,
        }
        with open(f"{project_dir}/training_metrics.pickle", 'wb') as f:
            pickle.dump(training_metrics, f)

        # Plot discriminator and generator losses separately
        plt.figure(figsize=(10, 5))
        plt.plot(disc_losses, label="Discriminator Loss")
        plt.plot(gen_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{project_dir}/training_performance.png")
        plt.xlim(0, numEpochs+1)
        plt.savefig(f"{project_dir}/loss_animation/training_performance_animation_{epoch}.png")
        plt.close()

if __name__ == "__main__":
    main()