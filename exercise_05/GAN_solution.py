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

# Hyperparameters
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
batchSize = 32  # Batch size
numEpochs = 100
logStep = 625  # the number of steps to log the images and losses to tensorboard

latent_dimension = 128 # 64, 128, 256
image_dimension = 28 * 28 * 1  # 784

# we define a tranform that converts the image to tensor and normalizes it with mean and std of 0.5
# which will convert the image range from [0, 1] to [-1, 1]
myTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

class Generator(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super().__init__()
        hiddenDim = 256
        self.gen = nn.Sequential(
            nn.Linear(latent_dimension, hiddenDim),
            nn.LeakyReLU(0.01),
            nn.Linear(hiddenDim, image_dimension),
            nn.Tanh(),  # We normalize inputs to [-1, 1] to make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)



class Discriminator(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self):
        super().__init__()
        hiddenDim = 256
        self.disc = nn.Sequential(
            nn.Linear(image_dimension, hiddenDim),
            nn.LeakyReLU(0.01),
            nn.Linear(hiddenDim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

# initialize networks and optimizers
discriminator = Discriminator().to(device)
generator = Generator().to(device)
opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
opt_generator = optim.Adam(generator.parameters(), lr=lr)

# This is a binary classification task, so we use Binary Cross Entropy Loss
criterion = nn.BCELoss()

# generate one batch of random noise that we'll use to visualize the progression of the generator
fixed_noise = torch.randn((batchSize, latent_dimension)).to(device)

# Initialize the Tensorboard writer
writerFake = SummaryWriter("logs/fake")
writerReal = SummaryWriter("logs/real")
writerLoss = SummaryWriter("logs/loss")

# Training Loop

step = 0
print("Started Training and visualization...")
for epoch in range(numEpochs):
    # loop over batches
    print()
    for batch_idx, (real, _) in enumerate(loader):
        # First we train the discriminator on real images vs. generated images

        # Get the real images and flatten them
        # for simplicity, we flatten the image to a vector and to use simple MLP networks
        # 28 * 28 * 1 flattens to 784
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # generate fake images
        noise = torch.randn(batch_size, latent_dimension).to(device)
        fake = generator(noise)

        # Train Discriminator:
        disc_real = discriminator(real) # predict the discriminator output for real images
        labels_real = torch.ones_like(disc_real) # real images are labeled as 1
        loss_discriminator_real = criterion(disc_real, labels_real) # calculate the loss for real images

        disc_fake = discriminator(fake) # predict the discriminator output for fake images
        labels_fake = torch.zeros_like(disc_fake) # fake images are labeled as 0
        loss_discriminator_fake = criterion(disc_fake, labels_fake) # calculate the loss for fake images

        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2 # average the loss for real and fake images

        # now we upadate the weights of the discriminator by backpropagating the loss
        # through the discriminator and we use the optimizer to update the weights
        # the generator is not updated in this step
        discriminator.zero_grad()
        loss_discriminator.backward(retain_graph=True)
        opt_discriminator.step()

        # Train Generator:
        # we generate fake images and pass them through the discriminator
        # we do a little trick and modify the original objective function of 
        # minimizing the probabiuity of the discriminator predicting the fake images as fake
        # to maximizing the probability of the discriminator predicting the fake images as real
        # this leads to a faster training of the generator when it does not represent the real data well
        # this is a common trick in GANs
        # for moer information see section 17.1.2 of the book Deep Learning by Bishop and Bishop
        output = discriminator(fake)
        labels_real = torch.ones_like(disc_real) # real images are labeled as 1
        loss_generator = criterion(output, labels_real)
        generator.zero_grad()
        loss_generator.backward()
        opt_generator.step() # we only update the weights of the generator

        print(f"\rEpoch [{epoch}/{numEpochs}] Batch {batch_idx}/{len(loader)} \ Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}", end="")

        # Log the losses and example images to tensorboard
        if batch_idx % logStep == 0:
            with torch.no_grad():
                # Generate noise via Generator, we always use the same noise to see the progression
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                # Get real data
                data = real.reshape(-1, 1, 28, 28)
                # make grid of pictures and add to tensorboard
                imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                imgGridReal = torchvision.utils.make_grid(data, normalize=True)

                writerFake.add_image("Mnist Fake Images", imgGridFake, global_step=step)
                writerReal.add_image("Mnist Real Images", imgGridReal, global_step=step)

                writerLoss.add_scalar("Loss Discriminator", loss_discriminator, global_step=step)
                writerLoss.add_scalar("Loss Generator", loss_generator, global_step=step)

                # increment step
                step += 1