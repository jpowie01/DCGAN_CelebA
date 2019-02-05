import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


EPOCHS = 50
DISCRIMINATOR_TRAINING_ITERATIONS = 1
BATCH_SIZE = 32
DATALOADER_WORKERS = 4
TRAIN_VALIDATION_SPLIT = 0.2  # 20% for validation

# Prepare dataset for training
train_transformation = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = torchvision.datasets.CIFAR10(root='./dataset/', train=True,
                                             download=True, transform=train_transformation)

# Prepare dataset for validation
validation_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
validation_dataset = torchvision.datasets.CIFAR10(root='./dataset/', train=True,
                                                  download=True, transform=validation_transformation)

# Shuffle examples and prepare split point
number_of_training_examples = len(train_dataset)
training_indices = list(range(number_of_training_examples))
np.random.shuffle(training_indices)
split_point = int(np.floor(TRAIN_VALIDATION_SPLIT * number_of_training_examples))

# Split training dataset into real training and validation
# training_idx, validation_idx = training_indices[split_point:], training_indices[:split_point]
training_idx, validation_idx = training_indices[0:128], training_indices[32:64]
training_sampler = torch.utils.data.SubsetRandomSampler(training_idx)
validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)

# Prepare Data Loaders for training and validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=training_sampler,
                                           pin_memory=True, num_workers=DATALOADER_WORKERS)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler,
                                                pin_memory=True, num_workers=DATALOADER_WORKERS)

# Let's define our Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.projection = nn.Linear(100, 512*4*4)
        self.layers = nn.Sequential(
            # Before the convolution part
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # First block
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Second block
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Third block
            nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh(),
        )
        
    def forward(self, random_noise):
        x = self.projection(random_noise)
        x = x.view(-1, 512, 4, 4)
        return self.layers(x)

    @staticmethod
    def weights_init(layer):
        layer_class_name = layer.__class__.__name__
        if 'Conv' in layer_class_name:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in layer_class_name:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)


# Let's define our Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2),

            # Second block
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # Third block
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.output = nn.Linear(128*2*2, 1)
        self.output_function = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 128*2*2)
        x = self.output(x)
        return self.output_function(x)

    @staticmethod
    def weights_init(layer):
        layer_class_name = layer.__class__.__name__
        if 'Conv' in layer_class_name:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in layer_class_name:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)


# Prepare our networks for the training process
generator = Generator()
generator.apply(Generator.weights_init)
discriminator = Discriminator()
discriminator.apply(Discriminator.weights_init)

criterion = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(EPOCHS):
    total_discriminator_loss = 0.0
    total_generator_loss = 0.0
    train_loader_iterator = iter(train_loader)

    with tqdm.tqdm(total=len(train_loader), desc=f'Epoch #{(epoch+1):02}') as progress_bar:
        end_of_training = False
        while not end_of_training:
            # Train discriminator first
            for k in range(DISCRIMINATOR_TRAINING_ITERATIONS):
                discriminator_optimizer.zero_grad()

                # Sample random noise and pass it to generate images
                z = torch.normal(torch.zeros(BATCH_SIZE, 100), torch.ones(BATCH_SIZE, 100))
                discriminated_generated_images = discriminator(generator(z))
                batch_of_images, _ = next(train_loader_iterator, (None, None))
                if batch_of_images is None:
                    end_of_training = True
                    break  # Now train the discriminator
                discriminated_ground_truth_images = discriminator(batch_of_images)  # TODO: This should probably be changed...

                # Calculate loss
                ground_truth = torch.ones(BATCH_SIZE, 1)
                ground_truth_loss = criterion(discriminated_ground_truth_images, ground_truth)
                generated = torch.zeros(BATCH_SIZE, 1)
                generated_loss = criterion(discriminated_generated_images, generated)  # TODO: Shouldn't it be swapped?
                discriminator_loss = ground_truth_loss + generated_loss
                total_discriminator_loss += discriminator_loss.item()

                # Update the discriminator
                discriminator_loss.backward()
                discriminator_optimizer.step()

            # Now, train generator
            generator_optimizer.zero_grad()

            # Sample random noise and pass it to generate images
            z = torch.normal(torch.zeros(BATCH_SIZE, 100), torch.ones(BATCH_SIZE, 100))
            generated_images = generator(z)
            
            # Check how much the generator has fooled the discriminator
            ground_truth = torch.ones(BATCH_SIZE, 1)
            generator_loss = criterion(discriminator(generated_images), ground_truth)
            total_generator_loss += generator_loss.item()

            # Update the generator
            generator_loss.backward()
            generator_optimizer.step()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(g_loss=generator_loss.item(), d_loss=discriminator_loss.item())

    # Calculate mean discriminator loss over all K iterations
    total_discriminator_loss /= len(train_loader)
    total_discriminator_loss /= DISCRIMINATOR_TRAINING_ITERATIONS
    total_generator_loss /= len(train_loader)
    print(f'Epoch #{(epoch+1):02}: Train D-Loss: {total_discriminator_loss:.4f} Train G-Loss: {total_generator_loss:.4f}\n')

