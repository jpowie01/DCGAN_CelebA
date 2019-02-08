import argparse
import os

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import dataset
import models


# Parse configuration passed from the CLI
parser = argparse.ArgumentParser(description='Training DCGAN on CelebA dataset')
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs')
parser.add_argument('--discriminator-training-iterations', '-k', type=int, default=1,
                    help='Number of iterations for discriminator training (K parameter)')
parser.add_argument('--random-label-swap', '-s', type=float, default=0.05, help='Percentage of labels that will be swapped')
parser.add_argument('--batch-size', '-b', type=int, default=256, help='Size of a single batch')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.0002, help='Learning rate for Adam')
parser.add_argument('--dataloader-workers', '-w', type=int, default=8, help='Number of threads used by Data Loader')
parser.add_argument('--checkpoints-directory', '-c', type=str, default='./checkpoints/',
                    help='Directory where all models checkpoints will be collected')
parser.add_argument('--dataset', '-d', type=str, default='./dataset/', help='Directory where dataset will be stored')
parser.add_argument('--use-gpu', '-g', type=bool, default=torch.cuda.is_available(), help='Use GPU if True')
configuration = parser.parse_args()

# Prepare placeholder for device used by this script (CPU or GPU)
DEVICE = torch.device('cuda' if configuration.use_gpu else 'cpu')

# Prepare directory for models checkpoints
os.makedirs(configuration.checkpoints_directory, exist_ok=True)

# Prepare dataset for training
train_dataset, train_loader = dataset.get(configuration.batch_size, configuration.dataset, configuration.dataloader_workers)
number_of_training_examples = len(train_dataset)

# Prepare our networks for the training process
generator = models.Generator()
generator.apply(models.Generator.weights_init)
discriminator = models.Discriminator()
discriminator.apply(models.Discriminator.weights_init)

# Move models to the GPU if needed
generator.to(DEVICE)
discriminator.to(DEVICE)

# Prepare optimizers
criterion = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=configuration.learning_rate, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=configuration.learning_rate, betas=(0.5, 0.999))

# Let's train our GAN!
for epoch in range(configuration.epochs):
    batch = 0
    total_discriminator_loss = 0.0
    total_discriminator_ground_truth_accuracy = 0.0
    total_discriminator_generated_accuracy = 0.0
    total_generator_loss = 0.0
    train_loader_iterator = iter(train_loader)

    with tqdm.tqdm(total=len(train_loader), desc=f'Epoch #{(epoch+1):02}/{configuration.epochs:02}') as progress_bar:
        end_of_training = False
        while not end_of_training:
            # Train discriminator first
            for k in range(configuration.discriminator_training_iterations):
                discriminator_optimizer.zero_grad()

                # Prepare batch of ground truth images
                batch_of_images, _ = next(train_loader_iterator, (None, None))
                if batch_of_images is None:
                    end_of_training = True
                    break  # Now train the discriminator
                current_batch_size = len(batch_of_images)
                discriminated_ground_truth_images = discriminator(batch_of_images.to(DEVICE))

		# Sample random noise and pass it to generate images
                z = torch.normal(torch.zeros(current_batch_size, 100), torch.ones(current_batch_size, 100)).to(DEVICE)
                discriminated_generated_images = discriminator(generator(z))

                # Calculate loss
                ground_truth = torch.ones(current_batch_size, 1).to(DEVICE) * 0.1
                ground_truth[torch.rand(current_batch_size) < configuration.random_label_swap] = 0.9  # Some of labels will be swapped
                ground_truth_loss = criterion(discriminated_ground_truth_images, ground_truth)
                generated = torch.ones(current_batch_size, 1).to(DEVICE) * 0.9
                generated[torch.rand(current_batch_size) < configuration.random_label_swap] = 0.1  # Some of labels will be swapped
                generated_loss = criterion(discriminated_generated_images, generated)
                discriminator_loss = ground_truth_loss + generated_loss
                total_discriminator_ground_truth_accuracy += torch.sum(discriminated_ground_truth_images < 0.5).item()
                total_discriminator_generated_accuracy += torch.sum(discriminated_generated_images > 0.5).item()
                total_discriminator_loss += discriminator_loss.item()

                # Update the discriminator
                discriminator_loss.backward()
                discriminator_optimizer.step()

            # Now, train generator
            generator_optimizer.zero_grad()

            # Sample random noise and pass it to generate images
            z = torch.normal(torch.zeros(configuration.batch_size, 100), torch.ones(configuration.batch_size, 100)).to(DEVICE)
            generated_images = generator(z)
            
            # Check how much the generator has fooled the discriminator
            ground_truth = torch.ones(configuration.batch_size, 1).to(DEVICE) * 0.1
            ground_truth[torch.rand(configuration.batch_size) < configuration.random_label_swap] = 0.9  # Some of labels will be swapped
            generator_loss = criterion(discriminator(generated_images), ground_truth)
            total_generator_loss += generator_loss.item()

            # Update the generator
            generator_loss.backward()
            generator_optimizer.step()

            # Update progress bar
            batch += 1
            if batch % 10 == 0:
                progress_bar.update(10)
                progress_bar.set_postfix(g_loss=f'{generator_loss.item():.4f}', d_loss=f'{discriminator_loss.item():.4f}')

    # Calculate mean discriminator loss over all K iterations
    total_discriminator_loss /= len(train_loader)
    total_discriminator_ground_truth_accuracy /= number_of_training_examples
    total_discriminator_generated_accuracy /= number_of_training_examples
    total_generator_loss /= len(train_loader)
    print(f'Epoch #{(epoch+1):02}/{configuration.epochs:02}: Train D-Loss: {total_discriminator_loss:.4f} '
          f'(GT Acc: {total_discriminator_ground_truth_accuracy*100:.4f}%, Gen Acc: {total_discriminator_generated_accuracy*100:.4f}%) '
          f'Train G-Loss: {total_generator_loss:.4f}\n')

    torch.save({
        'epoch': epoch,
        'generator_model_state_dict': generator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_model_state_dict': discriminator.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        'generator_loss': generator_loss,
        'discriminator_loss': discriminator_loss,
    }, f'{configuration.checkpoints_directory}checkpoint_e{(epoch+1):02}_{total_discriminator_loss:.4f}_{total_generator_loss:.4f}.torch')
