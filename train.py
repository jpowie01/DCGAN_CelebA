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


EPOCHS = 1000
DISCRIMINATOR_TRAINING_ITERATIONS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.0002
DATALOADER_WORKERS = 8
WEIGHTS_DIR = './output/'
DATASET_DIR = './dataset/'

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Prepare directory for model weights
os.makedirs(DATASET_DIR, exist_ok=True)

# Prepare dataset for training
train_dataset, train_loader = dataset.get(BATCH_SIZE, DATASET_DIR, DATALOADER_WORKERS)

# Prepare our networks for the training process
generator = models.Generator()
generator.apply(models.Generator.weights_init)
discriminator = models.Discriminator()
discriminator.apply(models.Discriminator.weights_init)

# Move models to the GPU
if USE_GPU:
    generator.cuda()
    discriminator.cuda()

# Prepare optimizers
criterion = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

for epoch in range(EPOCHS):
    batch = 0
    total_discriminator_loss = 0.0
    total_discriminator_ground_truth_accuracy = 0.0
    total_discriminator_generated_accuracy = 0.0
    total_generator_loss = 0.0
    train_loader_iterator = iter(train_loader)

    with tqdm.tqdm(total=len(train_loader), desc=f'Epoch #{(epoch+1):02}/{EPOCHS:02}') as progress_bar:
        end_of_training = False
        while not end_of_training:
            # Train discriminator first
            for k in range(DISCRIMINATOR_TRAINING_ITERATIONS):
                discriminator_optimizer.zero_grad()

                # Prepare batch of ground truth images
                batch_of_images, _ = next(train_loader_iterator, (None, None))
                if batch_of_images is None:
                    end_of_training = True
                    break  # Now train the discriminator
                current_batch_size = len(batch_of_images)
                discriminated_ground_truth_images = discriminator(batch_of_images)  # TODO: This should probably be changed..

		# Sample random noise and pass it to generate images
                z = torch.normal(torch.zeros(current_batch_size, 100), torch.ones(current_batch_size, 100))
                discriminated_generated_images = discriminator(generator(z))

                # Calculate loss
                ground_truth = torch.ones(current_batch_size, 1) * 0.1
                ground_truth_loss = criterion(discriminated_ground_truth_images, ground_truth)
                generated = torch.ones(current_batch_size, 1) * 0.9
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
            z = torch.normal(torch.zeros(BATCH_SIZE, 100), torch.ones(BATCH_SIZE, 100))
            generated_images = generator(z)
            
            # Check how much the generator has fooled the discriminator
            ground_truth = torch.ones(BATCH_SIZE, 1) * 0.1
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
    print(f'Epoch #{(epoch+1):02}/{EPOCHS:02}: Train D-Loss: {total_discriminator_loss:.4f} '
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
    }, f'{DATASET_DIR}weights_e{(epoch+1):02}_{total_discriminator_loss:.4f}_{total_generator_loss:.4f}.torch')
