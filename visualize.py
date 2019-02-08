import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

import dataset
import models

WIDTH = 8
HEIGHT = 2

# Parse configuration passed from the CLI
parser = argparse.ArgumentParser(description='Visualize DCGAN on CelebA dataset')
parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Checkpoint which should be used for visualization')
parser.add_argument('--dataset', '-d', type=str, default='./dataset/', help='Directory where dataset will be stored')
parser.add_argument('--use-gpu', '-g', type=str, default=str(torch.cuda.is_available()), help='Use GPU if True')
parser.add_argument('--output-figure', '-o', type=str, default='figure', help='Output figure filename (without extension)')
configuration = parser.parse_args()

# Prepare placeholder for device used by this script (CPU or GPU)
DEVICE = torch.device('cuda' if configuration.use_gpu.lower() == 'true' else 'cpu')

# Fetch dataset for comparison
train_dataset, train_loader = dataset.get(WIDTH*HEIGHT, configuration.dataset, 1)

# Load checkpoint
checkpoint = torch.load(configuration.checkpoint, map_location=torch.device(DEVICE))

# Load models from checkpoint
generator = models.Generator()
discriminator = models.Discriminator()
generator.load_state_dict(checkpoint['generator_model_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_model_state_dict'])

# Move models to the GPU if needed
generator.to(DEVICE)
discriminator.to(DEVICE)

# Generate some images
z = torch.normal(torch.zeros(WIDTH*HEIGHT, 100), torch.ones(WIDTH*HEIGHT, 100)).to(DEVICE)
generated_images = generator(z)
generated_real_or_fake = discriminator(generated_images)

# Preprocess generated images
generated_images = generated_images.detach().numpy()
images = generated_images - generated_images.min(axis=0)
images /= generated_images.max(axis=0) - generated_images.min(axis=0)
generated_images = np.transpose(images * 255.0, (0, 2, 3, 1)).astype(np.uint8)

# Fetch batch of original images
original_images, _ = next(iter(train_loader))
original_real_or_fake = discriminator(original_images)

# Preprocess original images
original_images = original_images.detach().numpy()
images = original_images - original_images.min(axis=0)
images /= original_images.max(axis=0) - original_images.min(axis=0)
original_images = np.transpose(images * 255.0, (0, 2, 3, 1)).astype(np.uint8)

# Prepare outer grid on the plot
fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)

# Plot all generated images
ax = fig.add_subplot(outer[0])
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
plt.setp(ax, title='Generated Images')
inner = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
for i in range(2*8):
    ax = plt.Subplot(fig, inner[i])
    ax.imshow(generated_images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax, title='Fake' if generated_real_or_fake[i] > 0.5 else 'Real')
    fig.add_subplot(ax)

# Plot all original images
ax = fig.add_subplot(outer[1])
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
plt.setp(ax, title='Original Images')
inner = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
for i in range(2*8):
    ax = plt.Subplot(fig, inner[i])
    ax.imshow(original_images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax, title='Fake' if original_real_or_fake[i] > 0.5 else 'Real')
    fig.add_subplot(ax)

# Save the figure
fig.savefig(f'{configuration.output_figure}.png', format='png', dpi=300)
