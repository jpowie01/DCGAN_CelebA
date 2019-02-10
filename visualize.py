import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
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
parser.add_argument('--output', '-o', type=str, default='figure', help='Output figure filename (without extension)')
configuration = parser.parse_args()

# Prepare placeholder for device used by this script (CPU or GPU)
DEVICE = torch.device('cuda' if configuration.use_gpu.lower() == 'true' else 'cpu')

# Fetch dataset for comparison
train_dataset, train_loader = dataset.get_celeba(WIDTH*HEIGHT, configuration.dataset, 1)

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
generated_images = generated_images.detach().numpy() * 0.5 + 0.5
generated_images = np.transpose(generated_images * 255.0, (0, 2, 3, 1)).astype(np.uint8)

# Fetch batch of original images
original_images, _ = next(iter(train_loader))
original_real_or_fake = discriminator(original_images)

# Preprocess original images
original_images = original_images.detach().numpy() * 0.5 + 0.5
original_images = np.transpose(original_images * 255.0, (0, 2, 3, 1)).astype(np.uint8)

# Prepare outer grid on the plot
fig = plt.figure(figsize=(10, 7.5))
fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=None, hspace=None)
outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.15)

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
fig.savefig(f'{configuration.output}.png', format='png', dpi=300)

# Let's prepare a GIF that will transform one face to another
fig = plt.figure()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig.set_size_inches(1, 1, forward=True)
frames = []

# We will transform one latent space to another, so let's prepare first Z and calculate
# difference to the second Z vector
z = torch.normal(torch.zeros(1, 100), torch.ones(1, 100)).to(DEVICE)
diff = torch.normal(torch.zeros(1, 100), torch.ones(1, 100)).to(DEVICE) - z

# Prepare the animation by passing modified Z vector through generator
FRAMES = 600
SINGLE_MANIPULATION = 60
for frame in range(FRAMES):
    # Generate image
    generated_images = generator(z)
    generated_images = generated_images.detach().numpy() * 0.5 + 0.5
    generated_images = np.transpose(generated_images * 255.0, (0, 2, 3, 1)).astype(np.uint8)
    
    # Render it on the plot
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    image = plt.imshow(generated_images[0], animated=True)
    frames.append([image])

    # Find new destination if needed and slightly go into this destination
    if (frame + 1) % SINGLE_MANIPULATION == 0:
        diff = torch.normal(torch.zeros(1, 100), torch.ones(1, 100)).to(DEVICE) - z
    z += 1 / SINGLE_MANIPULATION * diff

# Save the GIF!
gif = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
gif.save(f'{configuration.output}_manipulation.gif')
