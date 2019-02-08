import torch
import torch.nn as nn
import torch.nn.functional as F


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
