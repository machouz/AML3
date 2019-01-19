import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, noise_dim=1, hidden_layer=64, output_dim=2):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_layer = hidden_layer
        self.output_dim = output_dim

        self.hidden = nn.Sequential(
            nn.Linear(self.noise_dim, self.hidden_layer),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(self.hidden_layer, self.output_dim),
            nn.ReLU()
        )

    def forward(self, noise):
        hidden = self.hidden(noise)
        out = self.out(hidden)
        return out


if __name__ == '__main__':
    generator = Generator()
    noise = torch.rand(10, 1)
    generator(noise)
