import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, noise_dim=2, hidden_layer=[16, 16], output_dim=2):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_layer = hidden_layer
        self.output_dim = output_dim

        self.hidden1 = nn.Linear(self.noise_dim, self.hidden_layer[0])
        self.hidden2 = nn.Linear(self.hidden_layer[0], self.hidden_layer[1])
        self.out = nn.Linear(self.hidden_layer[1], self.output_dim)

    def forward(self, noise):
        hidden = self.hidden1(noise)
        hidden = F.leaky_relu(hidden, 0.2)
        hidden = self.hidden2(hidden)
        hidden = F.leaky_relu(hidden, 0.2)
        out = self.out(hidden)
        return out


if __name__ == '__main__':
    generator = Generator()
    noise = torch.rand(10, 1)
    generator(noise)
