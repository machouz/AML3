import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_layer=64, tagset_size=1):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.tagset_size = tagset_size

        self.hidden = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_layer),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(self.hidden_layer, self.tagset_size),
            nn.Sigmoid()
        )

    def forward(self, input):
        hidden = self.hidden(input)
        out = self.out(hidden)
        return out


if __name__ == '__main__':
    discriminator = Discriminator()
    points = torch.rand(10, 2)
    print(discriminator(points))
