import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_layer=[16,16,2], tagset_size=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.tagset_size = tagset_size

        self.hidden1 = nn.Linear(self.input_dim, self.hidden_layer[0])
        self.hidden2 = nn.Linear(self.hidden_layer[0], self.hidden_layer[1])
        self.hidden3 = nn.Linear(self.hidden_layer[1], self.hidden_layer[2])
        self.out = nn.Linear(self.hidden_layer[2], self.tagset_size)

    def forward(self, input):
        hidden = self.hidden1(input)
        hidden = F.leaky_relu(hidden, 0.2)
        hidden = self.hidden2(hidden)
        hidden = F.leaky_relu(hidden, 0.2)
        hidden = self.hidden3(hidden)
        hidden = F.leaky_relu(hidden, 0.2)
        out = self.out(hidden)
        out = F.softmax(out)
        return out


if __name__ == '__main__':
    discriminator = Discriminator()
    points = torch.rand(10, 2)
    print(discriminator(points))
