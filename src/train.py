import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch.nn.functional as F
from generator import Generator
from generate_data import get_data
from discriminator import Discriminator
from utils import *

NOISE_DIM = 2
GEN_LR = 0.001
DIS_LR = 0.001
DATA = 'par'


def train_discriminator(example, discriminator, loss_fun, target):
    out = discriminator(example)
    loss = loss_fun(out, target)
    loss.backward()


def train(n_samples, data_type, generator, discriminator, gen_optimizer, dis_optimizer, loss_fun):
    generator.train()
    discriminator.train()
    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()

    # Train discriminator on real
    real_ex = get_data(n_points=n_samples, data_type=data_type)
    real_ex = torch.tensor(real_ex, dtype=torch.float)

    dis_real = discriminator(real_ex)
    dis_loss_real = loss_fun(dis_real, torch.ones(n_samples, 1))
    dis_loss_real.backward()

    # Train discriminator on fake
    noise = torch.rand(n_samples, NOISE_DIM)
    gen_out = generator(noise)

    dis_gen = discriminator(gen_out.detach())
    dis_loss_gen = loss_fun(dis_gen, torch.zeros(n_samples, 1))
    dis_loss_gen.backward()
    print("Discriminator loss : {}".format(dis_loss_gen.item()))
    dis_optimizer.step()

    # Train generator
    dis_gen = discriminator(gen_out)
    loss_gen = loss_fun(dis_gen, torch.ones(n_samples, 1))
    loss_gen.backward()
    print("Generator loss : {}".format(loss_gen.item()))
    gen_optimizer.step()


def plot_points(data_type, generator, n_samples=1000):
    noise = torch.rand(n_samples, NOISE_DIM)
    gen_out = generator(noise)
    gen_x, gen_y = gen_out.detach().numpy()[:, 0], gen_out.detach().numpy()[:, 1]
    if data_type == 'par':
        real_y = map(lambda x: x ** 2, gen_x)
    elif data_type == 'line':
        real_y = gen_x
    else:
        print("non")
    plt.figure()
    plt.plot(gen_x, gen_y, '.', label='generated')
    # plt.plot(gen_x, list(real_y),  '.',  label='real')
    plt.title('Generated set')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    generator = Generator(noise_dim=NOISE_DIM)
    gen_optimizer = optim.Adam(generator.parameters(), lr=GEN_LR)

    discriminator = Discriminator()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=DIS_LR)

    loss = nn.BCELoss()
    for i in range(0, 10000):
        train(100, DATA, generator, discriminator, gen_optimizer, dis_optimizer, loss)

    plot_points(DATA, generator)
