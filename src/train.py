import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from discriminator import Discriminator
from generate_data import get_data
from generator import Generator
from utils import *

NOISE_DIM = 2
GEN_LR = 0.001
DIS_LR = 0.002
DATA = 'line'


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
    dis_loss_real = loss_fun(dis_real, torch.ones(n_samples, dtype=torch.long))
    dis_loss_real.backward()

    # Train discriminator on fake
    noise = torch.rand(n_samples, NOISE_DIM)
    gen_out = generator(noise)

    dis_gen = discriminator(gen_out.detach())
    dis_loss_gen = loss_fun(dis_gen, torch.zeros(n_samples, dtype=torch.long))
    dis_loss_gen.backward()
    #print("Discriminator loss : {}".format(dis_loss_gen.item() + dis_loss_real.item()))
    dis_optimizer.step()

    # Train generator
    dis_gen = discriminator(gen_out)
    loss_gen = loss_fun(dis_gen, torch.ones(n_samples, dtype=torch.long))
    loss_gen.backward()
    #print("Generator loss : {}".format(loss_gen.item()))
    gen_optimizer.step()

    return dis_loss_gen.item() + dis_loss_real.item(), loss_gen.item()


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
    gen_optimizer = optim.RMSprop(generator.parameters(), lr=GEN_LR)

    discriminator = Discriminator()
    dis_optimizer = optim.RMSprop(discriminator.parameters(), lr=DIS_LR)

    loss = nn.CrossEntropyLoss()
    dis_loss_history = []
    gen_loss_history = []
    for i in range(0, 80000):
        dis_loss, gen_loss = train(100, DATA, generator, discriminator, gen_optimizer, dis_optimizer, loss)
        dis_loss_history.append(dis_loss)
        gen_loss_history.append(gen_loss)

    plot_points(DATA, generator)
    create_graph("Loss", array_datas=[dis_loss_history, gen_loss_history], array_legends=["Discriminator", "Generator"],
                 xlabel="Epoch", ylabel="Loss",
                 make_new=True)
