import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from generator import Generator
from generate_data import get_data
from discriminator import Discriminator

GEN_LR = 0.01
DIS_LR = 0.01


def train_discriminator(example, discriminator, loss_fun, target):
    out = discriminator(example)
    loss = loss_fun(out, target)
    loss.backward()


def train(n_samles, data_type, generator, discriminator, gen_optimizer, dis_optimizer, loss_fun):
    noise = torch.rand(n_samles, 1)
    gen_out = generator(noise)
    real_ex = torch.tensor(get_data(n_points=n_samles, data_type=data_type), dtype=torch.float)

    dis_gen_out = gen_out.clone().detach()

    dis_gen = discriminator(dis_gen_out)
    dis_real = discriminator(real_ex)

    loss_dis_real = loss_fun(dis_real, torch.ones(n_samles, 1))
    loss_dis_real.backward()

    loss_dis_gen = loss_fun(dis_gen, torch.zeros(n_samles, 1))
    loss_dis_gen.backward(retain_graph=True)

    dis_optimizer.step()

    loss_gen = loss_fun(dis_gen, torch.ones(n_samles, 1))
    loss_gen.backward()
    print(loss_gen.item())
    gen_optimizer.step()


if __name__ == '__main__':
    generator = Generator()
    gen_optimizer = optim.Adam(generator.parameters(), lr=GEN_LR)

    discriminator = Discriminator()
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=DIS_LR)

    loss = nn.BCELoss()
    for i in range(0,10000):
        train(100, 'par', generator, discriminator, gen_optimizer, dis_optimizer, loss)
