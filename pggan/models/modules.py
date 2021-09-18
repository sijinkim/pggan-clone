import torch.nn as nn


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

    def forward(self, x):
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

    def forward(self, x):
        return x
