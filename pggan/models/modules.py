import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self
    ) -> None:
        super(Generator, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(512, 512, 4, 1, 3),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU()
        )  # (512, 4, 4)

        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # (512, 8, 8)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (512, 16, 16)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (512, 32, 32)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (512, 64, 64)
            nn.Conv2d(512, 256, 3, 1, 1),  # (256, 64, 64)
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (256, 128, 128)
            nn.Conv2d(256, 128, 3, 1, 1),  # (128, 128, 128)
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (128, 256, 256)
            nn.Conv2d(128, 64, 3, 1, 1),  # (64, 256, 256)
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (64, 512, 512)
            nn.Conv2d(64, 32, 3, 1, 1),  # (32, 512, 512)
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.tail = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # (32, 1024, 1024)
            nn.Conv2d(32, 16, 3, 1, 1),  # (16, 1024, 1024)
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 1, 1, 0)  # (3, 1024, 1024) ToRGB layer
        )

    def forward(
        self,
        x
    ) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

    def forward(self, x):
        return x
