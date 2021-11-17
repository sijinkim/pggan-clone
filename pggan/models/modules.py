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

        self.conv_blocks = torch.nn.ModuleList([
            DiscriminatorConvBlock(
                in_channel=16,
                out_channel=32,
                leakiness=0.2,
            ),
            DiscriminatorConvBlock(
                in_channel=32,
                out_channel=64,
                leakiness=0.2,
            ),
            DiscriminatorConvBlock(
                in_channel=64,
                out_channel=128,
                leakiness=0.2,
            ),
            DiscriminatorConvBlock(
                in_channel=128,
                out_channel=256,
                leakiness=0.2,
            ),
            DiscriminatorConvBlock(
                in_channel=256,
                out_channel=512,
                leakiness=0.2,
            ),
            DiscriminatorConvBlock(
                in_channel=512,
                out_channel=512,
                leakiness=0.2,
            ),
            DiscriminatorConvBlock(
                in_channel=512,
                out_channel=512,
                leakiness=0.2,
            ),
            DiscriminatorConvBlock(
                in_channel=512,
                out_channel=512,
                leakiness=0.2,
            ),
        ])
        self.minibatch_stddev = MinibatchStdDev()

    def forward(self, x) -> torch.Tensor:
        return x


class DiscriminatorConvBlock(nn.Module):
    def __init__(
            self,
            in_channel:     int,
            out_channel:    int,
            leakiness:      float,
    ) -> None:
        super(DiscriminatorConvBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.lrelu = torch.nn.LeakyReLU(leakiness)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
        )

    def forward(
            self,
            x:      torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.lrelu(x)
        x = self.conv_2(x)
        x = self.lrelu(x)
        x = self.avg_pool(x)
        return x


class MinibatchStdDev(nn.Module):
    def __init__(self) -> None:
        super(MinibatchStdDev, self).__init__()

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        '''
            input:
                x: torch.Tensor  [B x C x H x W]

            output:
                torch.Tensor [B x C+1 x H x W]
        '''
        B, C, H, W = x.size()
        # We first compute the standard deviation for each feature
        # in each spatial location over the minibatch.
        additional_feature = x.std(dim=0)  # dim 0 stands for minibatch.

        # We then average these estimates over all features and spatial locations
        # to arrive at a single value.
        additional_feature = additional_feature.mean()
        assert additional_feature.dim() == 0

        # We replicate the value and concatenate it to all spatial locations and over the minibatch,
        # yielding one additional (constant) feature map.
        additional_feature = additional_feature.new_full(
            size=(B, 1, H, W),
            fill_value=additional_feature.data,
        )

        # The concatenated tensor should have a shape like [B, C+1, H, W]
        x = torch.cat(
            tensors=(x, additional_feature),
            dim=1,
        )
        assert x.size() == (B, C+1, H, W)

        return x
