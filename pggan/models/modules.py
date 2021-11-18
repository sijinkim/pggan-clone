from collections import OrderedDict
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
    def __init__(
        self,
        leakiness: float = 0.2,
        maximum_channel: int = 512,
    ) -> None:
        super(Discriminator, self).__init__()

        self.leakiness = leakiness
        self.maximum_channel = maximum_channel
        self.current_input_image_size = 4

        self.fromRGB = self._get_fromRGB(out_channels=512)

        self.conv_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv_1', torch.nn.Conv2d(513, 512, 3, 1, 1)),
                ('lrelu_1', torch.nn.LeakyReLU(self.leakiness)),
                ('conv_2', torch.nn.Conv2d(512, 512, 4, 1, 0)),
                ('lrelu_2', torch.nn.LeakyReLU(self.leakiness)),
            ])),
        ])

        self.minibatch_stddev = MinibatchStdDev()

        self.fc = nn.Linear(
            in_features=512,
            out_features=1,
        )

    def forward(self, x) -> torch.Tensor:
        x = self.fromRGB(x)
        for block_idx, module in enumerate(self.conv_blocks):
            if block_idx == len(self.conv_blocks)-1:
                x = self.minibatch_stddev(x)
            x = module(x)
        x = self.fc(x.squeeze())

        return x

    def _get_fromRGB(self, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def set_input_image_size(self, image_size):
        if image_size > self.current_input_image_size:
            # compute in_channels and out_channels
            in_channels = min(self.maximum_channel, 2**(15-(len(bin(image_size))-2)))
            out_channels = min(self.maximum_channel, in_channels << 1)

            new_fromRGB = self._get_fromRGB(out_channels=in_channels)
            new_conv_block = torch.nn.Sequential(OrderedDict([
                ('conv_1', torch.nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
                ('lrelu_1', torch.nn.LeakyReLU(self.leakiness)),
                ('conv_2', torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
                ('lrelu_2', torch.nn.LeakyReLU(self.leakiness)),
                ('avg_pooling', torch.nn.AvgPool2d(2)),
            ]))

            self.fromRGB = new_fromRGB
            self.conv_blocks.insert(0, new_conv_block)
        else:
            pass


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
        additional_feature = additional_feature.expand(
            size=(B, 1, H, W),
        )

        # The concatenated tensor should have a shape like [B, C+1, H, W]
        x = torch.cat(
            tensors=(x, additional_feature),
            dim=1,
        )
        assert x.size() == (B, C+1, H, W)

        return x
