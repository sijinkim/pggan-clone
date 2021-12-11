from collections import OrderedDict
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        leakiness: float = 0.2,
        maximum_channel: int = 512,
    ) -> None:
        super(Generator, self).__init__()

        self.leakiness = leakiness
        self.maximum_channel = maximum_channel
        self.current_output_image_size = 4

        self.toRGBs = {
            'from_512': self._get_toRGB(in_channels=512),
        }
        self.fade_in_weight = 1

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv_1', nn.Conv2d(512, 512, 4, 1, 3)),  # (512, 4, 4)
                ('lrelu_1', nn.LeakyReLU(self.leakiness)),
                ('conv_2', nn.Conv2d(512, 512, 3, 1, 1)),  # (512, 4, 4)
                ('lrelu_2', nn.LeakyReLU(self.leakiness)),
            ])),
        ])  # default conv_blocks

    def forward(self, x) -> torch.Tensor:
        for m in self.conv_blocks:
            x = m(x)
        return x

    def _get_toRGB(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=3,
                kernel_size=1,
                padding=0,
            ),
        )

    def _get_out_channels(self, resolution):
        return min(self.maximum_channel, 2**(15-(len(bin(resolution))-2)))

    def set_output_image_size(self, image_size):
        if image_size > self.current_output_image_size:
            in_channels = min(self.maximum_channel,
                              self._get_out_channels(image_size) << 1)
            out_channels = self._get_out_channels(image_size)

            # 특정 중간 레이어에서 rgb 아웃풋낼 수 있도록 모든 out_channels 사이즈에 맞춰서 toRGB 레이어 생성
            if f'from_{out_channels}' not in self.toRGBs.keys():
                self.toRGBs[f'from_{out_channels}'] = self._get_toRGB(
                    in_channels=out_channels)

            new_conv_block = nn.Sequential(OrderedDict([
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
                ('conv_1', nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
                ('lrelu_1', nn.LeakyReLU(self.leakiness)),
                ('conv_2', nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
                ('lrelu_2', nn.LeakyReLU(self.leakiness))
            ]))

            self.conv_blocks.append(new_conv_block)
            self.current_output_image_size = image_size
        else:
            pass

    def PixelwizeNorm(self):
        pass


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

        self.fromRGBs = {
            'to_512': self._get_fromRGB(out_channels=512)
        }
        # 1 means that no lower resolution is considerd.
        self.fade_in_weight = 1

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv_1', nn.Conv2d(513, 512, 3, 1, 1)),
                ('lrelu_1', nn.LeakyReLU(self.leakiness)),
                ('conv_2', nn.Conv2d(512, 512, 4, 1, 0)),
                ('lrelu_2', nn.LeakyReLU(self.leakiness)),
            ])),
        ])

        self.minibatch_stddev = MinibatchStdDev()

        self.fc = nn.Linear(
            in_features=512,
            out_features=1,
        )

    def forward(self, x) -> torch.Tensor:

        if self.fade_in_weight < 1:
            in_channels_to_fade_in = self._get_in_channels(x.size(-1) >> 1)
            residual_x = nn.AvgPool2d(2)(x)
            residual_x = self.fromRGBs[f'to_{in_channels_to_fade_in}'](
                residual_x)
            residual_x = residual_x.mul(1-self.fade_in_weight)

        in_channels = self._get_in_channels(x.size(-1))
        x = self.fromRGBs[f'to_{in_channels}'](x)
        for block_idx, module in enumerate(self.conv_blocks):
            if block_idx == 1 and self.fade_in_weight < 1:
                x = x.mul(self.fade_in_weight).add(residual_x)

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

    def _get_in_channels(self, resolution):
        return min(self.maximum_channel, 2**(15-(len(bin(resolution))-2)))

    def set_input_image_size(self, image_size):
        if image_size > self.current_input_image_size:
            # compute in_channels and out_channels
            in_channels = self._get_in_channels(image_size)
            out_channels = min(self.maximum_channel, in_channels << 1)

            if f'to_{in_channels}' not in self.fromRGBs.keys():
                self.fromRGBs[f'to_{in_channels}'] = self._get_fromRGB(
                    out_channels=in_channels)

            new_conv_block = nn.Sequential(OrderedDict([
                ('conv_1', nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
                ('lrelu_1', nn.LeakyReLU(self.leakiness)),
                ('conv_2', nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
                ('lrelu_2', nn.LeakyReLU(self.leakiness)),
                ('avg_pooling', nn.AvgPool2d(2)),
            ]))

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
