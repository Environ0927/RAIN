"""Small-image ResNet-34/50 for CIFAR-100 and Tiny-ImageNet.

Unlike ImageNet ResNet, the stem is a 3x3 stride-one convolution and has no
max-pool. Tiny-ImageNet downsampling starts in stage two, preserving useful
spatial resolution for 64x64 inputs.
"""
from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(value)
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        width = channels
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        output_channels = channels * self.expansion
        self.shortcut = nn.Identity() if stride == 1 and in_channels == output_channels else nn.Sequential(
            nn.Conv2d(in_channels, output_channels, 1, stride, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(value)
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        return self.relu(value + residual)


class SmallImageResNet(nn.Module):
    def __init__(self, block, layers: tuple[int, int, int, int], num_classes: int) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize()

    def _make_layer(self, block, channels: int, count: int, stride: int) -> nn.Sequential:
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels * block.expansion
        layers.extend(block(self.in_channels, channels) for _ in range(1, count))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01); nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(value)))))
        return self.fc(torch.flatten(self.pool(value), 1))


def resnet34_small(num_classes: int = 100) -> SmallImageResNet:
    return SmallImageResNet(BasicBlock, (3, 4, 6, 3), num_classes)


def resnet50_small(num_classes: int = 200) -> SmallImageResNet:
    return SmallImageResNet(Bottleneck, (3, 4, 6, 3), num_classes)
