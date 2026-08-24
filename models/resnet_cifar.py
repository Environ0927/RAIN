"""CIFAR-style ResNet-18 (3x3 stem, stride one, no initial max-pool)."""

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        value = self.relu(self.bn1(self.conv1(inputs)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + residual)


class CifarResNet(nn.Module):
    def __init__(self, blocks=(2, 2, 2, 2), num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = self._layer(64, blocks[0], 1)
        self.layer2 = self._layer(128, blocks[1], 2)
        self.layer3 = self._layer(256, blocks[2], 2)
        self.layer4 = self._layer(512, blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _layer(self, channels: int, count: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, count))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs)))))
        return self.fc(torch.flatten(self.pool(value), 1))


def resnet18_cifar(num_classes: int = 10) -> CifarResNet:
    return CifarResNet((2, 2, 2, 2), num_classes)
