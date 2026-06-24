from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


BlockFactory = type[BasicBlock] | type[Bottleneck]


class ResNet(nn.Module):
    def __init__(
        self,
        block: BlockFactory,
        layers: list[int],
        num_classes: int = 10,
        cifar_stem: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = 16 if cifar_stem else 64

        if cifar_stem:
            self.stem = nn.Sequential(
                conv3x3(3, 16),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )
            channels = [16, 32, 64]
            strides = [1, 2, 2]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            channels = [64, 128, 256, 512]
            strides = [1, 2, 2, 2]

        self.layers = nn.Sequential(
            *[
                self._make_layer(block, channel, count, stride)
                for channel, count, stride in zip(channels, layers, strides)
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1] * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: BlockFactory,
        channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        out_channels = channels * block.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def cifar_resnet(depth: int, num_classes: int = 10) -> ResNet:
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR ResNet depth must be 6n + 2, such as 20, 32, 44, 56, or 110.")
    blocks_per_stage = (depth - 2) // 6
    return ResNet(BasicBlock, [blocks_per_stage] * 3, num_classes=num_classes, cifar_stem=True)


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar_stem=False)


def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cifar_stem=False)


def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, cifar_stem=False)


def resnet101(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, cifar_stem=False)


def resnet152(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, cifar_stem=False)


MODEL_BUILDERS: dict[str, Callable[[int], ResNet]] = {
    "resnet20": lambda num_classes: cifar_resnet(20, num_classes),
    "resnet32": lambda num_classes: cifar_resnet(32, num_classes),
    "resnet44": lambda num_classes: cifar_resnet(44, num_classes),
    "resnet56": lambda num_classes: cifar_resnet(56, num_classes),
    "resnet110": lambda num_classes: cifar_resnet(110, num_classes),
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


def build_model(name: str, num_classes: int = 10) -> ResNet:
    try:
        return MODEL_BUILDERS[name.lower()](num_classes)
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"Unknown model '{name}'. Choices: {choices}") from exc
