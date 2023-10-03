# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, style="ImageNet"):
        super(ResNet, self).__init__()
        self.style = style
        self.in_planes = 64 if style == "ImageNet" else 16

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        if style == "ImageNet":
            self.layer1 = self.__make__layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self.__make__layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self.__make__layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self.__make__layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        elif style == "Cifar":
            self.layer1 = self.__make__layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self.__make__layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self.__make__layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64, num_classes)

    def __make__layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.style == "ImageNet":
            out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=(out.shape[2], out.shape[3]))
        self.penult = out.view(out.size(0), -1)
        out = self.linear(self.penult)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def CifarResnet20(num_classes):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, style="Cifar")


def CifarResnet32(num_classes):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, style="Cifar")


def CifarResnet56(num_classes):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, style="Cifar")


def test():
    net = ResNet18(10)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
