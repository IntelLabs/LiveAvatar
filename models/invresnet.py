import math
from torch import nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def deconv4x4(in_planes, out_planes, stride=1, padding=1):
    """4x4 up-convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=padding, bias=False)


def norm2d(type):
    if type == 'batch':
        return nn.BatchNorm2d
    elif type == 'instance':
        return nn.InstanceNorm2d
    elif type == 'none':
        return nn.Identity
    else:
        raise ValueError("Invalid normalization type: ", type)


def upsampling_layer(inplanes, outplanes, stride, mode='deconv'):
    if mode == 'deconv':
        return  deconv4x4(inplanes, outplanes, stride)
    elif mode == 'bilinear':
        return torch.nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv1x1(inplanes, outplanes)
        )
    elif mode == 'pixel_shuffle':
        return torch.nn.Sequential(
            # conv3x3(inplanes, outplanes * 4),
            conv1x1(inplanes, outplanes*4),
            torch.nn.PixelShuffle(2),
            # conv3x3(inplanes // 4, outplanes),
            # conv1x1(inplanes // 4, outplanes)
        )
    else:
        raise ValueError


class InvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, layer_normalization='batch', upsampling_mode='deconv'):
        super(InvBasicBlock, self).__init__()
        self.layer_normalization = layer_normalization
        if upsample is not None:
            self.conv1 = upsampling_layer(inplanes, planes, stride, mode=upsampling_mode)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(layer_normalization)(planes)
        self.bn2 = norm2d(layer_normalization)(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.tanh = nn.Tanh()
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        # out = self.relu(out)
        return out
        # return self.tanh(out)


class InvResNet(nn.Module):

    def __init__(
            self,
            input_dims,
            num_blocks,
            planes=None,
            block=InvBasicBlock,
            hidden_dim=512,
            output_size=256,
            output_channels=3,
            layer_normalization='none',
            upsampling_mode='deconv',
    ):
        super(InvResNet, self).__init__()
        self.output_size = output_size
        self.output_channels = output_channels
        self.upsampling_mode = upsampling_mode

        num_layers = len(num_blocks)

        if planes is None:
            planes = [256, 128, 64, 32, 32, 32][:num_layers]

        self.layer_normalization = layer_normalization
        self.norm = norm2d(layer_normalization)

        if hidden_dim is not None:
            self.fc = nn.Linear(input_dims, hidden_dim)
            self.conv1 = deconv4x4(hidden_dim, hidden_dim, padding=0)
            self.bn1 = self.norm(hidden_dim)
            self.relu = nn.ReLU(inplace=True)
            self.inplanes = hidden_dim
        else:
            self.inplanes = input_dims

        self.layers = nn.Sequential(*[
            self._make_layer(block, planes[i], num_blocks[i], stride=2, upsampling_mode=upsampling_mode)
            for i in range(num_layers)
        ])

        # self.out = nn.Conv2d(planes[-1], output_channels, kernel_size=3, padding=1)
        self.out = nn.Conv2d(planes[-1], output_channels, kernel_size=1, padding=0)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n) * 0.1)
                # m.weight.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_down(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1, upsampling_mode='deconv'):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = upsampling_layer(self.inplanes, planes * block.expansion, stride, mode=upsampling_mode)

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, upsample, layer_normalization=self.layer_normalization, upsampling_mode=upsampling_mode
        ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, layer_normalization=self.layer_normalization
            ))

        return nn.Sequential(*layers)

    def forward(self, x, _=None):

        if len(x.shape) == 2:
            x = self.fc(x)
            x = x.view(x.size(0), -1, 1,1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.layers(x)

        scale_factor = dict(
            bilinear=0.01,
            pixel_shuffle=0.1,
            deconv=1.0,
        )[self.upsampling_mode]

        x = self.out(x) * scale_factor
        x = self.tanh(x)
        return x
