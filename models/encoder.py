import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Any, Callable, List, Optional, Type, Union
from utils.nn import conv1x1, conv3x3
from torchvision.models import resnet


class ResNetEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_blocks,
            strides,
            hidden_dim = 256,
            flatten=True,
            groups: int = 1,
            width_per_group: int = 64,
            planes=None,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            zero_init_residual=True
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        num_layers = len(num_blocks)

        if planes is None:
            planes = [64, 128, 256, 512][:num_layers]

        self.flatten = flatten

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.hidden_dim = hidden_dim

        if hidden_dim is not None:
            self.inplanes = input_dim
            self.stem = nn.Identity()
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # nn.Conv2d(64, 128, 3, stride=1, padding=1)
            )
            self.inplanes = 64
            # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            # self.bn1 = norm_layer(self.inplanes)
            # self.relu = nn.ReLU(inplace=True)
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BasicBlock
        layers = [
            self._make_layer(block, planes=planes[i], blocks=num_blocks[i], stride=strides[i])
            for i in range(len(num_blocks))
        ]
        self.res_blocks = nn.Sequential(*layers)

        if flatten:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(planes[-1], output_dim)
        else:
            # self.out = conv3x3(input_dim, output_dim)
            self.out = nn.Sequential(
                conv1x1(planes[-1], output_dim),
                # conv3x3(planes[-1], output_dim),
                # nn.AdaptiveAvgPool2d((16, 16)),
            )
            # self.out_bn = self._norm_layer(output_dim)
            # torch.nn.init.xavier_uniform(self.out.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.stem(x)

        x = self.res_blocks(x)

        if self.flatten:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else:
            x = self.out(x)
            # x = self.out_bn(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

