import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor


class RepVggBlock(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = None
        self.big_conv = ConvModule(
            in_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.small_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.act = (nn.Identity() if act_cfg is None
                    else build_activation_layer(act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        if self.conv is not None:
            y = self.conv(x)
        else:
            y = self.big_conv(x) + self.small_conv(x)
        return self.act(y)

    def convert_to_deploy(self) -> None:
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.in_channels,
                                  self.out_channels, 3, 1, padding=1)
        kernel, bias = self._get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

        delattr(self, 'big_conv')
        delattr(self, 'small_conv')

    def _get_equivalent_kernel_bias(self) -> tuple:
        kernel_big, bias_big = self._fuse_bn_tensor(self.big_conv)
        kernel_small, bias_small = self._fuse_bn_tensor(self.small_conv)

        # pad kernel_small
        kernel_small = F.pad(kernel_small, [1, 1, 1, 1])

        equivalent_kernel = kernel_big + kernel_small
        equivalent_bias = bias_big + bias_small
        return equivalent_kernel, equivalent_bias

    @staticmethod
    def _fuse_bn_tensor(branch: ConvModule) -> tuple:
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 expand_ratio: float = 1.0,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if mid_channels != out_channels:
            self.final_conv = ConvModule(
                mid_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.final_conv = nn.Identity()

        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                mid_channels,
                mid_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.bottlenecks(x_main)
        return self.final_conv(x_main + x_short)
