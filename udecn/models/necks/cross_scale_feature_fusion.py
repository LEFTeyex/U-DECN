from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule

from ..layers import CSPRepLayer


@MODELS.register_module()
class CrossScaleFeatureFusion(BaseModule):
    def __init__(
            self,
            in_channels: Sequence[int],
            mid_channels: int,
            out_channels: Optional[int] = None,
            out_indices: Sequence[int] = (-1,),
            num_csp_blocks: int = 3,
            use_out_norm_act: bool = False,
            expand_ratio: float = 1.0,
            depth_ratio: float = 1.0,
            upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
            conv_cfg: ConfigType = None,
            norm_cfg: ConfigType = dict(type='BN'),
            act_cfg: ConfigType = dict(type='SiLU'),
            init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.out_indices = out_indices

        # input channel projections
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                ConvModule(
                    in_channel,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    mid_channels,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPRepLayer(
                    mid_channels * 2,
                    mid_channels,
                    num_blocks=round(num_csp_blocks * depth_ratio),
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPRepLayer(
                    mid_channels * 2,
                    mid_channels,
                    num_blocks=round(num_csp_blocks * depth_ratio),
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        if out_channels is not None:
            self.out_convs = nn.ModuleList()
            for i in range(len(out_indices)):
                self.out_convs.append(
                    ConvModule(
                        mid_channels,
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg if use_out_norm_act else None,
                        act_cfg=act_cfg if use_out_norm_act else None))

    def forward(self, inputs: tuple) -> tuple:
        assert len(inputs) == len(self.in_channels)
        # inputs project
        inputs = [self.input_proj[i](x) for i, x in enumerate(inputs)]

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_height = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_height = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_height)
            inner_outs[0] = feat_height

            upsample_feat = self.upsample(feat_height)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        new_outs = []
        for idx, i_out in enumerate(self.out_indices):
            new_out = outs[i_out]
            if self.out_channels is not None:
                conv = self.out_convs[idx]
                new_out = conv(new_out)

            new_outs.append(new_out)
        return tuple(new_outs)
