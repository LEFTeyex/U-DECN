import numbers
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from torch import Tensor


class DECOEncoder(nn.Module):
    """DECOEncoder which is an implement from ConvNeXt."""

    def __init__(self,
                 embed_dims: List[int] = [120, 240, 480],
                 depths: List[int] = [2, 6, 2],
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.embed_dims = embed_dims
        self.depths = depths

        self.downsample_layers = nn.ModuleList()
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(embed_dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[DECOEncoderLayer(embed_dims[i], dp_rates[cur + j], layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, feat: Tensor) -> Tensor:
        for i in range(len(self.stages) - 1):
            feat = self.stages[i](feat)
            feat = self.downsample_layers[i](feat)
        feat = self.stages[-1](feat)
        return feat


class DECODecoder(nn.Module):
    def __init__(self,
                 embed_dims: int,
                 query_h: int = 10,
                 query_w: int = 10,
                 num_layers: int = 6,
                 return_intermediate: bool = True,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.embed_dims = embed_dims
        self.query_h = query_h
        self.query_w = query_w
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = nn.ModuleList([
            DECODecoderLayer(embed_dims, query_h, query_w,
                             drop_path, layer_scale_init_value)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, tgt: Tensor, query_pos: Tensor,
                memory: Tensor, reg_branches: nn.Module = None) -> Tensor:
        intermediate = []
        out_tgt = None
        for layer in self.layers:
            tgt = layer(tgt, query_pos, memory)
            # (bs, embed_dims, h, w) -> (bs, num_queries(h*w), embed_dims)
            out_tgt = tgt.flatten(2).transpose(1, 2)

            if self.return_intermediate:
                intermediate.append(self.norm(out_tgt))

        if self.return_intermediate:
            # (lvl, bs, num_queries, embed_dims)
            return torch.stack(intermediate)

        # DINO without norm when return_intermediate is False
        return self.norm(out_tgt).unsqueeze(0)


class DECOEncoderLayer(nn.Module):
    """ConvNeXt Layer."""

    def __init__(self,
                 embed_dims: int,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(embed_dims, embed_dims,
                                kernel_size=7, padding=3, groups=embed_dims)
        self.norm = LayerNorm(embed_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(embed_dims, 4 * embed_dims)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * embed_dims, embed_dims)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(embed_dims),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class DECODecoderLayer(nn.Module):
    def __init__(self,
                 embed_dims: int,
                 query_h: int = 10,
                 query_w: int = 10,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.embed_dims = embed_dims
        self.query_h = query_h
        self.query_w = query_w

        # SIM
        self.dwconv1 = nn.Conv2d(embed_dims, embed_dims,
                                 kernel_size=9, padding=4, groups=embed_dims)
        self.norm1 = LayerNorm(embed_dims, eps=1e-6)
        self.pwconv1_1 = nn.Linear(embed_dims, 4 * embed_dims)
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Linear(4 * embed_dims, embed_dims)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(embed_dims),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # CIM
        self.dwconv2 = nn.Conv2d(embed_dims, embed_dims,
                                 kernel_size=9, padding=4, groups=embed_dims)
        self.norm2 = LayerNorm(embed_dims, eps=1e-6)
        self.pwconv2_1 = nn.Linear(embed_dims, 4 * embed_dims)
        self.act2 = nn.GELU()
        self.pwconv2_2 = nn.Linear(4 * embed_dims, embed_dims)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(embed_dims),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pooling = nn.AdaptiveMaxPool2d((self.query_h, self.query_w))

    def forward(self,
                tgt: Tensor,
                query_pos: Tensor,
                memory: Tensor) -> Tensor:
        b, d, h, w = memory.shape

        # SIM
        tgt2 = tgt + query_pos
        tgt2 = self.dwconv1(tgt2)
        tgt2 = tgt2.permute(0, 2, 3, 1)  # (b,d,qh,qw) -> (b,qh,qw,d)
        tgt2 = self.norm1(tgt2)

        tgt2 = self.pwconv1_1(tgt2)
        tgt2 = self.act1(tgt2)
        tgt2 = self.pwconv1_2(tgt2)
        if self.gamma1 is not None:
            tgt2 = self.gamma1 * tgt2
        tgt2 = tgt2.permute(0, 3, 1, 2)  # (b,qh,qw,d) -> (b,d,qh,qw)
        tgt = tgt + self.drop_path1(tgt2)

        # CIM
        tgt = F.interpolate(tgt, size=(h, w))
        tgt2 = tgt + memory
        tgt2 = self.dwconv2(tgt2)
        tgt2 = tgt2 + tgt
        tgt2 = tgt2.permute(0, 2, 3, 1)  # (b,d,h,w) -> (b,h,w,d)
        tgt2 = self.norm2(tgt2)

        # FFN
        tgt = tgt2
        tgt2 = self.pwconv2_1(tgt2)
        tgt2 = self.act2(tgt2)
        tgt2 = self.pwconv2_2(tgt2)
        if self.gamma2 is not None:
            tgt2 = self.gamma2 * tgt2
        tgt2 = tgt2.permute(0, 3, 1, 2)  # (b,h,w,d) -> (b,d,h,w)
        tgt = tgt.permute(0, 3, 1, 2)  # (b,h,w,d) -> (b,d,h,w)
        tgt = tgt + self.drop_path1(tgt2)

        # Pooling
        tgt = self.pooling(tgt)
        return tgt


class LayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 data_format: str = 'channels_last'):
        super().__init__()
        if data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError

        self.eps = eps
        self.data_format = data_format
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        else:  # self.data_format == 'channels_first'
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
