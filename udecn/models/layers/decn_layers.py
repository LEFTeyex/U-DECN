from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.layers.transformer import CdnQueryGenerator
from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding, inverse_sigmoid
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType
from torch import Tensor

from .deco_layers import DECODecoderLayer
from ...dcn_ops import DCNv3 as _DCNv3


class DECNDecoder(nn.Module):
    def __init__(self,
                 embed_dims: int,
                 query_h: int = 10,
                 query_w: int = 10,
                 num_layers: int = 6,
                 sim_dcn: dict = None,
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
            DECNDecoderLayer(embed_dims, query_h, query_w,
                             drop_path, sim_dcn,
                             layer_scale_init_value)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims)

        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)

    def forward(self,
                tgt: Union[Tensor, List[Tensor]],
                reference_points: Union[Tensor, List[Tensor]],
                memory: Tensor,
                memory_mask: Tensor,
                memory_mask_flatten: Tensor,
                dn_mask: Tensor,
                spatial_shapes: Tensor,
                valid_ratios: Tensor,
                reg_branches: nn.Module) -> Tuple[Tensor, Tensor]:
        if isinstance(tgt, list):  # has denoise query
            assert len(tgt) == 2
            intermediates = []
            intermediates_reference_points = []
            for tt, rps in zip(tgt, reference_points):
                out, out_rps = self.inner_forward(tt, rps,
                                                  memory, memory_mask,
                                                  memory_mask_flatten,
                                                  dn_mask, spatial_shapes,
                                                  valid_ratios, reg_branches)
                intermediates.append(out)
                intermediates_reference_points.append(out_rps)
            # the first is dn query and the second is learnable query
            return torch.concat(intermediates, 2), torch.concat(
                intermediates_reference_points, 2)
        else:
            return self.inner_forward(tgt, reference_points,
                                      memory, memory_mask,
                                      memory_mask_flatten,
                                      dn_mask, spatial_shapes,
                                      valid_ratios, reg_branches)

    def inner_forward(self,
                      tgt: Tensor,
                      reference_points: Tensor,
                      memory: Tensor,
                      memory_mask: Tensor,
                      memory_mask_flatten: Tensor,
                      dn_mask: Tensor,
                      spatial_shapes: Tensor,
                      valid_ratios: Tensor,
                      reg_branches: nn.Module) -> Tuple[Tensor, Tensor]:
        intermediate = []
        intermediate_reference_points = [reference_points]
        out_tgt = None
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :], self.embed_dims // 2)

            query_pos = self.ref_point_head(query_sine_embed)
            # reshape for reference_points query_pos
            # (bs, num_queries, embed_dims) ->
            # (bs, embed_dims, query_h, query_w)
            bs = query_pos.size(0)
            query_pos = query_pos.permute(0, 2, 1).view(
                bs, self.embed_dims, self.query_h, self.query_w)

            tgt = layer(tgt, query_pos, memory)
            # (bs, embed_dims, h, w) -> (bs, num_queries(h*w), embed_dims)
            out_tgt = tgt.flatten(2).transpose(1, 2)

            tmp = reg_branches[lid](out_tgt)
            assert reference_points.shape[-1] == 4
            new_reference_points = tmp + inverse_sigmoid(
                reference_points, eps=1e-3)
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(out_tgt))
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        # DINO without norm when return_intermediate is False
        return self.norm(out_tgt).unsqueeze(0), reference_points.unsqueeze(0)


class DECNDecoderLayer(DECODecoderLayer):
    def __init__(self,
                 embed_dims: int,
                 query_h: int = 10,
                 query_w: int = 10,
                 drop_path: float = 0.,
                 sim_dcn: dict = None,
                 layer_scale_init_value: float = 1e-6):
        super().__init__(embed_dims, query_h, query_w,
                         drop_path, layer_scale_init_value)
        if sim_dcn is not None:
            dcn = dict(channels=self.embed_dims, kernel_size=9,
                       padding=4, groups=self.embed_dims)
            dcn.update(sim_dcn)
            self.dwconv1 = DCNv3(**dcn)


class CdnConvQueryGenerator(CdnQueryGenerator):
    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super(CdnQueryGenerator, self).__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        # prepare grouping strategy
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        if self.dynamic_dn_groups:
            self.num_dn_queries = group_cfg.get('num_dn_queries', num_matching_queries // 2)
            assert isinstance(self.num_dn_queries, int), \
                f'Expected the num_dn_queries to have type int, but got ' \
                f'{self.num_dn_queries}({type(self.num_dn_queries)}). '
            assert self.num_dn_queries == num_matching_queries // 2, (
                'make sure num_dn_queries is equal to half num_queries')
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using static dn groups'
            self.num_groups = group_cfg['num_groups']
            assert isinstance(self.num_groups, int), \
                f'Expected the num_groups to have type int, but got ' \
                f'{self.num_groups}({type(self.num_groups)}). '

        self.label_embedding = nn.Embedding(self.num_classes, self.embed_dims)

    def __call__(self, batch_data_samples: SampleList) -> tuple:
        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            gt_labels_list.append(sample.gt_instances.labels)
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups,
            num_matching_queries=self.num_matching_queries)

        return dn_label_query, dn_bbox_query, dn_meta

    def collate_dn_queries(self, input_label_query: Tensor,
                           input_bbox_query: Tensor, batch_idx: Tensor,
                           batch_size: int, num_groups: int) -> Tuple[Tensor, Tensor]:
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        if num_denoising_queries < self.num_matching_queries:
            num_denoising_queries = self.num_matching_queries

        map_query_index = torch.cat([
            torch.arange(num_target, device=device)
            for num_target in num_target_list
        ])
        map_query_index = torch.cat([
            map_query_index + max_num_target * i for i in range(2 * num_groups)
        ]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device)
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device)

        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query

        # drop the num_drop denoising queries which exceed the num_matching_queries (num_queries of DECN)
        # num_drop = num_denoising_queries - self.num_matching_queries
        if num_denoising_queries > self.num_matching_queries:
            # drop the last
            batched_label_query = batched_label_query[:, :self.num_matching_queries]
            batched_bbox_query = batched_bbox_query[:, :self.num_matching_queries]
        return batched_label_query, batched_bbox_query


class DCNv3(_DCNv3):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            offset_scale: float = 1.0,
            act_layer: str = 'GELU',
            norm_layer: str = 'LN',
            center_feature_scale: bool = False):
        super().__init__(channels, kernel_size, kernel_size,
                         stride, padding, dilation, groups, offset_scale,
                         act_layer, norm_layer, center_feature_scale)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x
