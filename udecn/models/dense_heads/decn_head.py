import copy
from typing import Dict

import torch
import torch.nn as nn
from mmdet.models.dense_heads import DINOHead
from mmdet.models.layers.transformer.utils import MLP
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType
from mmengine.model import bias_init_with_prob, constant_init
from mmengine.structures import InstanceData
from torch import Tensor


@MODELS.register_module()
class DECNHead(DINOHead):
    def __init__(self,
                 *args,
                 num_reg_fcs: int = 3,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 train_cfg: ConfigType = dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         match_costs=[
                             dict(type='FocalLossCost', weight=1.0),
                             dict(type='BBoxL1Cost', box_format='xywh', weight=5.0),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ])),
                 **kwargs):
        super().__init__(*args,
                         num_reg_fcs=num_reg_fcs,
                         loss_cls=loss_cls,
                         train_cfg=train_cfg,
                         **kwargs)

    def _init_layers(self) -> None:
        fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        fc_reg = MLP(self.embed_dims, self.embed_dims, 4,
                     self.num_reg_fcs)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [fc_reg for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList([
                copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(fc_reg) for _ in range(self.num_pred_layer)])

    def init_weights(self) -> None:
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m.layers[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0].layers[-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m.layers[-1].bias.data[2:], 0.0)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str, int]) -> tuple:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_queries = dn_meta['num_matching_queries']
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        if num_denoising_queries < num_queries:
            num_denoising_queries = num_queries
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        if num_denoising_queries > num_queries:
            labels = labels[:num_queries]
            label_weights = label_weights[:num_queries]
            bbox_targets = bbox_targets[:num_queries]
            bbox_weights = bbox_weights[:num_queries]
            pos_inds = pos_inds[pos_inds < num_queries]
            neg_inds = neg_inds[neg_inds < num_queries]
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> tuple:
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_matching_queries']
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)
