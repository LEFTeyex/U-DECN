import copy
from typing import Tuple

import torch
import torch.nn as nn
from mmdet.models.dense_heads import DETRHead
from mmdet.models.layers.transformer.utils import MLP
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, InstanceList
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor


@MODELS.register_module()
class DECOHead(DETRHead):
    def __init__(self,
                 *args,
                 num_reg_fcs: int = 3,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 as_two_stage: bool = False,
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
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
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

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def predict(self,
                hidden_states: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions
