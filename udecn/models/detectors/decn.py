from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import OptConfigType
from torch import Tensor

from .deco import DECO
from ..layers import CdnConvQueryGenerator, DECNDecoder, DECOEncoder


@MODELS.register_module()
class DECN(DECO):
    """DEtection Convnet with deNoising training."""

    def __init__(self, *args,
                 dn_cfg: OptConfigType = None,
                 with_dn_training: bool = True,
                 with_u_color_query: bool = True,
                 mlvl_query_selection: bool = False,
                 **kwargs) -> None:
        self.with_dn_training = with_dn_training
        self.mlvl_query_selection = mlvl_query_selection
        # with underwater color query
        self.with_u_color_query = with_u_color_query
        super().__init__(*args, **kwargs)
        assert self.as_two_stage, 'as_two_stage must be True for DECN'
        assert self.with_box_refine, 'with_box_refine must be True for DECN'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_conv_query_generator = CdnConvQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        self.encoder = DECOEncoder(**self.encoder)
        self.decoder = DECNDecoder(**self.decoder)
        if self.hybrid is not None:
            self.hybrid = MODELS.build(self.hybrid)

        self.embed_dims = self.encoder.embed_dims[-1]
        self.encoder_proj = nn.Conv2d(
            self.feat_dims, self.encoder.embed_dims[0], 1, 1)

        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        if self.with_u_color_query:
            self.u_color_query_proj = nn.Linear(3, self.embed_dims)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)

    @staticmethod
    def _get_underwater_color(batch_inputs: Tensor,
                              batch_data_samples: SampleList) -> None:
        for input_img, sample in zip(batch_inputs, batch_data_samples):
            img_h, img_w = sample.img_shape
            color = torch.mean(
                input_img[:, :img_h, :img_w], dim=(-2, -1))  # shape (3,)
            sample.set_data({'color': color})

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        if self.with_u_color_query:
            self._get_underwater_color(batch_inputs, batch_data_samples)
        return super().loss(batch_inputs, batch_data_samples)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        if self.with_u_color_query:
            self._get_underwater_color(batch_inputs, batch_data_samples)
        return super().predict(batch_inputs, batch_data_samples, rescale)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        if self.with_u_color_query:
            self._get_underwater_color(batch_inputs, batch_data_samples)
        return super()._forward(batch_inputs, batch_data_samples)

    def pre_transformer(
            self,
            mlvl_feats: List[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[dict, dict]:
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))

        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask) in enumerate(zip(mlvl_feats, mlvl_masks)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(
                batch_size, len(mlvl_feats), 2)

        encoder_inputs_dict = dict(
            feat=mlvl_feats,
            feat_mask=mlvl_masks,
            feat_mask_flatten=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mlvl_masks,
            memory_mask_flatten=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_hybrid_encoder(self, feat: List[Tensor]) -> List[Tensor]:
        feat = list(feat)
        enc_feat = feat[-1]
        enc_feat = self.encoder_proj(enc_feat)
        feat[-1] = self.encoder(feat=enc_feat)
        memories = feat
        if self.hybrid is not None:
            memories = self.hybrid(memories)
        return memories

    def forward_encoder(self, feat: List[Tensor], feat_mask: List[Tensor],
                        feat_mask_flatten: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> dict:
        memories = self.forward_hybrid_encoder(feat)
        encoder_outputs_dict = dict(
            memories=memories,
            memory_mask=feat_mask,
            memory_mask_flatten=feat_mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_outputs_dict

    def pre_decoder(
            self,
            memories: List[Tensor],
            memory_mask: Tensor,
            memory_mask_flatten: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[dict, dict]:
        if not self.mlvl_query_selection:
            memories = memories[-1:]
            memory_mask = memory_mask[-1]
            spatial_shapes = spatial_shapes[-1:]
            valid_ratios = valid_ratios[:, -1:]
            if memory_mask_flatten is not None:
                memory_mask_flatten = memory_mask_flatten[:, level_start_index[-1]:]

        memory_flatten = []
        for memory in memories:
            bs, c, _, _ = memory.shape
            assert c == self.embed_dims
            memory = memory.flatten(2).permute(0, 2, 1)
            memory_flatten.append(memory)
        memory_flatten = torch.cat(memory_flatten, 1)

        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        # maybe the start lvl affect the base size of output_proposals
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory_flatten, memory_mask_flatten, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = enc_outputs_coord_unact + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight
        # (num_queries, embed_dims) -> (num_queries, bs, embed_dims)
        query = query.unsqueeze(1).repeat(1, bs, 1)
        # (num_queries, bs, embed_dims) -> (bs, embed_dims, query_h, query_w)
        query = query.permute(1, 2, 0).view(
            bs, self.embed_dims, self.query_h, self.query_w)

        if self.with_u_color_query:
            query_device = query.device
            batch_colors = []
            for sample in batch_data_samples:
                batch_colors.append(sample.color)
            batch_colors = torch.stack(batch_colors, dim=0)
            batch_colors = batch_colors.to(query_device)

            # (bs, 3) -> (bs, embed_dims)
            batch_colors = self.u_color_query_proj(batch_colors)
            # (bs, embed_dims) -> (bs, embed_dims, 1, 1)
            batch_colors = batch_colors[:, :, None, None]
            # (bs, embed_dims, 1, 1) -> (bs, embed_dims, query_h, query_w)
            batch_colors = batch_colors.repeat(1, 1, self.query_h, self.query_w)

        # append denoise query
        if self.training and self.with_dn_training:
            dn_label_query, dn_bbox_query, dn_meta = \
                self.dn_conv_query_generator(batch_data_samples)
            # reshape for dn_label_query
            # (bs, num_matching_queries, embed_dims) ->
            # (bs, embed_dims, query_h, query_w)
            dn_label_query = dn_label_query.permute(0, 2, 1).view(
                bs, self.embed_dims, self.query_h, self.query_w)
            query = [dn_label_query, query]
            reference_points = [dn_bbox_query, topk_coords_unact]
        else:
            reference_points = topk_coords_unact
            dn_meta = None
        reference_points = ([rp.sigmoid() for rp in reference_points]
                            if isinstance(reference_points, list)
                            else reference_points.sigmoid())

        if self.with_u_color_query:
            if isinstance(query, list):
                query = [q + batch_colors for q in query]
            else:
                query = query + batch_colors

        # take the last level as the input of decoder
        memories = memories[-1]
        if self.mlvl_query_selection:
            memory_mask = memory_mask[-1]
            spatial_shapes = spatial_shapes[-1:]
            valid_ratios = valid_ratios[:, -1:]
            if memory_mask_flatten is not None:
                memory_mask_flatten = memory_mask_flatten[:, level_start_index[-1]:]

        decoder_inputs_dict = dict(
            query=query,
            reference_points=reference_points,
            memory=memories,
            memory_mask=memory_mask,
            memory_mask_flatten=memory_mask_flatten,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Union[Tensor, List[Tensor]],
                        memory: Tensor,
                        memory_mask: Tensor,
                        memory_mask_flatten: Tensor,
                        reference_points: Union[Tensor, List[Tensor]],
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None) -> dict:
        inter_states, references = self.decoder(
            tgt=query,
            reference_points=reference_points,
            memory=memory,
            memory_mask=memory_mask,
            memory_mask_flatten=memory_mask_flatten,
            dn_mask=dn_mask,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches)

        # NOTE: This is to make sure label_embeding can be involved to
        # produce loss even if there is no denoising query (no ground truth
        # target in this GPU), otherwise, this will raise runtime error in
        # distributed training.
        inter_states[0] += \
            self.dn_conv_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
