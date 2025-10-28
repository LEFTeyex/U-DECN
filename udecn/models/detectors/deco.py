from typing import List, Tuple

import torch
import torch.nn as nn
from mmdet.models.detectors import (DINO, DeformableDETR, DetectionTransformer)
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from torch import Tensor

from ..layers import DECODecoder, DECOEncoder


@MODELS.register_module()
class DECO(DINO):
    """DEtection COnvnet."""

    def __init__(self,
                 *args,
                 hybrid: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 num_queries: int = 100,
                 feat_dims: int = 256,
                 query_h: int = 10,
                 with_box_refine: bool = True,
                 as_two_stage: bool = False,
                 **kwargs) -> None:
        self.feat_dims = feat_dims
        self.query_h = query_h
        self.query_w = int(num_queries / self.query_h)
        self.with_box_refine = with_box_refine
        # Not support as_two_stage
        self.as_two_stage = as_two_stage
        self.hybrid = hybrid

        if bbox_head is not None:
            assert 'share_pred_layer' not in bbox_head and \
                   'num_pred_layer' not in bbox_head and \
                   'as_two_stage' not in bbox_head, \
                'The two keyword args `share_pred_layer`, `num_pred_layer`, ' \
                'and `as_two_stage are set in `detector.__init__()`, users ' \
                'should not set them in `bbox_head` config.'
            # The last prediction layer is used to generate proposal
            # from encode feature map when `as_two_stage` is `True`.
            # And all the prediction layers should share parameters
            # when `with_box_refine` is `True`.
            bbox_head['share_pred_layer'] = not with_box_refine
            bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
                if self.as_two_stage else decoder['num_layers']
            bbox_head['as_two_stage'] = as_two_stage

        super(DeformableDETR, self).__init__(
            *args, decoder=decoder, bbox_head=bbox_head,
            num_queries=num_queries, **kwargs)

    def _init_layers(self) -> None:
        self.encoder = DECOEncoder(**self.encoder)
        self.decoder = DECODecoder(**self.decoder)
        if self.hybrid is not None:
            self.hybrid = MODELS.build(self.hybrid)

        self.embed_dims = self.encoder.embed_dims[-1]
        self.encoder_proj = nn.Conv2d(
            self.feat_dims, self.encoder.embed_dims[0], 1, 1)

        # NOTE The query_embedding will be split into query and query_pos
        # in self.pre_decoder, hence, the embed_dims are doubled.
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims * 2)

    def init_weights(self) -> None:
        super(DetectionTransformer, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.query_embedding.weight)

    def pre_transformer(
            self,
            mlvl_feats: List[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[dict, dict]:
        encoder_inputs_dict = dict(feat=mlvl_feats)
        decoder_inputs_dict = dict()
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_hybrid_encoder(self, feat: List[Tensor]) -> Tensor:
        feat = list(feat)
        enc_feat = feat[-1]
        enc_feat = self.encoder_proj(enc_feat)
        memory = self.encoder(feat=enc_feat)
        if self.hybrid is not None:
            feat[-1] = memory
            memory = self.hybrid(feat)[-1]
        return memory

    def forward_encoder(self, feat: List[Tensor]) -> dict:
        memory = self.forward_hybrid_encoder(feat)
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict

    def pre_decoder(
            self,
            memory: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[dict, dict]:
        bs, c, _, _ = memory.shape
        assert c == self.embed_dims
        query_embedding = self.query_embedding.weight
        # (num_queries, embed_dims) -> (num_queries, bs, embed_dims)
        query_embedding = query_embedding.unsqueeze(1).repeat(1, bs, 1)
        # (num_queries, bs, embed_dims) -> (bs, embed_dims, query_h, query_w)
        query_embedding = query_embedding.permute(1, 2, 0).view(
            bs, self.embed_dims * 2, self.query_h, self.query_w)
        tgt, query_pos = torch.split(query_embedding, c, dim=1)

        decoder_inputs_dict = dict(
            tgt=tgt,
            query_pos=query_pos,
            memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        tgt: Tensor,
                        query_pos: Tensor,
                        memory: Tensor,
                        **kwargs) -> dict:
        inter_states = self.decoder(
            tgt=tgt,
            query_pos=query_pos,
            memory=memory,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)
        decoder_outputs_dict = dict(
            hidden_states=inter_states)
        return decoder_outputs_dict
