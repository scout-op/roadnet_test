# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init


def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


@MODELS.register_module()
class ParallelSeqTransformerDecoderLayer(BaseTransformerLayer):
    """Transformer decoder layer for parallel sequence processing.
    
    This layer supports intra-sequence self-attention with causal masking
    while allowing parallel processing of multiple sequences.
    """
    
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(ParallelSeqTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for ParallelSeqTransformerDecoderLayer.
        
        Args:
            query (Tensor): Query embeddings [B*M, L, D].
            key (Tensor): Key embeddings from BEV features.
            value (Tensor): Value embeddings from BEV features.
            query_pos (Tensor): Query position embeddings.
            key_pos (Tensor): Key position embeddings.
            attn_masks (list): Attention masks [self_attn_mask, cross_attn_mask].
            query_key_padding_mask (Tensor): Query padding mask.
            key_padding_mask (Tensor): Key padding mask.
            
        Returns:
            Tensor: Output features [B*M, L, D].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        
        if attn_masks is None:
            attn_masks = [None, None]
        
        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[0],  # Causal mask for intra-sequence attention
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
                
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
                
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[1],  # No mask for cross attention
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
                
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        
        return query


@MODELS.register_module()
class ParallelSeqTransformer(BaseModule):
    """Parallel Sequence Transformer for SAR-RNTR.
    
    This transformer processes multiple subsequences in parallel while maintaining
    autoregressive dependencies within each subsequence.
    """
    
    def __init__(self,
                 keypoint_decoder=None,
                 parallel_seq_decoder=None,
                 init_cfg=None):
        super(ParallelSeqTransformer, self).__init__(init_cfg=init_cfg)
        
        # Keypoint decoder for initial keypoint detection/refinement
        if keypoint_decoder is not None:
            self.keypoint_decoder = build_transformer_layer_sequence(keypoint_decoder)
        else:
            self.keypoint_decoder = None
            
        # Parallel sequence decoder
        self.parallel_seq_decoder = build_transformer_layer_sequence(parallel_seq_decoder)
        self.embed_dims = self.parallel_seq_decoder.embed_dims
    
    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
    
    def forward(self,
                tgt,
                bev_features,
                bev_mask,
                query_embed,
                pos_embed,
                keypoints=None):
        """Forward function.
        
        Args:
            tgt (Tensor): Target sequences [B*M, L, D].
            bev_features (Tensor): BEV features [B, C, H, W].
            bev_mask (Tensor): BEV mask [B, H, W].
            query_embed (Tensor): Query embeddings [B*M, L, D].
            pos_embed (Tensor): Position embeddings [B, C, H, W].
            keypoints (Tensor, optional): Keypoint coordinates.
            
        Returns:
            tuple: (decoder_output, bev_memory)
        """
        B, C, H, W = bev_features.shape
        BM, L, D = tgt.shape
        M = BM // B
        
        # Prepare BEV memory
        bev_memory = bev_features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        bev_pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        bev_mask_flat = bev_mask.flatten(1)  # [B, H*W]
        
        # Expand BEV memory for parallel sequences
        # Each subsequence shares the same BEV memory
        bev_memory_expanded = bev_memory.unsqueeze(1).repeat(1, M, 1, 1).view(H*W, BM, C)
        bev_pos_expanded = bev_pos_embed.unsqueeze(1).repeat(1, M, 1, 1).view(H*W, BM, C)
        bev_mask_expanded = bev_mask_flat.unsqueeze(1).repeat(1, M, 1).view(BM, H*W)
        
        # Generate causal mask for intra-sequence attention
        causal_mask = generate_square_subsequent_mask(L).to(tgt.device)
        
        # Transpose for transformer: [L, B*M, D]
        tgt = tgt.transpose(0, 1)
        query_embed = query_embed.transpose(0, 1)
        
        # Apply parallel sequence decoder
        decoder_output = self.parallel_seq_decoder(
            query=tgt,
            key=bev_memory_expanded,
            value=bev_memory_expanded,
            key_pos=bev_pos_expanded,
            query_pos=query_embed,
            key_padding_mask=bev_mask_expanded,
            attn_masks=[causal_mask, None]  # Causal for self-attn, None for cross-attn
        )
        
        # Transpose back: [B*M, L, D]
        decoder_output = decoder_output.transpose(0, 1)
        
        return decoder_output, bev_memory


@MODELS.register_module()
class LssParallelSeqLineTransformer(BaseModule):
    """LSS-based Parallel Sequence Line Transformer.
    
    Wrapper around ParallelSeqTransformer to match the interface of existing
    LSS transformers in the codebase.
    """
    
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 init_cfg=None):
        super(LssParallelSeqLineTransformer, self).__init__(init_cfg=init_cfg)
        
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
            
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
    
    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
    
    def forward(self, tgt, x, mask, query_embed, pos_embed, query_key_padding_mask=None):
        """Forward function compatible with existing LSS transformers.
        
        Args:
            tgt (Tensor): Target sequences [B*M, L, D].
            x (Tensor): BEV features [B, C, H, W].
            mask (Tensor): BEV mask [B, H, W].
            query_embed (Tensor): Query embeddings [B*M, L, D].
            pos_embed (Tensor): Position embeddings [B, C, H, W].
            query_key_padding_mask (Tensor, optional): Padding mask for queries
                with shape [B*M, L]. True indicates positions that should be masked.
            
        Returns:
            tuple: (decoder_output, memory)
        """
        bs, c, h, w = x.shape
        bm, l, d = tgt.shape
        
        # Prepare memory
        memory = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        pos_embed_flat = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        mask_flat = mask.flatten(1)  # [B, H*W]
        
        # Expand for parallel processing
        m = bm // bs  # Number of parallel sequences
        memory_expanded = memory.unsqueeze(1).repeat(1, m, 1, 1).view(h*w, bm, c)
        pos_expanded = pos_embed_flat.unsqueeze(1).repeat(1, m, 1, 1).view(h*w, bm, c)
        mask_expanded = mask_flat.unsqueeze(1).repeat(1, m, 1).view(bm, h*w)
        
        # Generate causal mask
        causal_mask = generate_square_subsequent_mask(l).to(tgt.device)
        
        # Transpose for transformer
        tgt = tgt.transpose(0, 1)  # [L, B*M, D]
        query_embed = query_embed.transpose(0, 1)  # [L, B*M, D]
        
        # Apply decoder
        out_dec = self.decoder(
            query=tgt,
            key=memory_expanded,
            value=memory_expanded,
            key_pos=pos_expanded,
            query_pos=query_embed,
            key_padding_mask=mask_expanded,
            attn_masks=[causal_mask, None],
            query_key_padding_mask=query_key_padding_mask
        )
        
        # Make shape consistent with other transformers: [num_layers, B*M, L, D]
        out_dec = out_dec.transpose(1, 2)  # [num_layers, B*M, L, D]
        
        return out_dec, memory
