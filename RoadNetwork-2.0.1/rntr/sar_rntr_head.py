# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
import numpy as np
import math
from mmseg.models.losses import accuracy
from mmseg.models.builder import build_loss

from .ar_rntr_head import MLP, PryDecoderEmbeddings


@MODELS.register_module()
class SARRNTRHead(AnchorFreeHead):
    """Semi-Autoregressive RoadNet Transformer Head.
    
    Implements parallel sequence decoding where the road network is decomposed
    into multiple parallel subsequences, each starting from a keypoint.
    Each subsequence maintains autoregressive dependencies internally while
    different subsequences can be decoded in parallel.
    
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_center_classes (int): Size of the vocabulary for sequence tokens.
        max_parallel_seqs (int): Maximum number of parallel subsequences (M).
        max_seq_len (int): Maximum length of each subsequence (L).
        embed_dims (int): Embedding dimensions.
        transformer (dict): Config for transformer.
        positional_encoding (dict): Config for position encoding.
        bev_positional_encoding (dict): Config for BEV position encoding.
        loss_coords (dict): Config for coordinate loss.
        loss_labels (dict): Config for label loss.
        loss_connects (dict): Config for connection loss.
        loss_coeffs (dict): Config for coefficient loss.
        **kwargs: Other arguments.
    """
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_center_classes,
                 max_parallel_seqs=16,
                 max_seq_len=128,
                 embed_dims=256,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 bev_positional_encoding=dict(
                     type='PositionEmbeddingSineBEV',
                     num_feats=128,
                     normalize=True),
                 loss_coords=dict(type='CrossEntropyLoss'),
                 loss_labels=dict(type='CrossEntropyLoss'),
                 loss_connects=dict(type='CrossEntropyLoss'),
                 loss_coeffs=dict(type='CrossEntropyLoss'),
                 bbox_coder=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.max_parallel_seqs = max_parallel_seqs
        self.max_seq_len = max_seq_len
        self.num_center_classes = num_center_classes
        
        # Token definitions (same as AR-RNTR)
        self.box_range = 200
        self.coeff_range = 200
        self.category_start = 200
        self.connect_start = 250
        self.coeff_start = 350
        self.no_known = 575
        self.start = 574
        self.end = 573
        self.noise_connect = 572
        self.noise_label = 571
        self.noise_coeff = 570
        
        super(SARRNTRHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=dict(type='mmdet.FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=1.0),
            loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
            bbox_coder=bbox_coder,
            init_cfg=init_cfg)
        
        # Build losses
        self.loss_coords = MODELS.build(loss_coords)
        self.loss_labels = MODELS.build(loss_labels)
        self.loss_connects = MODELS.build(loss_connects)
        self.loss_coeffs = MODELS.build(loss_coeffs)
        
        # Build transformer
        self.transformer = MODELS.build(transformer)
        
        # Build position encodings
        self.positional_encoding = TASK_UTILS.build(positional_encoding)
        self.bev_position_encoding = TASK_UTILS.build(bev_positional_encoding)
        
        # BEV projection if needed
        if self.in_channels != self.embed_dims:
            self.bev_proj = nn.Sequential(
                nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        
        # Embedding and output layers
        self.embedding = PryDecoderEmbeddings(
            num_center_classes, self.embed_dims, max_seq_len)
        self.vocab_embed = MLP(self.embed_dims, self.embed_dims, num_center_classes, 3)
        
        # Keypoint detection head (simple version)
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims // 2, 1, 1),
            nn.Sigmoid()
        )
        
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
    
    def init_weights(self):
        """Initialize weights of the transformer head."""
        self.transformer.init_weights()
    
    def extract_keypoints(self, bev_feats, img_metas, num_keypoints=None):
        """Extract keypoints from BEV features.
        
        Args:
            bev_feats (Tensor): BEV features [B, C, H, W].
            img_metas (list[dict]): Meta information.
            num_keypoints (int): Number of keypoints to extract per sample.
            
        Returns:
            list[Tensor]: Keypoint coordinates for each sample [N, 2].
        """
        if num_keypoints is None:
            num_keypoints = self.max_parallel_seqs
            
        B, C, H, W = bev_feats.shape
        keypoint_heatmap = self.keypoint_head(bev_feats)  # [B, 1, H, W]
        
        keypoints_list = []
        for b in range(B):
            heatmap = keypoint_heatmap[b, 0]  # [H, W]
            
            # Simple peak detection - can be improved with NMS
            flat_heatmap = heatmap.flatten()
            _, top_indices = torch.topk(flat_heatmap, min(num_keypoints, flat_heatmap.size(0)))
            
            # Convert flat indices to 2D coordinates
            y_coords = top_indices // W
            x_coords = top_indices % W
            keypoints = torch.stack([x_coords, y_coords], dim=1).float()
            
            # Pad if needed
            if keypoints.size(0) < num_keypoints:
                padding = torch.zeros(num_keypoints - keypoints.size(0), 2, 
                                    device=keypoints.device, dtype=keypoints.dtype)
                keypoints = torch.cat([keypoints, padding], dim=0)
            
            keypoints_list.append(keypoints[:num_keypoints])
        
        return keypoints_list
    
    def prepare_parallel_sequences(self, parallel_seqs_in, parallel_seqs_mask):
        """Prepare parallel sequences for transformer input.
        
        Args:
            parallel_seqs_in (Tensor): Input sequences [B, M, L].
            parallel_seqs_mask (Tensor): Sequence masks [B, M, L].
            
        Returns:
            tuple: (embedded_seqs, query_embed, seq_masks)
        """
        B, M, L = parallel_seqs_in.shape
        
        # Reshape to [B*M, L] for batch processing
        flat_seqs = parallel_seqs_in.view(B * M, L)
        flat_masks = parallel_seqs_mask.view(B * M, L)
        
        # Embed sequences
        embedded_seqs = self.embedding(flat_seqs.long())  # [B*M, L, D]
        
        # Position embeddings for sequences
        query_embed = self.embedding.position_embeddings.weight[:L]  # [L, D]
        query_embed = query_embed.unsqueeze(0).repeat(B * M, 1, 1)  # [B*M, L, D]
        
        return embedded_seqs, query_embed, flat_masks
    
    def forward(self, mlvl_feats, parallel_seqs_in=None, parallel_seqs_mask=None, img_metas=None):
        """Forward function.
        
        Args:
            mlvl_feats (Tensor): BEV features [B, C, H, W].
            parallel_seqs_in (Tensor): Input sequences [B, M, L] for training.
            parallel_seqs_mask (Tensor): Sequence masks [B, M, L] for training.
            img_metas (list[dict]): Meta information.
            
        Returns:
            Tensor or tuple: Training outputs or inference results.
        """
        x = mlvl_feats
        if self.in_channels != self.embed_dims:
            x = self.bev_proj(x)
        
        pos_embed = self.bev_position_encoding(x)  # [B, C, H, W]
        B, _, H, W = x.shape
        masks = torch.zeros(B, H, W).bool().to(x.device)
        
        if self.training and parallel_seqs_in is not None:
            # Training mode: use teacher forcing
            tgt, query_embed, seq_masks = self.prepare_parallel_sequences(
                parallel_seqs_in, parallel_seqs_mask)
            
            # Transformer forward
            outs_dec, _ = self.transformer(tgt, x, masks, query_embed, pos_embed)
            outs_dec = torch.nan_to_num(outs_dec)
            
            # Vocabulary prediction
            out = self.vocab_embed(outs_dec)  # [num_layers, B*M, L, vocab_size]
            
            # Reshape back to [num_layers, B, M, L, vocab_size]
            num_layers = out.shape[0]
            out = out.view(num_layers, B, self.max_parallel_seqs, self.max_seq_len, -1)
            
            return out
        else:
            # Inference mode: extract keypoints and generate sequences
            keypoints_list = self.extract_keypoints(x, img_metas)
            
            # Initialize sequences with keypoint tokens
            # For simplicity, start each sequence with a start token
            # In practice, you'd convert keypoints to appropriate tokens
            input_seqs = torch.full((B, self.max_parallel_seqs, 1), 
                                  self.start, device=x.device, dtype=torch.long)
            
            # Autoregressive generation for each subsequence
            max_iterations = self.max_seq_len - 1
            all_sequences = []
            
            for step in range(max_iterations):
                # Prepare current sequences
                tgt, query_embed, _ = self.prepare_parallel_sequences(
                    input_seqs, torch.ones_like(input_seqs).float())
                
                # Forward pass
                outs_dec, _ = self.transformer(tgt, x, masks, query_embed, pos_embed)
                outs_dec = torch.nan_to_num(outs_dec)[-1]  # Use last layer
                
                # Get predictions for the last position
                last_pos_features = outs_dec[:, -1, :]  # [B*M, D]
                out = self.vocab_embed(last_pos_features)  # [B*M, vocab_size]
                
                # Sample next tokens
                out = out.softmax(-1)
                _, next_tokens = out.topk(dim=-1, k=1)  # [B*M, 1]
                next_tokens = next_tokens.view(B, self.max_parallel_seqs, 1)
                
                # Append to sequences
                input_seqs = torch.cat([input_seqs, next_tokens], dim=-1)
                
                # Check for end tokens (simplified)
                if torch.all(next_tokens == self.end):
                    break
            
            return input_seqs, keypoints_list
    
    def loss_by_feat(self,
                    gt_coords_list,
                    gt_labels_list, 
                    gt_connects_list,
                    gt_coeffs_list,
                    preds_dicts,
                    gt_bboxes_ignore=None):
        """Compute losses.
        
        Args:
            gt_coords_list (list[Tensor]): Ground truth coordinates.
            gt_labels_list (list[Tensor]): Ground truth labels.
            gt_connects_list (list[Tensor]): Ground truth connections.
            gt_coeffs_list (list[Tensor]): Ground truth coefficients.
            preds_dicts (dict): Prediction dictionaries.
            gt_bboxes_ignore: Ignored bboxes.
            
        Returns:
            dict: Loss dictionary.
        """
        preds_coords = preds_dicts['preds_coords']
        preds_labels = preds_dicts['preds_labels']
        preds_connects = preds_dicts['preds_connects']
        preds_coeffs = preds_dicts['preds_coeffs']
        
        # Compute losses (similar to AR-RNTR but handle parallel sequences)
        loss_coords, loss_labels, loss_connects, loss_coeffs = multi_apply(
            self.loss_by_feat_single,
            preds_coords,
            preds_labels,
            preds_connects,
            preds_coeffs,
            gt_coords_list,
            gt_labels_list,
            gt_connects_list,
            gt_coeffs_list,
        )
        
        loss_dict = dict()
        loss_dict['loss_coords'] = loss_coords
        loss_dict['loss_labels'] = loss_labels
        loss_dict['loss_connects'] = loss_connects
        loss_dict['loss_coeffs'] = loss_coeffs
        return loss_dict
    
    def loss_by_feat_single(self,
                           preds_coords,
                           preds_labels,
                           preds_connects,
                           preds_coeffs,
                           gt_coords,
                           gt_labels,
                           gt_connects,
                           gt_coeffs):
        """Compute loss for a single sample."""
        loss_coords = self.loss_coords(preds_coords, gt_coords)
        loss_labels = self.loss_labels(preds_labels, gt_labels)
        loss_connects = self.loss_connects(preds_connects, gt_connects)
        loss_coeffs = self.loss_coeffs(preds_coeffs, gt_coeffs)
        
        loss_coords = torch.nan_to_num(loss_coords)
        loss_labels = torch.nan_to_num(loss_labels)
        loss_connects = torch.nan_to_num(loss_connects)
        loss_coeffs = torch.nan_to_num(loss_coeffs)
        
        return loss_coords, loss_labels, loss_connects, loss_coeffs
