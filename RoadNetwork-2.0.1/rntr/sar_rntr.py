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
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet.models import DETECTORS
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


@MODELS.register_module()
class SAR_RNTR(MVXTwoStageDetector):
    """Semi-Autoregressive Road Network Transformer.
    
    This model implements the SAR-RNTR approach where road networks are
    decomposed into parallel subsequences for efficient generation while
    maintaining autoregressive dependencies within each subsequence.
    
    Args:
        pts_voxel_layer (dict, optional): Config of voxelization layer.
        pts_voxel_encoder (dict, optional): Config of voxel encoder.
        pts_middle_encoder (dict, optional): Config of middle encoder.
        pts_fusion_layer (dict, optional): Config of fusion layer.
        img_backbone (dict, optional): Config of image backbone.
        pts_backbone (dict, optional): Config of point cloud backbone.
        img_neck (dict, optional): Config of image neck.
        pts_neck (dict, optional): Config of point cloud neck.
        pts_bbox_head (dict, optional): Config of bbox head.
        img_roi_head (dict, optional): Config of roi head.
        img_rpn_head (dict, optional): Config of rpn head.
        train_cfg (dict, optional): Config of training.
        test_cfg (dict, optional): Config of testing.
        pretrained (str, optional): Path of pretrained model.
        init_cfg (dict, optional): Config of initialization.
    """
    
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None):
        
        super(SAR_RNTR, self).__init__(
            pts_voxel_layer=pts_voxel_layer,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.fp16_enabled = False
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # img is List[Tensor] or Tensor
            if isinstance(img, list):
                img = torch.stack(img, dim=0)
            
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            
            if self.with_img_backbone and img is not None:
                img_feats = self.img_backbone(img)
                if isinstance(img_feats, dict):
                    img_feats = list(img_feats.values())
            else:
                return None
            
            if self.with_img_neck:
                img_feats = self.img_neck(img_feats)
            
            img_feats_reshaped = []
            for img_feat in img_feats:
                BN, C, H, W = img_feat.size()
                if len_queue is not None:
                    img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B * len_queue), C, H, W))
                else:
                    img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
            return img_feats_reshaped
        else:
            return None
    
    def extract_feat(self, img, img_metas, len_queue=None):
        """Extract features from images and point clouds."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue)
        return img_feats
    
    def forward_pts_train(self,
                         pts_feats,
                         parallel_seqs_in,
                         parallel_seqs_tgt,
                         parallel_seqs_mask,
                         img_metas,
                         **kwargs):
        """Forward function for point cloud branch in training.
        
        Args:
            pts_feats (list[Tensor]): Features of point cloud branch.
            parallel_seqs_in (Tensor): Input parallel sequences [B, M, L].
            parallel_seqs_tgt (Tensor): Target parallel sequences [B, M, L].
            parallel_seqs_mask (Tensor): Sequence masks [B, M, L].
            img_metas (list[dict]): Meta information of samples.
            
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, parallel_seqs_in, parallel_seqs_mask, img_metas)
        
        # Parse predictions from transformer output
        preds_dicts = self.parse_predictions(outs, parallel_seqs_tgt, parallel_seqs_mask)
        
        # Compute losses
        loss_inputs = [
            preds_dicts['gt_coords_list'],
            preds_dicts['gt_labels_list'],
            preds_dicts['gt_connects_list'],
            preds_dicts['gt_coeffs_list'],
            preds_dicts,
        ]
        losses = self.pts_bbox_head.loss_by_feat(*loss_inputs)
        return losses
    
    def parse_predictions(self, outs, parallel_seqs_tgt, parallel_seqs_mask):
        """Parse predictions from transformer output.
        
        Args:
            outs (Tensor): Transformer output [num_layers, B, M, L, vocab_size].
            parallel_seqs_tgt (Tensor): Target sequences [B, M, L].
            parallel_seqs_mask (Tensor): Sequence masks [B, M, L].
            
        Returns:
            dict: Parsed predictions and ground truth.
        """
        num_layers, B, M, L, vocab_size = outs.shape
        
        # Use last layer predictions
        preds = outs[-1]  # [B, M, L, vocab_size]
        
        # Split vocabulary predictions
        box_range = 200
        category_start = 200
        connect_start = 250
        coeff_start = 350
        
        # Extract different components
        preds_coords_x = preds[..., :box_range]  # [B, M, L, 200]
        preds_coords_y = preds[..., box_range:box_range*2]  # [B, M, L, 200]
        preds_labels = preds[..., category_start:connect_start]  # [B, M, L, 50]
        preds_connects = preds[..., connect_start:coeff_start]  # [B, M, L, 100]
        preds_coeffs_x = preds[..., coeff_start:coeff_start+110]  # [B, M, L, 110]
        preds_coeffs_y = preds[..., coeff_start+110:vocab_size]  # [B, M, L, 110]
        
        # Combine coordinate predictions
        preds_coords = torch.stack([preds_coords_x, preds_coords_y], dim=-1)  # [B, M, L, 2, 200]
        preds_coeffs = torch.stack([preds_coeffs_x, preds_coeffs_y], dim=-1)  # [B, M, L, 2, 110]
        
        # Parse ground truth targets
        gt_coords_list = []
        gt_labels_list = []
        gt_connects_list = []
        gt_coeffs_list = []
        
        for b in range(B):
            # Extract valid sequences for this batch
            batch_mask = parallel_seqs_mask[b]  # [M, L]
            batch_tgt = parallel_seqs_tgt[b]  # [M, L]
            
            # Parse each sequence
            for m in range(M):
                seq_mask = batch_mask[m]  # [L]
                seq_tgt = batch_tgt[m]  # [L]
                
                # Only process valid tokens
                valid_indices = seq_mask.bool()
                if valid_indices.sum() == 0:
                    continue
                
                valid_tokens = seq_tgt[valid_indices]  # [valid_len]
                
                # Parse tokens into components (simplified)
                # In practice, you'd need proper token parsing logic
                gt_coords = torch.zeros(valid_tokens.size(0), 2, dtype=torch.long, device=valid_tokens.device)
                gt_labels = torch.zeros(valid_tokens.size(0), dtype=torch.long, device=valid_tokens.device)
                gt_connects = torch.zeros(valid_tokens.size(0), dtype=torch.long, device=valid_tokens.device)
                gt_coeffs = torch.zeros(valid_tokens.size(0), 2, dtype=torch.long, device=valid_tokens.device)
                
                gt_coords_list.append(gt_coords)
                gt_labels_list.append(gt_labels)
                gt_connects_list.append(gt_connects)
                gt_coeffs_list.append(gt_coeffs)
        
        return {
            'preds_coords': preds_coords,
            'preds_labels': preds_labels,
            'preds_connects': preds_connects,
            'preds_coeffs': preds_coeffs,
            'gt_coords_list': gt_coords_list,
            'gt_labels_list': gt_labels_list,
            'gt_connects_list': gt_connects_list,
            'gt_coeffs_list': gt_coeffs_list,
        }
    
    def forward_train(self,
                     points=None,
                     img_metas=None,
                     gt_bboxes_3d=None,
                     gt_labels_3d=None,
                     gt_labels=None,
                     gt_bboxes=None,
                     img=None,
                     proposals=None,
                     gt_bboxes_ignore=None,
                     img_depth=None,
                     img_mask=None,
                     **kwargs):
        """Forward training function.
        
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
            img_metas (list[dict], optional): Meta information of samples.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional): Ground truth 3D boxes.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels of 3D boxes.
            gt_labels (list[torch.Tensor], optional): Ground truth labels of 2D boxes.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes.
            img (torch.Tensor, optional): Images of each sample with shape (N, C, H, W).
            proposals ([list[torch.Tensor], optional): Predicted proposals used for training Fast RCNN.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth 2D boxes to be ignored.
            img_depth (torch.Tensor, optional): Depth maps.
            img_mask (torch.Tensor, optional): Image masks.
            
        Returns:
            dict: Losses of different branches.
        """
        len_queue = img.size(1) if img is not None and img.dim() == 5 else 1
        prev_img = img[:, :-len_queue, ...] if img is not None and img.dim() == 5 else None
        img = img[:, -len_queue:, ...] if img is not None and img.dim() == 5 else img
        
        img_feats = self.extract_feat(img=img, img_metas=img_metas, len_queue=len_queue)
        losses = dict()
        
        # Extract parallel sequence data from kwargs
        parallel_seqs_in = kwargs.get('parallel_seqs_in', None)
        parallel_seqs_tgt = kwargs.get('parallel_seqs_tgt', None)
        parallel_seqs_mask = kwargs.get('parallel_seqs_mask', None)
        
        if parallel_seqs_in is not None and parallel_seqs_tgt is not None:
            losses_pts = self.forward_pts_train(
                img_feats, parallel_seqs_in, parallel_seqs_tgt, 
                parallel_seqs_mask, img_metas, **kwargs)
            losses.update(losses_pts)
        
        return losses
    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs, keypoints_list = self.pts_bbox_head(x, img_metas=img_metas)
        
        # Convert sequences back to road network format
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, keypoints_list, img_metas, rescale=rescale)
        
        bbox_results = [
            dict(pts_bbox=bboxes) for bboxes in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentation."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict.update(pts_bbox)
        
        return bbox_list
