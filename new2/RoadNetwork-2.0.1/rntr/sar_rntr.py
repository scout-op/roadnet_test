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
from .LiftSplatShoot import LiftSplatShootEgo


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
                 init_cfg=None,
                 lss_cfg=None,
                 grid_conf=None,
                 data_aug_conf=None):
        
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
        # Initialize LSS BEV projection modules
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **(lss_cfg or {}))
        self.downsample = (lss_cfg or {}).get('downsample', 8)
        self.final_dim = (data_aug_conf or {}).get('final_dim', (128, 352))
    
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
        """Extract BEV features from multi-view images."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue)
        # Select proper level according to AR implementation
        largest_feat_shape = img_feats[0].shape[3]
        down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
        bev_feats = self.view_transformers(img_feats[down_level], img_metas)
        return bev_feats
    
    def forward_pts_train(self,
                         bev_feats,
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
        outs = self.pts_bbox_head(bev_feats, parallel_seqs_in, parallel_seqs_mask, img_metas)
        
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
        preds = outs[-1]  # [B, M, L, V]

        # Constants for vocab slicing
        box_range = 200
        category_start = 200
        connect_start = 250
        coeff_start = 350
        coeff_split = 110  # split 220 into 110 + 110

        # Flatten over batch and parallel sequences
        BM = B * M
        preds_flat = preds.view(BM, L, vocab_size)
        tgt_flat = parallel_seqs_tgt.view(BM, L)
        mask_flat = parallel_seqs_mask.view(BM, L).bool()

        # Position groups in 6-int units: 0: vx, 1: vy, 2: vc, 3: vd, 4: epx, 5: epy
        pos_idx = torch.arange(L, device=preds.device).unsqueeze(0).expand(BM, L)
        valid = mask_flat

        def gather_group(logit_slice, target_tokens, select_mask):
            """Gather logits and targets by boolean mask; adjust targets by slice offset."""
            logits = logit_slice[select_mask]
            targets = target_tokens[select_mask]
            return logits, targets

        # vx
        sel_vx = valid & (pos_idx % 6 == 0)
        logits_vx = preds_flat[..., :box_range]
        gt_vx = tgt_flat
        preds_vx, target_vx = gather_group(logits_vx, gt_vx, sel_vx)

        # vy
        sel_vy = valid & (pos_idx % 6 == 1)
        logits_vy = preds_flat[..., :box_range]
        gt_vy = tgt_flat
        preds_vy, target_vy = gather_group(logits_vy, gt_vy, sel_vy)

        # coords stacked
        if preds_vx.numel() > 0 and preds_vy.numel() > 0:
            preds_coords = torch.cat([preds_vx, preds_vy], dim=0)
            gt_coords = torch.cat([target_vx, target_vy], dim=0)
        elif preds_vx.numel() > 0:
            preds_coords, gt_coords = preds_vx, target_vx
        else:
            preds_coords, gt_coords = preds_vy, target_vy

        # labels (vc)
        sel_vc = valid & (pos_idx % 6 == 2)
        logits_vc = preds_flat[..., category_start:connect_start]
        gt_vc = (tgt_flat - category_start).clamp(min=0, max=connect_start - category_start - 1)
        preds_labels, gt_labels = gather_group(logits_vc, gt_vc, sel_vc)

        # connects (vd)
        sel_vd = valid & (pos_idx % 6 == 3)
        logits_vd = preds_flat[..., connect_start:coeff_start]
        gt_vd = (tgt_flat - connect_start).clamp(min=0, max=coeff_start - connect_start - 1)
        preds_connects, gt_connects = gather_group(logits_vd, gt_vd, sel_vd)

        # coeffs (epx, epy)
        sel_epx = valid & (pos_idx % 6 == 4)
        logits_epx = preds_flat[..., coeff_start:coeff_start + coeff_split]
        gt_epx = (tgt_flat - coeff_start).clamp(min=0, max=coeff_split - 1)
        preds_epx, gt_epx = gather_group(logits_epx, gt_epx, sel_epx)

        sel_epy = valid & (pos_idx % 6 == 5)
        logits_epy = preds_flat[..., coeff_start + coeff_split:]
        gt_epy = (tgt_flat - (coeff_start + coeff_split)).clamp(min=0, max=coeff_split - 1)
        preds_epy, gt_epy = gather_group(logits_epy, gt_epy, sel_epy)

        if preds_epx.numel() > 0 and preds_epy.numel() > 0:
            preds_coeffs = torch.cat([preds_epx, preds_epy], dim=0)
            gt_coeffs = torch.cat([gt_epx, gt_epy], dim=0)
        elif preds_epx.numel() > 0:
            preds_coeffs, gt_coeffs = preds_epx, gt_epx
        else:
            preds_coeffs, gt_coeffs = preds_epy, gt_epy

        # Package as lists for loss_by_feat
        gt_coords_list = [gt_coords.long()]
        gt_labels_list = [gt_labels.long()]
        gt_connects_list = [gt_connects.long()]
        gt_coeffs_list = [gt_coeffs.long()]

        preds_dict = {
            'preds_coords': [preds_coords],
            'preds_labels': [preds_labels],
            'preds_connects': [preds_connects],
            'preds_coeffs': [preds_coeffs],
            'gt_coords_list': gt_coords_list,
            'gt_labels_list': gt_labels_list,
            'gt_connects_list': gt_connects_list,
            'gt_coeffs_list': gt_coeffs_list,
        }
        return preds_dict
    
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
        
        bev_feats = self.extract_feat(img=img, img_metas=img_metas, len_queue=len_queue)
        losses = dict()
        
        # Extract parallel sequence data from kwargs
        parallel_seqs_in = kwargs.get('parallel_seqs_in', None)
        parallel_seqs_tgt = kwargs.get('parallel_seqs_tgt', None)
        parallel_seqs_mask = kwargs.get('parallel_seqs_mask', None)
        
        if parallel_seqs_in is not None and parallel_seqs_tgt is not None:
            losses_pts = self.forward_pts_train(
                bev_feats, parallel_seqs_in, parallel_seqs_tgt, 
                parallel_seqs_mask, img_metas, **kwargs)
            losses.update(losses_pts)
        
        return losses
    
    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs, keypoints_list = self.pts_bbox_head(x, img_metas=img_metas)
        
        # Convert sequences back to road network format
        line_results = self.pts_bbox_head.get_bboxes(
            outs, keypoints_list, img_metas, rescale=rescale)
        return line_results
    
    def simple_test(self, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentation."""
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        results = [dict() for _ in range(len(img_metas))]
        line_results = self.simple_test_pts(bev_feats, img_metas, rescale)
        for result_dict, lr, img_meta in zip(results, line_results, img_metas):
            result_dict['line_results'] = lr
            if 'token' in img_meta:
                result_dict['token'] = img_meta['token']
        
        return results

    def loss(self, inputs=None, data_samples=None, **kwargs):
        """Compute SAR-RNTR losses using BEV features and parallel sequences.
        
        Args:
            inputs (dict): Contains 'img' tensor.
            data_samples (list): List of Det3DDataSample with metainfo.
        Returns:
            dict: Loss dict.
        """
        assert inputs is not None and data_samples is not None
        img = inputs['img']
        img_metas = [ds.metainfo for ds in data_samples]

        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()

        # Gather parallel sequence tensors from metainfo
        def to_tensor_batch(key):
            vals = [m[key] for m in img_metas if key in m]
            if len(vals) == 0:
                return None
            if not isinstance(vals[0], torch.Tensor):
                arr = np.stack(vals, axis=0)
                return torch.as_tensor(arr, device=bev_feats.device)
            else:
                return torch.stack(vals, dim=0).to(bev_feats.device)

        parallel_seqs_in = to_tensor_batch('parallel_seqs_in')
        parallel_seqs_tgt = to_tensor_batch('parallel_seqs_tgt')
        parallel_seqs_mask = to_tensor_batch('parallel_seqs_mask')

        if parallel_seqs_in is not None and parallel_seqs_tgt is not None:
            losses_pts = self.forward_pts_train(
                bev_feats,
                parallel_seqs_in,
                parallel_seqs_tgt,
                parallel_seqs_mask if parallel_seqs_mask is not None else torch.ones_like(parallel_seqs_in, dtype=torch.float32),
                img_metas,
                **kwargs,
            )
            losses.update(losses_pts)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward of testing (single-image path)."""
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_imgs = batch_inputs_dict['img']
        # keep only first sample as in AR_RNTR
        return self.simple_test(batch_input_metas[:1], img=batch_input_imgs[:1])
