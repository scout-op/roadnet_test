_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]

# Global
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
custom_imports = dict(imports=['projects.RoadNetwork.rntr'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.

model = dict(
    type='SAR_RNTR',
    use_grid_mask=True,
    video_test_mode=True,
    # LSS/BEV settings
    lss_cfg=dict(downsample=8, d_in=_dim_, d_out=_dim_),
    grid_conf=dict(
        xbound=[-48.0, 48.0, 0.5],
        ybound=[-32.0, 32.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0],),
    data_aug_conf=dict(
        final_dim=(128, 352),
        H=900, W=1600,
        bot_pct_lim=(0.0, 0.0),
        rot_lim=(0.0, 0.0),
        rand_flip=False,
    ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='CPFPN',  # use CPFPN
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        num_outs=_num_levels_),
    pts_bbox_head=dict(
        type='SARRNTRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        num_center_classes=576,
        max_parallel_seqs=16,
        max_seq_len=128,
        embed_dims=_dim_,
        transformer=dict(
            type='LssParallelSeqLineTransformer',
            decoder=dict(
                type='TransformerLayerSequence',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='ParallelSeqTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='RNTR2MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type='RNTR2MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_num_fcs=2,
                    ffn_dropout=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                )
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10
        ), 
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_pos_dim_,
            normalize=True,
            offset=-0.5
        ),
        bev_positional_encoding=dict(
            type='PositionEmbeddingSineBEV',
            num_feats=_pos_dim_,
            normalize=True
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_coords=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_labels=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_connects=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_coeffs=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range
            )
        )
    )
)

dataset_type = 'CenterlineNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


ida_aug_conf = {
    "resize_lim": (0.193, 0.225),
    "final_dim": (128, 352),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

grid_conf = dict(
    xbound=[-48.0, 48.0, 0.5],
    ybound=[-32.0, 32.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 48.0, 1.0],)
bz_grid_conf = dict(
    xbound=[-55.0, 55.0, 0.5],
    ybound=[-55.0, 55.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 48.0, 1.0],)

train_pipeline = [
    dict(type='OrgLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf),
    dict(type='CenterlineFlip', prob=0.5),
    dict(type='CenterlineRotateScale', prob=0.5, max_rotate_degree=22.5, scaling_ratio_range=(0.95, 1.05)),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(type='ParallelSeqTransform',
         max_parallel_seqs=16,
         max_seq_len=128,
         keypoint_strategy='intersection',
         min_seq_len=3,
         noise_ratio=0.1),
    dict(type='Pack3DDetInputs', 
         keys=['img'], 
         meta_keys=('filename', 'token', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 'centerline_label', 
                'centerline_connect', 'centerline_coeff', 'centerline_sequence', 'lidar2ego', 'n_control',
                'parallel_seqs_in', 'parallel_seqs_tgt', 'parallel_seqs_mask')),
]

test_pipeline = [
    dict(type='OrgLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(type='Pack3DDetInputs', keys=['img'], 
         meta_keys=('filename', 'token', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 'centerline_label', 
                'centerline_connect', 'centerline_coeff', 'centerline_sequence', 'lidar2ego', 'n_control'))
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_centerline_infos_pon_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_centerline_infos_pon_val.pkl',
             pipeline=test_pipeline, classes=class_names, modality=input_modality, samples_per_gpu=1,
             bev_size=(bev_h_, bev_w_),
             queue_length=queue_length,),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_centerline_infos_pon_val.pkl',
              pipeline=test_pipeline, classes=class_names, modality=input_modality,
              bev_size=(bev_h_, bev_w_),
              queue_length=queue_length,),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=6, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r50_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
