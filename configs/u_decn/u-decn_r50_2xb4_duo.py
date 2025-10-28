_base_ = [
    '../_base_/datasets/duo_detection.py',
    '../_base_/default_runtime.py',
]

default_hooks = dict(
    checkpoint=dict(save_best='coco/bbox_mAP'))

# model settings
num_classes = 4
model = dict(
    type='DECN',
    num_queries=100,
    query_h=10,  # query_w = int(num_queries / query_h)
    feat_dims=2048,
    with_box_refine=True,
    as_two_stage=True,
    with_dn_training=True,
    with_u_color_query=True,
    mlvl_query_selection=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),  # [512, 1024, 2048]
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    encoder=dict(
        embed_dims=[120, 240, 480],
        depths=[2, 6, 2]),
    hybrid=dict(
        type='CrossScaleFeatureFusion',
        in_channels=[512, 1024, 480],
        mid_channels=256,
        out_channels=480,
        out_indices=(-1,),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decoder=dict(
        sim_dcn=dict(offset_scale=1.0, groups=16),
        embed_dims=480,
        query_h=10,
        query_w=10,
        num_layers=6,
        return_intermediate=True),
    bbox_head=dict(
        type='DECNHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        embed_dims=480,
        num_reg_fcs=3,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', box_format='xywh', weight=5.0),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
    test_cfg=dict(max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))
