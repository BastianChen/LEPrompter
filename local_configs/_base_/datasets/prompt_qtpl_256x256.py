from local_configs.leprompter.leprompter_256x256_qtpl_160k import train_step1_steps
from local_configs._base_.models.leprompter import use_center_points as ucp
from local_configs._base_.models.leprompter import use_filled_mask as ufm

# dataset settings
dataset_type = 'PromptQTPLDataset'
data_root = 'data/QTPL'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 0, 0, 0],
    std=[58.395, 57.12, 57.375, 1, 1, 1],
    to_rgb=False)
crop_size = (256, 256)
train_step2_pipeline = [
    dict(type='LoadImageAndPromptFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
train_step1_pipeline = [
    dict(type='LoadImageAndPromptFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageAndPromptFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            prompts_steps=train_step1_steps,
            use_center_points=ucp,
            use_filled_mask=ufm,
            img_dir='images/training',
            ann_dir='binary_annotations/training',
            prompts_dir='prompts/training',
            pipeline=train_step1_pipeline,
            pipeline_stage2=train_step2_pipeline,
        )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='binary_annotations/validation',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='binary_annotations/validation',
        pipeline=test_pipeline,
    ))
