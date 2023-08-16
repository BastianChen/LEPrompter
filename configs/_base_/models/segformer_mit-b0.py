# # model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# prompts_steps = 50000
# use_center_points = False  # use center_points or random_points
# use_filled_mask = True  # use filled_mask or unfilled_mask
# use_prompts = [True, False, False]  # use points, boxes and masks
# model = dict(
#     type='EncoderDecoder',
#     pretrained=None,
#     backbone=dict(
#         type='PromptMixVisionTransformer',
#         in_channels=3,
#         embed_dims=32,
#         num_stages=4,
#         num_layers=[2, 2, 2, 2],
#         num_heads=[1, 2, 5, 8],
#         patch_sizes=[7, 3, 3, 3],
#         sr_ratios=[8, 4, 2, 1],
#         out_indices=(0, 1, 2, 3),
#         mlp_ratio=4,
#         qkv_bias=True,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.1,
#         image_size=[256, 256],
#         use_prompts=use_prompts,
#         prompts_steps=prompts_steps,
#         point_nums=3
#     ),
#     decode_head=dict(
#         type='PromptLEFormerHead',
#         in_channels=[32, 64, 160, 256],
#         in_index=[0, 1, 2, 3],
#         channels=256,
#         dropout_ratio=0.1,
#         num_classes=2,
#         depth=2,
#         embedding_dim=256,
#         num_heads=4,
#         mlp_dim=384,
#         prompts_steps=prompts_steps,
#         use_prompts=use_prompts,
#         sr_ratio=8,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))