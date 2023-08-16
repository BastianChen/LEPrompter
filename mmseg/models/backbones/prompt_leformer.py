# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torchvision.transforms as transforms
from typing import Sequence
from mmcv.cnn import Conv2d
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import (BaseModule, ModuleList, Sequential, CheckpointLoader,
                         load_state_dict)

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from .mit import MixFFN
from .le_prompter_backbone import LEPrompter


class DepthWiseConvModule(BaseModule):
    """An implementation of one Depth-wise Conv Module of LEFormer.

    Args:
        embed_dims (int): The feature dimension.
        feedforward_channels (int): The hidden dimension for FFNs.
        output_channels (int): The output channles of each cnn encoder layer.
        kernel_size (int): The kernel size of Conv2d. Default: 3.
        stride (int): The stride of Conv2d. Default: 2.
        padding (int): The padding of Conv2d. Default: 1.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default: 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(DepthWiseConvModule, self).__init__(init_cfg)
        self.activate = build_activation_layer(act_cfg)
        fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Pooling(nn.Module):
    """Pooling module.

    Args:
        pool_size (int): Pooling size. Defaults: 3.
    """

    def __init__(self, pool_size=3):
        super(Pooling, self).__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolingBlock(BaseModule):
    """Pooling Block.

    Args:
        embed_dims (int): The feature dimension.
        pool_size (int): Pooling size. Defaults to 3.
        mlp_ratio (float): Mlp expansion ratio. Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='GN', num_groups=1)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-5.
    """

    def __init__(self,
                 embed_dims,
                 pool_size=3,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.,
                 layer_scale_init_value=1e-5):
        super(PoolingBlock, self).__init__()

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dims,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((embed_dims)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of LEFormer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of LEFormer. Default: 1.
        pool_size (int): Pooling size. Default: 3.
        pool (bool): Whether to use Pooling Transformer Layer. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 pool_size=3,
                 pool=False
                 ):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.pool = pool
        if not self.pool:
            self.sr_ratio = sr_ratio

            if sr_ratio > 1:
                self.sr = Conv2d(
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=sr_ratio,
                    stride=sr_ratio)
                # The ret[0] of build_norm_layer is norm name.
                self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.pool_former_block = PoolingBlock(
                embed_dims=embed_dims,
                pool_size=pool_size
            )

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):
        if self.pool:
            out = nlc_to_nchw(x, hw_shape)
            out = self.pool_former_block(out)
            out = nchw_to_nlc(out)
            return out
        else:
            x_q = x
            if identity is None:
                identity = x_q
            if self.sr_ratio > 1:
                x_kv = nlc_to_nchw(x, hw_shape)
                x_kv = self.sr(x_kv)
                x_kv = nchw_to_nlc(x_kv)
                x_kv = self.norm(x_kv)
            else:
                x_kv = x

            # Because the dataflow('key', 'query', 'value') of
            # ``torch.nn.MultiheadAttention`` is (num_query, batch,
            # embed_dims), We should adjust the shape of dataflow from
            # batch_first (batch, num_query, embed_dims) to num_query_first
            # (num_query ,batch, embed_dims), and recover ``attn_output``
            # from num_query_first to batch_first.
            if self.batch_first:
                x_q = x_q.transpose(0, 1)
                x_kv = x_kv.transpose(0, 1)

            out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

            if self.batch_first:
                out = out.transpose(0, 1)

            return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one transformer encoder layer in LEFormer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of LEFormer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pool_size (int): Pooling size. Default: 3.
        pool (bool): Whether to use Pooling Transformer Layer. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False,
                 pool_size=3,
                 pool=False
                 ):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio,
            pool_size=pool_size,
            pool=pool
        )

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.pool = pool
        if not self.pool:
            self.ffn = MixFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            if not self.pool:
                x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class ChannelAttentionModule(BaseModule):
    """An implementation of one Channel Attention Module of LEFormer.

        Args:
            embed_dims (int): The embedding dimension.
    """

    def __init__(self, embed_dims):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            Conv2d(embed_dims, embed_dims // 4, 1, bias=False),
            nn.ReLU(),
            Conv2d(embed_dims // 4, embed_dims, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(BaseModule):
    """An implementation of one Spatial Attention Module of LEFormer.

        Args:
            kernel_size (int): The kernel size of Conv2d. Default: 3.
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttentionModule, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiscaleCBAMLayer(BaseModule):
    """An implementation of Multiscale CBAM layer of LEFormer.

        Args:
            embed_dims (int): The feature dimension.
            kernel_size (int): The kernel size of Conv2d. Default: 7.
        """

    def __init__(self, embed_dims, kernel_size=7):
        super(MultiscaleCBAMLayer, self).__init__()
        self.channel_attention = ChannelAttentionModule(embed_dims // 4)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
        self.multiscale_conv = ModuleList()
        for i in range(1, 5):
            self.multiscale_conv.append(
                Conv2d(
                    in_channels=embed_dims // 4,
                    out_channels=embed_dims // 4,
                    kernel_size=3,
                    stride=1,
                    padding=(2 * i + 1) // 2,
                    bias=True,
                    dilation=(2 * i + 1) // 2)
            )

    def forward(self, x):
        outs = torch.split(x, x.shape[1] // 4, dim=1)
        out_list = []
        for (i, out) in enumerate(outs):
            out = self.multiscale_conv[i](out)
            out = self.channel_attention(out) * out
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = self.spatial_attention(out) * out
        return out


class CnnEncoderLayer(BaseModule):
    """Implements one cnn encoder layer in LEFormer.

        Args:
            embed_dims (int): The feature dimension.
            feedforward_channels (int): The hidden dimension for FFNs.
            output_channels (int): The output channles of each cnn encoder layer.
            kernel_size (int): The kernel size of Conv2d. Default: 3.
            stride (int): The stride of Conv2d. Default: 2.
            padding (int): The padding of Conv2d. Default: 0.
            act_cfg (dict): The activation config for FFNs.
                Default: dict(type='GELU').
            ffn_drop (float, optional): Probability of an element to be
                zeroed in FFN. Default 0.0.
            init_cfg (dict, optional): Initialization config dict.
                Default: None.
        """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.):
        super(CnnEncoderLayer, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.output_channels = output_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        self.layers = DepthWiseConvModule(embed_dims=embed_dims,
                                          feedforward_channels=feedforward_channels // 2,
                                          output_channels=output_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          act_cfg=dict(type='GELU'),
                                          ffn_drop=ffn_drop)

        self.multiscale_cbam = MultiscaleCBAMLayer(output_channels, kernel_size)

    def forward(self, x):
        out = self.layers(x)
        out = self.multiscale_cbam(out)
        return out


# class PromptEncoderLayer(BaseModule):
#     def __init__(
#             self,
#             embed_dim,
#             # image_embedding_size,
#             input_image_size,
#             mask_in_chans,
#             point_nums=1,
#             kernel_size=7,
#             stride=4,
#             padding=3,
#             ffn_drop=0.,
#             act_cfg=dict(type='GELU'),
#     ):
#         """
#         Encodes prompts for input to SAM's mask decoder.
#
#         Arguments:
#           embed_dim (int): The prompts' embedding dimension
#           image_embedding_size (tuple(int, int)): The spatial size of the
#             image embedding, as (H, W).
#           input_image_size (int): The padded size of the image as input
#             to the image encoder, as (H, W).
#           mask_in_chans (int): The number of hidden channels used for
#             encoding input masks.
#           activation (nn.Module): The activation to use when encoding
#             input masks.
#         """
#         super(PromptEncoderLayer, self).__init__()
#         self.embed_dim = embed_dim
#         self.input_image_size = input_image_size
#         self.point_nums = point_nums
#         self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
#
#         num_point_embeddings = 3 + point_nums  # 1 no point + n point + 2 box corners
#         point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(num_point_embeddings)]
#         self.point_embeddings = ModuleList(point_embeddings)
#
#         self.mask_downscaling = DepthWiseConvModule(embed_dims=1,
#                                                     feedforward_channels=mask_in_chans // 4,
#                                                     output_channels=embed_dim,
#                                                     kernel_size=kernel_size,
#                                                     stride=stride,
#                                                     padding=padding,
#                                                     act_cfg=act_cfg,
#                                                     ffn_drop=ffn_drop)
#
#     def pre_process_prompts(self, bs, points=None, boxes=None, masks=None):
#         if points is not None:
#             points_indices_temp = torch.nonzero(points)
#
#             # Get the coordinates of the first point_nums points.
#             grouped_tensor = [points_indices_temp[points_indices_temp[:, 0] == i] for i in range(bs)]
#             selected_points = [grouped_tensor[i][:self.point_nums] for i in range(bs)]
#             selected_points = torch.cat(selected_points, dim=0)
#
#             points_indices = torch.zeros(len(selected_points) // self.point_nums, self.point_nums * 2 + 1,
#                                          device=selected_points.device).to(torch.long)
#             selected_indices_list = torch.unique(selected_points[:, 0])
#             # [[0,100,100],[0,105,105],[0,110,110],[1,105,105],[1,110,110],[1,115,115]]
#             # --> [[0,100,100,105,105,110,110],[1,105,105,110,110,115,115]]
#             points_indices[:, 0] = selected_indices_list
#             points_indices[:, 1:] = torch.cat([selected_points[selected_points[:, 0] == i][:, 1:]
#                                                for i in selected_indices_list]).reshape(-1, self.point_nums * 2)
#             points = self.get_prompts(points_indices, bs, size=[self.point_nums * 2])
#         if boxes is not None:
#             boxes_indices_temp = torch.nonzero(boxes)
#             boxes_indices = torch.zeros(len(boxes_indices_temp) // 2, 5, device=boxes_indices_temp.device).to(
#                 torch.long)
#             # [[0,100,100],[0,105,105],[1,100,100],[1,105,105]] --> [[0,100,100,105,105],[1,100,100,105,105]]
#             boxes_indices[:, 0] = boxes_indices_temp[::2, 0]
#             boxes_indices[:, 1:3] = boxes_indices_temp[::2, 1:]
#             boxes_indices[:, 3:] = boxes_indices_temp[1::2, 1:]
#             boxes = self.get_prompts(boxes_indices, bs, size=[4])
#
#         if masks is not None:
#             sum_masks = torch.sum(torch.sum(masks, dim=1), dim=1)
#             masks_indices = torch.nonzero(sum_masks)
#             masks = self.get_prompts(masks_indices, bs, size=self.input_image_size, isMask=True, mask=masks)
#         return points, boxes, masks
#
#     def get_prompts(self, indices, bs, size, isMask=False, mask=None):
#         data = torch.full((bs, *size), -1, dtype=torch.long).to(indices.device)
#         if indices.numel() > 0:
#             batch_indices = indices[:, 0]
#             if not isMask:
#                 data[batch_indices, :] = indices[:, 1:]
#             else:
#                 data = data.to(torch.float)
#                 data[batch_indices] = mask[batch_indices]
#         return data
#
#     def _embed_points(self, points):
#         """Embeds point prompts."""
#         points = points.reshape(-1, self.point_nums, 2)
#         point_embedding = self.pe_layer(points, self.input_image_size)
#         point_embedding[torch.nonzero(points == -1)[:, 0].unique()] += self.point_embeddings[0].weight
#         for i in range(self.point_nums):
#             point_embedding[torch.nonzero(points >= 0)[:, 0].unique(), i] += self.point_embeddings[i + 1].weight
#         return point_embedding
#
#     def _embed_boxes(self, boxes):
#         """Embeds box prompts."""
#         boxes = boxes.reshape(-1, 2, 2)
#         corner_embedding = self.pe_layer(boxes, self.input_image_size)
#         corner_embedding[torch.nonzero(boxes == -1)[:, 0].unique()] += self.point_embeddings[0].weight
#         corner_embedding[torch.nonzero(boxes[:, 0] >= 0)[:, 0].unique(), 0] += self.point_embeddings[
#             self.point_nums + 1].weight
#         corner_embedding[torch.nonzero(boxes[:, 1] >= 0)[:, 0].unique(), 1] += self.point_embeddings[
#             self.point_nums + 2].weight
#         return corner_embedding
#
#     def _embed_masks(self, masks):
#         """Embeds mask inputs."""
#         masks = masks.reshape(masks.shape[0], 1, *masks.shape[1:])
#         # print(f"1================{masks}")
#         masks = (masks - 128.0) / 127.0
#         # normalize = transforms.Normalize(mean=[0.5], std=[0.5])
#         # masks = masks / 255.0
#         # print(f"2================{masks}")
#         # masks = normalize(masks)
#         # print(f"3================{masks}")
#         mask_embedding = self.mask_downscaling(masks)
#         return mask_embedding
#
#     def forward(self, bs, device, points=None, boxes=None, masks=None):
#         """
#         Embeds different types of prompts, returning both sparse and dense
#         embeddings.
#
#         Arguments:
#           points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates to embed
#           boxes (torch.Tensor or none): boxes to embed
#           masks (torch.Tensor or none): masks to embed
#
#         Returns:
#           torch.Tensor: sparse embeddings for the points and boxes, with shape
#             BxNx(embed_dim), where N is determined by the number of input points
#             and boxes.
#           torch.Tensor: dense embeddings for the masks, in the shape
#             Bx(embed_dim)x(embed_H)x(embed_W)
#         """
#         points, boxes, masks = self.pre_process_prompts(bs, points, boxes, masks)
#         sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device)
#         dense_embeddings = torch.empty((bs, 0, self.input_image_size[0], self.input_image_size[1]), device=device)
#         if points is not None:
#             point_embeddings = self._embed_points(points)
#             sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
#         if boxes is not None:
#             box_embeddings = self._embed_boxes(boxes)
#             sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
#         if masks is not None:
#             dense_embeddings = self._embed_masks(masks)
#         return sparse_embeddings, dense_embeddings
#
#
# class PositionEmbeddingRandom(BaseModule):
#     """
#     Positional encoding using random spatial frequencies.
#     """
#
#     def __init__(self, num_pos_feats=64, scale=None):
#         super(PositionEmbeddingRandom, self).__init__()
#         if scale is None or scale <= 0.0:
#             scale = 1.0
#         self.register_buffer(
#             "positional_encoding_gaussian_matrix",
#             scale * torch.randn((2, num_pos_feats)),
#         )
#
#     def _pe_encoding(self, coords):
#         """Positionally encode points that are normalized to [0,1]."""
#         # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
#         coords = 2 * coords - 1
#         coords = coords @ self.positional_encoding_gaussian_matrix
#         coords = 2 * torch.pi * coords
#         # outputs d_1 x ... x d_n x C shape
#         return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
#
#     def forward(self, coords_input, image_size):
#         """Positionally encode points that are not normalized to [0,1]."""
#         coords = coords_input.clone().to(torch.float)
#         coords[:, :, 0] = coords[:, :, 0] / image_size[1]
#         coords[:, :, 1] = coords[:, :, 1] / image_size[0]
#         return self._pe_encoding(coords.to(torch.float))  # B x N x C


@BACKBONES.register_module()
class PromptLEFormer(BaseModule):
    """The backbone of LEFormer.

    This backbone is the implementation of `LEFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 32.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [2, 2, 2, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 6].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        pool_numbers (int): the number of Pooling Transformer Layers. Default 1.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=32,
                 num_stages=4,
                 num_layers=(2, 2, 2, 3),
                 num_heads=(1, 2, 5, 6),
                 patch_sizes=(7, 3, 3, 3),
                 strides=(4, 2, 2, 2),
                 sr_ratios=(8, 4, 2, 1),
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 pool_numbers=1,
                 image_size=(256, 256),
                 use_prompts=[True, True, True],
                 prompts_steps=0,
                 point_nums=1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super(PromptLEFormer, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        embed_dims_list = []
        feedforward_channels_list = []
        self.transformer_encoder_layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            embed_dims_list.append(embed_dims_i)
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            feedforward_channels_list.append(mlp_ratio * embed_dims_i)
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i],
                    pool=i < pool_numbers
                ) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.transformer_encoder_layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

        self.cnn_encoder_layers = ModuleList()
        self.fusion_conv_layers = ModuleList()

        for i in range(num_stages):
            self.cnn_encoder_layers.append(
                CnnEncoderLayer(
                    embed_dims=self.in_channels if i == 0 else embed_dims_list[i - 1],
                    feedforward_channels=feedforward_channels_list[i],
                    output_channels=embed_dims_list[i],
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding=patch_sizes[i] // 2,
                    ffn_drop=drop_rate
                )
            )
            self.fusion_conv_layers.append(
                Conv2d(
                    in_channels=embed_dims_list[i] * 2,
                    out_channels=embed_dims_list[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
            )

        # self.prompts_enconder_layer = PromptEncoderLayer(embed_dims_list[-1], image_size, embed_dims_list[0],
        #                                                  point_nums=point_nums)

        if self.training:
            self.le_prompter = LEPrompter(embed_dims_list[-1], image_size, embed_dims_list[0],
                                          point_nums=point_nums)
            self.use_prompts = use_prompts
            self.prompts_steps = prompts_steps

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(PromptLEFormer, self).init_weights()

    def forward(self, x):
        # print(f"==============={self.prompts_steps}")
        # import cv2
        # path = "/home/ubuntu/datasets/leformer/SW/prompts_new/training"
        # point = cv2.imread(f"{path}/center_points_training_11013.png", 0)
        # point = torch.unsqueeze(torch.unsqueeze(torch.Tensor(point), 0), 0).to(x.device)
        # box = cv2.imread(f"{path}/box_training_11013.png", 0)
        # box = torch.unsqueeze(torch.unsqueeze(torch.Tensor(box), 0), 0).to(x.device)
        # mask = cv2.imread(f"{path}/mask_training_11013.png", 0)
        # mask = torch.unsqueeze(torch.unsqueeze(torch.Tensor(mask), 0), 0).to(x.device)
        # x = torch.cat((x, point, box, mask), dim=1)
        # self.training = True

        if self.prompts_steps < 0:
            self.training = False
        elif self.prompts_steps >= 0 and self.training:
            self.prompts_steps -= 1
        if self.training and True in self.use_prompts:
            bs = x.shape[0]
            device = x.device
            points, boxes, masks = None, None, None
            if self.use_prompts[0]:
                points = x[:, 3]
            if self.use_prompts[1]:
                boxes = x[:, 4]
            if self.use_prompts[2]:
                masks = x[:, 5]
            sparse_embeddings, dense_embeddings = self.le_prompter(bs, device, points, boxes, masks)
        x = x[:, :3]
        outs = []
        cnn_encoder_out = x

        for i, (cnn_encoder_layer, transformer_encoder_layer) in enumerate(
                zip(self.cnn_encoder_layers, self.transformer_encoder_layers)):
            cnn_encoder_out = cnn_encoder_layer(cnn_encoder_out)
            x, hw_shape = transformer_encoder_layer[0](x)
            for block in transformer_encoder_layer[1]:
                x = block(x, hw_shape)
            x = transformer_encoder_layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            x = torch.cat((x, cnn_encoder_out), dim=1)
            x = self.fusion_conv_layers[i](x)

            if i in self.out_indices:
                if self.training and True in self.use_prompts and i == self.out_indices[-1]:
                    x = [x, sparse_embeddings, dense_embeddings]
                outs.append(x)
        return outs
