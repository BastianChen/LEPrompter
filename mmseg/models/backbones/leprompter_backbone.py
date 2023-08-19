# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d

from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmcv.runner import (BaseModule, ModuleList, Sequential, CheckpointLoader,
                         load_state_dict)
from ..builder import BACKBONES


class DepthWiseConvModule(BaseModule):
    """An implementation of one Depth-wise Conv Module of LEPrompter.

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


class PositionEmbeddingRandom(BaseModule):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats=64, scale=None):
        super(PositionEmbeddingRandom, self).__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords_input, image_size):
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone().to(torch.float)
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


@BACKBONES.register_module()
class LEPrompter(BaseModule):
    def __init__(
            self,
            embed_dim,
            input_image_size,
            mask_in_chans,
            point_nums=1,
            kernel_size=7,
            stride=4,
            padding=3,
            ffn_drop=0.,
            act_cfg=dict(type='GELU'),
    ):
        """
        Encodes prompts for input to LEPrompter's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          point_nums (int): the number of points in the point prompt.
          kernel_size (int): The kernel size of Conv2d. Default: 3.
          stride (int): The stride of Conv2d. Default: 2.
          padding (int): The padding of Conv2d. Default: 0.
          ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super(LEPrompter, self).__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.point_nums = point_nums
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        num_point_embeddings = 3 + point_nums  # 1 no point + n point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(num_point_embeddings)]
        self.point_embeddings = ModuleList(point_embeddings)

        self.mask_downscaling = DepthWiseConvModule(embed_dims=1,
                                                    feedforward_channels=mask_in_chans // 4,
                                                    output_channels=embed_dim,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    act_cfg=act_cfg,
                                                    ffn_drop=ffn_drop)

    def pre_process_prompts(self, bs, points=None, boxes=None, masks=None):
        if points is not None:
            points_indices_temp = torch.nonzero(points)

            # Get the coordinates of the first point_nums points.
            grouped_tensor = [points_indices_temp[points_indices_temp[:, 0] == i] for i in range(bs)]
            selected_points = [grouped_tensor[i][:self.point_nums] for i in range(bs)]
            selected_points = torch.cat(selected_points, dim=0)

            points_indices = torch.zeros(len(selected_points) // self.point_nums, self.point_nums * 2 + 1,
                                         device=selected_points.device).to(torch.long)
            selected_indices_list = torch.unique(selected_points[:, 0])
            # [[0,100,100],[0,105,105],[0,110,110],[1,105,105],[1,110,110],[1,115,115]]
            # --> [[0,100,100,105,105,110,110],[1,105,105,110,110,115,115]]
            points_indices[:, 0] = selected_indices_list
            points_indices[:, 1:] = torch.cat([selected_points[selected_points[:, 0] == i][:, 1:]
                                               for i in selected_indices_list]).reshape(-1, self.point_nums * 2)
            points = self.get_prompts(points_indices, bs, size=[self.point_nums * 2])
        if boxes is not None:
            boxes_indices_temp = torch.nonzero(boxes)
            boxes_indices = torch.zeros(len(boxes_indices_temp) // 2, 5, device=boxes_indices_temp.device).to(
                torch.long)
            # [[0,100,100],[0,105,105],[1,100,100],[1,105,105]] --> [[0,100,100,105,105],[1,100,100,105,105]]
            boxes_indices[:, 0] = boxes_indices_temp[::2, 0]
            boxes_indices[:, 1:3] = boxes_indices_temp[::2, 1:]
            boxes_indices[:, 3:] = boxes_indices_temp[1::2, 1:]
            boxes = self.get_prompts(boxes_indices, bs, size=[4])

        if masks is not None:
            sum_masks = torch.sum(torch.sum(masks, dim=1), dim=1)
            masks_indices = torch.nonzero(sum_masks)
            masks = self.get_prompts(masks_indices, bs, size=self.input_image_size, isMask=True, mask=masks)
        return points, boxes, masks

    def get_prompts(self, indices, bs, size, isMask=False, mask=None):
        data = torch.full((bs, *size), -1, dtype=torch.long).to(indices.device)
        if indices.numel() > 0:
            batch_indices = indices[:, 0]
            if not isMask:
                data[batch_indices, :] = indices[:, 1:]
            else:
                data = data.to(torch.float)
                data[batch_indices] = mask[batch_indices]
        return data

    def _embed_points(self, points):
        """Embeds point prompts."""
        points = points.reshape(-1, self.point_nums, 2)
        point_embedding = self.pe_layer(points, self.input_image_size)
        point_embedding[torch.nonzero(points == -1)[:, 0].unique()] += self.point_embeddings[0].weight
        for i in range(self.point_nums):
            point_embedding[torch.nonzero(points >= 0)[:, 0].unique(), i] += self.point_embeddings[i + 1].weight
        return point_embedding

    def _embed_boxes(self, boxes):
        """Embeds box prompts."""
        boxes = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer(boxes, self.input_image_size)
        corner_embedding[torch.nonzero(boxes == -1)[:, 0].unique()] += self.point_embeddings[0].weight
        corner_embedding[torch.nonzero(boxes[:, 0] >= 0)[:, 0].unique(), 0] += self.point_embeddings[
            self.point_nums + 1].weight
        corner_embedding[torch.nonzero(boxes[:, 1] >= 0)[:, 0].unique(), 1] += self.point_embeddings[
            self.point_nums + 2].weight
        return corner_embedding

    def _embed_masks(self, masks):
        """Embeds mask inputs."""
        masks = masks.reshape(masks.shape[0], 1, *masks.shape[1:])
        masks = (masks - 128.0) / 127.0
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def forward(self, bs, device, points=None, boxes=None, masks=None):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          bs: batch size
          device: device id
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates to embed
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        points, boxes, masks = self.pre_process_prompts(bs, points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device)
        dense_embeddings = torch.empty((bs, 0, self.input_image_size[0], self.input_image_size[1]), device=device)
        if points is not None:
            point_embeddings = self._embed_points(points)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        return sparse_embeddings, dense_embeddings
