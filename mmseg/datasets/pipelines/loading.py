# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageAndPromptFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results['prompts_prefix']:
            prompts_name = f"{results['img_info']['filename'].split('.')[0]}.png"
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
            if results['prompts_prefix']:
                point_type_name = f"center_points_{prompts_name}" if results[
                    'use_center_points'] else f"random_points_{prompts_name}"
                mask_type_name = f"filled_mask_{prompts_name}" if results[
                    'use_filled_mask'] else f"unfilled_mask_{prompts_name}"
                point_prompt_name = osp.join(results['prompts_prefix'], point_type_name)
                box_prompt_name = osp.join(results['prompts_prefix'], f"box_{prompts_name}")
                mask_prompt_name = osp.join(results['prompts_prefix'], mask_type_name)
        else:
            filename = results['img_info']['filename']
            if results['prompts_prefix']:
                point_prompt_name = f"center_points_{prompts_name}" if results[
                    'use_center_points'] else f"random_points_{prompts_name}"
                box_prompt_name = f"box_{prompts_name}"
                mask_prompt_name = f"filled_mask_{prompts_name}" if results[
                    'use_filled_mask'] else f"unfilled_mask_{prompts_name}"
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        if results['prompts_prefix']:
            point_prompt_bytes = self.file_client.get(point_prompt_name)
            box_prompt_bytes = self.file_client.get(box_prompt_name)
            mask_prompt_bytes = self.file_client.get(mask_prompt_name)
            point_prompt = mmcv.imfrombytes(point_prompt_bytes, flag='unchanged',
                                            backend=self.imdecode_backend).squeeze().astype(
                np.uint8)
            point_prompt = np.expand_dims(point_prompt, axis=-1)
            box_prompt = mmcv.imfrombytes(box_prompt_bytes, flag='unchanged',
                                          backend=self.imdecode_backend).squeeze().astype(
                np.uint8)
            box_prompt = np.expand_dims(box_prompt, axis=-1)
            mask_prompt = mmcv.imfrombytes(mask_prompt_bytes, flag='unchanged',
                                           backend=self.imdecode_backend).squeeze().astype(
                np.uint8)
            mask_prompt = np.expand_dims(mask_prompt, axis=-1)
            if (point_prompt == box_prompt).all() and (box_prompt == mask_prompt).all():
                prompts = np.zeros((256, 256, 3))
                img_prompt = np.concatenate([img, prompts], axis=-1)
            else:
                img_prompt = np.concatenate([img, point_prompt, box_prompt, mask_prompt], axis=-1)
        else:
            prompts = np.zeros((256, 256, 3))
            img_prompt = np.concatenate([img, prompts], axis=-1)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img_prompt
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


# @PIPELINES.register_module()
# class LoadImageAndPromptFromFile(object):
#     """Load an image from file.
#
#     Required keys are "img_prefix" and "img_info" (a dict that must contain the
#     key "filename"). Added or updated keys are "filename", "img", "img_shape",
#     "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
#     "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
#
#     Args:
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is an uint8 array.
#             Defaults to False.
#         color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
#             Defaults to 'color'.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'cv2'
#     """
#
#     def __init__(self,
#                  to_float32=False,
#                  color_type='color',
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='cv2'):
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend
#
#     def __call__(self, results):
#         """Call functions to load image and get image meta information.
#
#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.
#
#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """
#
#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)
#         if results['prompts_prefix']:
#             prompts_name = f"{results['img_info']['filename'].split('.')[0]}.png"
#         if results.get('img_prefix') is not None:
#             filename = osp.join(results['img_prefix'],
#                                 results['img_info']['filename'])
#             if results['prompts_prefix']:
#                 point_prompt_name = osp.join(results['prompts_prefix'], f"point_{prompts_name}")
#                 box_prompt_name = osp.join(results['prompts_prefix'], f"bbox_{prompts_name}")
#                 mask_prompt_name = osp.join(results['prompts_prefix'], f"mask_{prompts_name}")
#         else:
#             filename = results['img_info']['filename']
#             if results['prompts_prefix']:
#                 point_prompt_name = f"point_{prompts_name}"
#                 box_prompt_name = f"bbox_{prompts_name}"
#                 mask_prompt_name = f"mask_{prompts_name}"
#         img_bytes = self.file_client.get(filename)
#         img = mmcv.imfrombytes(
#             img_bytes, flag=self.color_type, backend=self.imdecode_backend)
#
#         if self.to_float32:
#             img = img.astype(np.float32)
#
#         if results['prompts_prefix']:
#             point_prompt_bytes = self.file_client.get(point_prompt_name)
#             box_prompt_bytes = self.file_client.get(box_prompt_name)
#             mask_prompt_bytes = self.file_client.get(mask_prompt_name)
#             point_prompt = mmcv.imfrombytes(point_prompt_bytes, flag='unchanged',
#                                             backend=self.imdecode_backend).squeeze().astype(
#                 np.uint8)
#             point_prompt = np.expand_dims(point_prompt, axis=-1)
#             box_prompt = mmcv.imfrombytes(box_prompt_bytes, flag='unchanged',
#                                            backend=self.imdecode_backend).squeeze().astype(
#                 np.uint8)
#             box_prompt = np.expand_dims(box_prompt, axis=-1)
#             mask_prompt = mmcv.imfrombytes(mask_prompt_bytes, flag='unchanged',
#                                            backend=self.imdecode_backend).squeeze().astype(
#                 np.uint8)
#             mask_prompt = np.expand_dims(mask_prompt, axis=-1)
#             if (point_prompt == box_prompt).all() and (box_prompt == mask_prompt).all():
#                 prompts = np.zeros((256, 256, 3))
#                 img_prompt = np.concatenate([img, prompts], axis=-1)
#             else:
#                 img_prompt = np.concatenate([img, point_prompt, box_prompt, mask_prompt], axis=-1)
#         else:
#             prompts = np.zeros((256, 256, 3))
#             img_prompt = np.concatenate([img, prompts], axis=-1)
#
#         results['filename'] = filename
#         results['ori_filename'] = results['img_info']['filename']
#         results['img'] = img_prompt
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         # Set initial values for default meta_keys
#         results['pad_shape'] = img.shape
#         results['scale_factor'] = 1.0
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results['img_norm_cfg'] = dict(
#             mean=np.zeros(num_channels, dtype=np.float32),
#             std=np.ones(num_channels, dtype=np.float32),
#             to_rgb=False)
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(to_float32={self.to_float32},'
#         repr_str += f"color_type='{self.color_type}',"
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
