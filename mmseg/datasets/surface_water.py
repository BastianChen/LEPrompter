import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose


@DATASETS.register_module()
class SurfaceWaterDataset(CustomDataset):
    CLASSES = ('background', 'lake')
    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, **kwargs):
        super(SurfaceWaterDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)


@DATASETS.register_module()
class PromptSurfaceWaterDataset(CustomDataset):
    CLASSES = ('background', 'lake')
    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, pipeline, pipeline_stage2=None, prompts_steps=50000,
                 prompts_dir=None, use_center_points=True, use_filled_mask=True, **kwargs):
        super(PromptSurfaceWaterDataset, self).__init__(
            pipeline=pipeline,
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
        self.pipeline1 = Compose(pipeline)
        self.pipeline2 = Compose(pipeline_stage2) if pipeline_stage2 else self.pipeline1
        self.prompts_steps = prompts_steps
        self.prompts_dir = prompts_dir
        self.use_center_points = use_center_points
        self.use_filled_mask = use_filled_mask

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['use_center_points'] = self.use_center_points
        results['use_filled_mask'] = self.use_filled_mask
        if self.prompts_dir:
            results['prompts_prefix'] = osp.join(self.data_root, self.prompts_dir)
        else:
            results['prompts_prefix'] = None
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            # print(f"********{self.prompts_steps}")
            if self.prompts_steps >= 0:
                self.prompts_steps -= 1
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline2(results) if self.prompts_steps < 0 else self.pipeline1(results)
        # return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline1(results)
        # return self.pipeline(results)
