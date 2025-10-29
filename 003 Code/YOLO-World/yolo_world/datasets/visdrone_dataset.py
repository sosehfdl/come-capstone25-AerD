# 파일명: mmyolo/datasets/yolov5_dior.py

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmdet.datasets import Objects365V1Dataset
from mmyolo.registry import DATASETS

@DATASETS.register_module()
class YoloWorldVisDroneBaseDataset(BatchShapePolicyDataset, Objects365V1Dataset):
    """Custom YOLO-World dataset class for VisDrone dataset, inheriting CocoDataset."""

    METAINFO = {
        'classes': (
            'pedestrian', 'people', 'bicycle', 'car', 'tricycle', 'awning-tricycle', 'bus'
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
@DATASETS.register_module()
class YoloWorldVisDroneNovelDataset(BatchShapePolicyDataset, Objects365V1Dataset):
    """Custom YOLO-World dataset class for VisDrone dataset, inheriting CocoDataset."""

    METAINFO = {
        'classes': (
            'van', 'truck', 'motor'
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)