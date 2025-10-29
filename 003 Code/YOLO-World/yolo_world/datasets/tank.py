# 파일명: mmyolo/datasets/yolov5_dior.py

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from mmdet.datasets import Objects365V1Dataset
from mmyolo.registry import DATASETS

@DATASETS.register_module()
class YoloWorldTankBaseDataset(BatchShapePolicyDataset, Objects365V1Dataset):
    """Custom YOLO-World dataset class for VisDrone dataset, inheriting CocoDataset."""

    METAINFO = {
        'classes': (
            'tank', 'armoredcar', 'militarytruck'
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
@DATASETS.register_module()
class YoloWorldTankNovelDataset(BatchShapePolicyDataset, Objects365V1Dataset):
    """Custom YOLO-World dataset class for VisDrone dataset, inheriting CocoDataset."""

    METAINFO = {
        'classes': (
            'tank_k2', 'tank_t80', 'armoredcar_k200', 'armoredcar_bmp3', 'militarytruck_k311'
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)