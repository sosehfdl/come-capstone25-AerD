_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)


# hyper-parameters
num_classes = 3
num_training_classes = 3 # 
max_epochs = 20 # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 32
load_from = '/mnt/d/py/AIM/Projects/Drone_detection/OVD/YOLO-World/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackboneV2',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]

# no_mosaic_pipeline=[
#     *_base_.pre_transform,
#     dict(
#         type='MultiModalMosaic',
#         img_scale=_base_.img_scale,
#         pad_val=114.0,
#         prob=0.0   # ← prob=0이면 아예 실행 안 함
#     ),
#     dict(
#         type='YOLOv5RandomAffine',
#         max_rotate_degree=0.0,
#         max_shear_degree=0.0,
#         scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
#         max_aspect_ratio=_base_.max_aspect_ratio,
#         border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
#         border_val=(114, 114, 114),
#         ),
#     *_base_.last_transform[:-1],
#     *text_transform
# ]
    
# train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]

train_pipeline_stage2 = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform
]

train_detection_dataset1 = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YoloWorldTankBaseDataset',
        data_root='/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset/1_1_1',
        ann_file='v1_modified_0328.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/tank/tank_json/tank_base_text.json',
    pipeline=train_pipeline)

train_detection_dataset2 = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YoloWorldTankBaseDataset',
        data_root='/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset/1_1_2',
        ann_file='v2_modified_0328.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/tank/tank_json/tank_base_text.json',
    pipeline=train_pipeline)

train_detection_dataset3 = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YoloWorldTankBaseDataset',
        data_root='/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset/1_1_3',
        ann_file='v3_modified_0328.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/tank/tank_json/tank_base_text.json',
    pipeline=train_pipeline)

train_detection_dataset4 = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YoloWorldTankBaseDataset',
        data_root='/mnt/d/py/AIM/Projects/Drone_detection/Image_Fusion/photo-background-generation/dataset/dataset/dataset2',
        ann_file='tank_armoredcar_militarytruck_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/tank/tank_json/tank_base_text.json',
    pipeline=train_pipeline)

train_detection_dataset5 = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YoloWorldTankBaseDataset',
        data_root='/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset_datamaker/train_dataset/OD_2',
        ann_file='OD_2_update.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/tank/tank_json/tank_base_text.json',
    pipeline=train_pipeline)

# train_grounding_dataset1 = dict(
#     type='YOLOv5MixedGroundingDataset',
#     data_root='/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset_datamaker/train_dataset/OVD_1',
#     ann_file='OVD_1_caption_A_updated.json',
#     data_prefix=dict(img='images/'),
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        num_workers = 0,
                        persistent_workers=False,
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                        #  train_detection_dataset1,
                                        #  train_detection_dataset2,
                                        #  train_detection_dataset3,
                                        #  train_detection_dataset4,
                                         train_detection_dataset5,
                                        #  train_grounding_dataset1
                                     ],
                                     ignore_keys=['classes', 'palette']))

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

valid_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YoloWorldTankBaseDataset',
        data_root='/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset_datamaker/test_dataset',
        test_mode=True,
        ann_file='russian_test.json',
        data_prefix=dict(img='russian_imgs/'),
        batch_shapes_cfg=None),
    class_text_path='data/tank/tank_json/tank_base_text.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=valid_dataset, num_workers=0, persistent_workers=False)
test_dataloader = val_dataloader

val_evaluator = dict(type='mmdet.CocoMetric',
                     ann_file='/mnt/d/py/AIM/Projects/Drone_detection/OVD/dataset_datamaker/test_dataset/russian_test.json',
                     metric='bbox',
                     classwise=True)
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=10,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')
